"""统一调用层：模型无关。

provider.type：
- api / local：OpenAI 兼容协议（云端 API 或 Ollama / LM Studio）
- mlx：Apple Silicon 进程内推理（mlx-lm），不起本地服务

切换调用方式只改 config.yaml 的 provider 段，本模块对外仍是 chat()。
"""

import base64
import importlib
import json
import os
import queue
import re
import threading
import time
from typing import List, Optional

from .metrics import logger, metrics

try:
    from openai import OpenAI
except ImportError:  # 允许先导入包再补依赖
    OpenAI = None


class _MlxExecutor:
    """专用推理线程：MLX Stream 绑定创建线程，HTTP worker 里直接 generate 会报
    'There is no Stream(gpu, N) in current thread'。load + generate 全部收拢到此线程。

    引擎分两类：
    - "lm"：纯文本，mlx_lm.load → (model, tokenizer)
    - "vlm"：多模态（图像/音频/视频），mlx_vlm.load → (model, processor)
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._engines = {}  # model_path -> ("lm"|"vlm", model, processor)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="mlx-infer", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30)

    def _loop(self):
        import mlx.core as mx

        try:
            self._stream = mx.new_stream(mx.gpu)
        except Exception:
            self._stream = None
        self._ready.set()
        while True:
            job = self._q.get()
            if job is None:
                break
            try:
                job["fn_result"] = job["fn"]()
            except Exception as exc:
                job["fn_error"] = exc
            job["done"].set()

    def run(self, fn, timeout=None):
        job = {"fn": fn, "done": threading.Event()}
        self._q.put(job)
        if not job["done"].wait(timeout):
            raise TimeoutError("MLX inference thread busy/timeout")
        if "fn_error" in job:
            raise job["fn_error"]
        return job.get("fn_result")

    def _ensure_stream(self):
        import mlx.core as mx

        if self._stream is not None:
            mx.set_default_stream(self._stream)

    def load(self, model_path, multimodal=False):
        def _do():
            with self._lock:
                if model_path in self._engines:
                    return self._engines[model_path]
                self._ensure_stream()
                t0 = time.perf_counter()
                try:
                    if multimodal:
                        from mlx_vlm import load as vlm_load

                        engine = ("vlm",) + vlm_load(model_path)
                    else:
                        from mlx_lm import load

                        engine = ("lm",) + load(model_path)
                except Exception as exc:
                    metrics.inc("mlx_load_errors")
                    metrics.emit("mlx_load_error", model=model_path, error=str(exc))
                    logger.error("mlx load failed model=%s err=%s", model_path, exc)
                    raise RuntimeError(
                        f"MLX 模型加载失败（检查 config.yaml 的 provider.model）：{exc}"
                    ) from exc
                load_ms = (time.perf_counter() - t0) * 1000.0
                self._engines[model_path] = engine
                metrics.observe_ms("mlx_load", load_ms)
                metrics.set_gauge("mlx_loaded", 1)
                metrics.emit("mlx_loaded", model=model_path, load_ms=round(load_ms, 1))
                logger.info("mlx loaded model=%s kind=%s in %.0fms", model_path,
                            engine[0], load_ms)
                return engine

        return self.run(_do)

    def clear(self):
        """清空引擎缓存（切换模型时释放显存/内存）。"""
        with self._lock:
            self._engines.clear()

    def generate(self, model_path, prompt, max_tokens, temperature, images=None):
        """images: list[路径]，仅 vlm 引擎使用。images 与 prompt 中占位符一一对应。"""

        def _do():
            self._ensure_stream()
            engine = self._engines.get(model_path)
            if engine is None:
                raise RuntimeError(f"model not loaded: {model_path}")
            kind, model, processor = engine

            parts = []
            last = None
            t0 = time.perf_counter()
            if kind == "vlm":
                from mlx_vlm import generate as vlm_generate
                from mlx_vlm.sample_utils import make_sampler

                imgs = [img for img in (images or [])]
                last = vlm_generate(
                    model,
                    processor,
                    prompt=prompt,
                    image=imgs or None,
                    max_tokens=max_tokens,
                    sampler=make_sampler(temp=temperature, top_p=0.95),
                    verbose=False,
                )
                parts.append(last.text or "")
            else:
                from mlx_lm import stream_generate
                from mlx_lm.sample_utils import make_sampler

                for response in stream_generate(
                    model,
                    processor,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=make_sampler(temp=temperature),
                ):
                    parts.append(response.text)
                    last = response
            gen_ms = (time.perf_counter() - t0) * 1000.0
            return "".join(parts), last, gen_ms, kind

        return self.run(_do, timeout=max(60.0, max_tokens * 0.5))


_mlx_executor: _MlxExecutor | None = None
_mlx_exec_lock = threading.Lock()


def get_mlx_executor() -> _MlxExecutor:
    global _mlx_executor
    with _mlx_exec_lock:
        if _mlx_executor is None:
            _mlx_executor = _MlxExecutor()
        return _mlx_executor


# ── 模型切换器：注册表 + 全局现行模型 ───────────────────────────────
_ACTIVE_MODEL: str = ""
_MODEL_REGISTRY: List[str] = []
_DEFAULT_MODEL_DIR = "/Users/dango_studio/.lmstudio/models/lmstudio-community"


def _looks_like_model_dir(path) -> bool:
    """目录内含 config.json 且带权重文件（safetensors/npz/gguf）即视为一个模型。"""
    try:
        if not os.path.isdir(path):
            return False
        if not os.path.isfile(os.path.join(path, "config.json")):
            return False
        for name in os.listdir(path):
            if name.endswith((".safetensors", ".npz", ".gguf")):
                return True
    except OSError:
        return False
    return False


def init_model_registry(model_dir: str = "") -> List[str]:
    """扫描 model_dir 下两级目录，收集含权重的模型目录。"""
    global _MODEL_REGISTRY
    _MODEL_REGISTRY = []
    root = (model_dir or _DEFAULT_MODEL_DIR).strip()
    if not root or not os.path.isdir(root):
        return _MODEL_REGISTRY
    for sub in sorted(os.listdir(root)):
        p = os.path.join(root, sub)
        if _looks_like_model_dir(p):
            _MODEL_REGISTRY.append(p)
        elif os.path.isdir(p):
            # 组织目录再往下一层（如 lmstudio-community/{model}）
            for sub2 in sorted(os.listdir(p)):
                p2 = os.path.join(p, sub2)
                if _looks_like_model_dir(p2):
                    _MODEL_REGISTRY.append(p2)
    # 去重保序
    seen, out = set(), []
    for m in _MODEL_REGISTRY:
        if m not in seen:
            seen.add(m)
            out.append(m)
    _MODEL_REGISTRY = out
    return _MODEL_REGISTRY


def get_model_registry() -> List[str]:
    return list(_MODEL_REGISTRY)


def get_active_model() -> str:
    return _ACTIVE_MODEL


def set_active_model(path: str) -> str:
    """切换全局现行模型（mlx 场景清空引擎缓存，下次请求惰性加载释放内存）。"""
    global _ACTIVE_MODEL
    path = (path or "").strip()
    if not path:
        raise ValueError("model 不能为空")
    _ACTIVE_MODEL = path
    try:
        get_mlx_executor().clear()
    except Exception:
        pass
    return _ACTIVE_MODEL


def resolve_model_switch(path: str) -> str:
    """把用户输入解析成可切换的模型路径/名称。"""
    path = (path or "").strip()
    if not path:
        raise ValueError("model 不能为空")
    reg = get_model_registry()
    for m in reg:
        if path in (m, os.path.basename(m)):
            return m
    if _looks_like_model_dir(path):
        return path
    return path  # api/local 场景：直接作为模型名放行


def active_vision() -> bool:
    if not _ACTIVE_MODEL:
        return False
    return detect_multimodal(_ACTIVE_MODEL, "auto")


def detect_multimodal(model_path, setting="auto") -> bool:
    """判断某 MLX 模型是否走多模态引擎（mlx_vlm）。

    - setting: True 强制多模态；False 强制纯文本；"auto" 按权重试探。
    - auto 判据：config.json 含 image_token_id / vision_config / audio_config，
      mlx_vlm 能加载该 model_type，且权重里含 vision_tower。
    """
    if setting is True or str(setting).lower() == "true":
        return True
    try:
        if str(setting).lower() in ("false", "none", "0"):
            return False
    except Exception:
        return False

    try:
        importlib.import_module("mlx_vlm")
    except ImportError:
        return False

    try:
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False

    # 显式模态标记最为可靠
    if any(k in cfg for k in ("image_token_id", "vision_config",
                              "audio_config", "video_token_id")):
        return True

    return False


# Gemma4 等 channel 模型：思考段 <|channel>…<channel|>，其后才是正文
_THINK_MARK = "<|channel>"
_THINK_END = "<channel|>"
_TURN_MARKS = ("<turn|>", "<|turn>", "<eos>")


def _strip_mlx_reply(text):
    """去掉 thought channel / 残留控制符，只留回复正文。

    与该模型 chat_template 的 strip_thinking 宏一致：
    按 <channel|> 切开，含 <|channel> 的段是思考，丢弃；其余段保留。
    """
    if not text:
        return ""
    parts = text.split(_THINK_END)
    kept = []
    for part in parts:
        if _THINK_MARK in part:
            kept.append(part.split(_THINK_MARK, 1)[0])
        else:
            kept.append(part)
    out = "".join(kept)
    for mark in _TURN_MARKS:
        out = out.replace(mark, "")
    return out.strip()


def _strip_vlm_reply(text):
    """Qwen3.5 等多模态模型：去残留思考段/控制符，只留正文。

    思考段以 "thinking\n…\nresponse\n" 分隔；仅当文本以 thinking 开头，
    且出现 response 段头时才裁剪。enable_thinking=false 时输出本就很干净。
    """
    if not text:
        return ""
    text = _strip_mlx_reply(text)
    lower = text.lstrip().lower()
    if lower.startswith("thinking"):
        m = re.search(r"\bresponse\s*\n", text)
        if m:
            text = text[m.end():]
        else:
            # 只有思考段头、未见正文——认为思考截断，按空处理（上层兜底）
            text = ""
    return text.strip()


class LLMClient:
    def __init__(self, config):
        provider = config["provider"]
        self.provider_type = (provider.get("type") or "local").lower()
        self.model = provider["model"]
        self.temperature = float(provider.get("temperature", 0.85))
        self.max_tokens = int(provider.get("max_tokens", 1024))
        # mlx + 支持 thinking 的模型：对话默认关思考（省 token、避免只吐思考截断成空回复）
        self.enable_thinking = bool(provider.get("enable_thinking", False))
        self.multimodal_setting = provider.get("multimodal", "auto")

        if self.provider_type == "mlx":
            self.client = None
            self.multimodal = detect_multimodal(self.model, self.multimodal_setting)
            get_mlx_executor().load(self.model, multimodal=self.multimodal)
            return

        self.multimodal = True  # OpenAI 兼容服务端透传内容块，由服务端决定是否支持
        self._engine = None
        base_url = provider["base_url"]
        api_key = provider.get("api_key", "")
        if api_key.startswith("${") and api_key.endswith("}"):
            api_key = os.environ.get(api_key[2:-1], "")

        if OpenAI is None:
            raise RuntimeError("缺少依赖，请先执行：pip install -r requirements.txt")
        self.client = OpenAI(base_url=base_url, api_key=api_key or "ollama")

    def chat(self, messages, temperature=None, images=None,
             enable_thinking=None):
        """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]

        images: 附加到本轮（最后一条 user 消息）的图片，
            每项 {"mime": str, "data": base64 str}。仅本参数非空时启用多模态。
        """
        temp = self.temperature if temperature is None else temperature
        model = get_active_model() or self.model
        if model != self.model:  # 模型切换器改动了现行模型，跟随同步
            self.model = model
            if self.provider_type == "mlx":
                self.multimodal = detect_multimodal(model, self.multimodal_setting)
        t0 = time.perf_counter()
        metrics.inc("llm_calls")
        try:
            if self.provider_type == "mlx":
                reply = self._chat_mlx(messages, temp, images=images,
                                       enable_thinking=enable_thinking)
            else:
                msgs = self._openai_messages(messages, images)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=temp,
                )
                reply = resp.choices[0].message.content
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            metrics.observe_ms("llm_chat", elapsed_ms)
            metrics.emit(
                "llm_chat",
                provider=self.provider_type,
                model=self.model,
                ms=round(elapsed_ms, 1),
                reply_chars=len(reply or ""),
                multimodal=bool(images),
            )
            return reply
        except RuntimeError:
            metrics.inc("llm_errors")
            metrics.emit("llm_error", provider=self.provider_type, model=self.model)
            logger.warning("llm error provider=%s model=%s", self.provider_type, self.model)
            raise
        except Exception as exc:
            metrics.inc("llm_errors")
            metrics.emit("llm_error", provider=self.provider_type, error=str(exc))
            raise RuntimeError(f"LLM 调用失败（检查 config.yaml 的 provider 段）：{exc}") from exc

    @staticmethod
    def _openai_messages(messages, images):
        """将 images 转为 OpenAI 多模态 content 块，挂到最后一条 user 消息。"""
        if not images:
            return messages
        msgs = list(messages)
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                content = [{"type": "text", "text": msgs[i].get("content", "")}]
                for img in images:
                    url = f"data:{img['mime']};base64,{img['data']}"
                    content.insert(0, {
                        "type": "image_url",
                        "image_url": {"url": url},
                    })
                msgs[i] = {**msgs[i], "content": content}
                break
        return msgs

    def _render_mlx_prompt(self, engine, messages, images, enable_thinking=None):
        """按引擎类型把消息序列渲染成 prompt。images: list[路径] 仅传计数。"""
        kind, model, processor = engine
        thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        if kind == "vlm":
            from mlx_vlm.prompt_utils import apply_chat_template as vlm_template

            return vlm_template(
                processor,
                model.config,
                prompt=messages,
                num_images=len(images or []),
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=thinking,
            )
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=thinking,
        )

    def _chat_mlx(self, messages, temperature, images=None, enable_thinking=None):
        executor = get_mlx_executor()
        image_paths = None
        tmp_files = []
        if images:
            if not self.multimodal:
                raise RuntimeError(
                    "当前模型不支持多模态（provider.multimodal 未开启），"
                    "上传的图片无法识别"
                )
            # mlx_vlm 只接受文件路径（BytesIO 会崩），base64 落盘到临时文件
            import tempfile

            for img in images:
                fd, path = tempfile.mkstemp(suffix=".img")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(base64.b64decode(img["data"]))
                tmp_files.append(path)
            if tmp_files:
                image_paths = tmp_files

        # chat template / 生成都收拢到推理线程（纯 CPU tokenize 那步也在此线程更稳）
        model_path = self.model  # 已在 chat() 同步为现行模型
        if executor._engines.get(model_path) is None:
            executor.load(model_path, multimodal=self.multimodal)

        def build_prompt():
            engine = executor._engines[model_path]
            return self._render_mlx_prompt(engine, messages, image_paths,
                                           enable_thinking=enable_thinking)

        prompt = executor.run(build_prompt)

        try:
            raw, last, gen_ms, kind = executor.generate(
                model_path, prompt, self.max_tokens, temperature,
                images=image_paths,
            )
        finally:
            for p in tmp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        reply = _strip_mlx_reply(raw) if kind == "lm" else _strip_vlm_reply(raw)

        prompt_tokens = getattr(last, "prompt_tokens", None)
        gen_tokens = getattr(last, "generation_tokens", None)
        prompt_tps = getattr(last, "prompt_tps", None)
        gen_tps = getattr(last, "generation_tps", None)
        peak_gb = getattr(last, "peak_memory", None)
        metrics.observe_ms("mlx_generate", gen_ms)
        if gen_tps is not None:
            metrics.set_gauge("mlx_gen_tps", round(float(gen_tps), 2))
        if prompt_tps is not None:
            metrics.set_gauge("mlx_prompt_tps", round(float(prompt_tps), 2))
        if peak_gb is not None:
            metrics.set_gauge("mlx_peak_mem_gb", round(float(peak_gb), 2))
        if prompt_tokens is not None:
            metrics.inc("mlx_prompt_tokens", int(prompt_tokens))
        if gen_tokens is not None:
            metrics.inc("mlx_gen_tokens", int(gen_tokens))
        metrics.emit(
            "mlx_generate",
            ms=round(gen_ms, 1),
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            gen_tps=round(float(gen_tps), 2) if gen_tps else None,
            peak_gb=round(float(peak_gb), 2) if peak_gb else None,
            reply_chars=len(reply),
        )

        if not reply:
            metrics.inc("mlx_empty_replies")
            metrics.emit("mlx_empty_reply", max_tokens=self.max_tokens, raw_head=raw[:120])
            raise RuntimeError(
                f"MLX 回复为空（max_tokens={self.max_tokens} 可能被思考耗尽；"
                f"可调大或保持 enable_thinking=false）。raw={raw[:200]!r}"
            )
        return reply
