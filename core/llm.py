"""统一调用层：模型无关。

provider.type：
- api / local：OpenAI 兼容协议（云端 API 或 Ollama / LM Studio）
- openvino：Intel 进程内推理（openvino-genai，CPU/iGPU/NPU），不起本地服务

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


# ── OpenVINO：Intel 进程内推理（openvino-genai）────────────────────────
# 目录识别：Optimum 导出的 OpenVINO IR 目录
# - 纯文本：openvino_model.xml（单文件式）或 openvino_language_model.xml（分图式）
# - 多模态：额外带 openvino_vision_embeddings_model.xml（SigLIP/CLIP 视觉塔）
_OV_WEIGHTS = ("openvino_model.xml", "openvino_language_model.xml")
_OV_VISION = "openvino_vision_embeddings_model.xml"


def is_openvino_model_dir(path) -> bool:
    """path 是否为 OpenVINO IR 模型目录。"""
    try:
        if not os.path.isdir(path):
            return False
        return any(os.path.isfile(os.path.join(path, name)) for name in _OV_WEIGHTS)
    except OSError:
        return False


def ov_has_vision(path) -> bool:
    """OpenVINO IR 目录是否带视觉塔（能走 VLMPipeline）。"""
    try:
        return os.path.isfile(os.path.join(path, _OV_VISION))
    except OSError:
        return False


class _OpenVINOExecutor:
    """专用推理线程：load/generate 串行执行。

    openvino-genai 的 pipeline 不是线程安全的，且首次 load 会编译/缓存 IR，
    收拢到一条线程既避免并发冲突，也让 /metrics 的耗时统计口径一致。

    引擎分两类：
    - "lm"：纯文本，openvino_genai.LLMPipeline
    - "vlm"：带视觉塔，openvino_genai.VLMPipeline（图片转 ov.Tensor 后喂入）
    """

    def __init__(self, device: str = "CPU"):
        self.device = (device or "CPU").upper()
        self._q: queue.Queue = queue.Queue()
        self._engines = {}  # model_path -> {"kind": "lm"|"vlm", "pipe": pipeline, "device": str}
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="ov-infer", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30)

    def _loop(self):
        # 不在启动阶段 import openvino：导入失败应在 load() 里报出可读错误，
        # 而不是让线程静默死掉、构造函数干等 30 秒。
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
            raise TimeoutError("OpenVINO inference thread busy/timeout")
        if "fn_error" in job:
            raise job["fn_error"]
        return job.get("fn_result")

    @staticmethod
    def _genai():
        try:
            import openvino_genai as genai
        except Exception as exc:  # ImportError + DLL 缺失都归到这一类
            raise RuntimeError(
                "缺少依赖/运行库，请先执行：uv add openvino-genai"
                f"（或 pip install openvino-genai）：{exc}"
            ) from exc
        return genai

    @staticmethod
    def engine_kind(model_path, multimodal=True) -> str:
        """按目录结构决定引擎：有视觉塔且未被配置强制关掉 → vlm。"""
        if not ov_has_vision(model_path):
            return "lm"
        return "vlm" if multimodal else "lm"

    def load(self, model_path, multimodal=True, device=None, properties=None):
        dev = (device or self.device).upper()
        props = dict(properties or {})

        def _do():
            with self._lock:
                cached = self._engines.get(model_path)
                if (cached is not None and cached["device"] == dev
                        and cached.get("properties") == props):
                    return cached
                genai = self._genai()
                kind = self.engine_kind(model_path, multimodal=multimodal)
                t0 = time.perf_counter()
                try:
                    # properties 直通 genai：GPU 用 CACHE_DIR 落盘编译产物
                    # （首次约 1 分钟，命中后 5 秒内），其余为设备属性。
                    if kind == "vlm":
                        pipe = genai.VLMPipeline(model_path, dev, **props)
                    else:
                        pipe = genai.LLMPipeline(model_path, dev, **props)
                except Exception as exc:
                    metrics.inc("ov_load_errors")
                    metrics.emit("ov_load_error", model=model_path, device=dev,
                                 error=str(exc))
                    logger.error("openvino load failed model=%s device=%s err=%s",
                                 model_path, dev, exc)
                    raise RuntimeError(
                        f"OpenVINO 模型加载失败（检查 config.yaml 的 provider.model/"
                        f"device）：{exc}"
                    ) from exc
                load_ms = (time.perf_counter() - t0) * 1000.0
                engine = {"kind": kind, "pipe": pipe, "device": dev, "properties": props}
                self._engines[model_path] = engine
                metrics.observe_ms("ov_load", load_ms)
                metrics.set_gauge("ov_loaded", 1)
                metrics.emit("ov_loaded", model=model_path, device=dev, engine=kind,
                             load_ms=round(load_ms, 1))
                logger.info("openvino loaded model=%s kind=%s device=%s in %.0fms",
                            model_path, kind, dev, load_ms)
                return engine

        return self.run(_do)

    def clear(self):
        """清空引擎缓存（切换模型时释放内存——IR 权重常驻 5~12GB）。"""
        with self._lock:
            self._engines.clear()
        import gc

        gc.collect()

    def generate(self, model_path, messages, max_tokens, temperature,
                 image_paths=None, enable_thinking=None):
        """messages: [{"role","content"}, ...]，由 genai 的 ChatHistory 自行套模板。

        image_paths: list[路径]，仅 vlm 引擎使用，与 content 里追加的
        <ov_genai_image_i> 占位符一一对应。
        enable_thinking: 透传进 chat template（Qwen3.5 等带思考开关的模型）。
        genai 套模板时用模板默认值（通常开思考），必须显式传 False 才会
        插入空 <think> 块关掉思考。
        """

        def _do():
            engine = self._engines.get(model_path)
            if engine is None:
                raise RuntimeError(f"model not loaded: {model_path}")
            genai = self._genai()

            image_tensors = []
            if engine["kind"] == "vlm" and image_paths:
                import numpy as np
                import openvino as ov
                from PIL import Image

                for p in image_paths:
                    with Image.open(p) as pic:
                        rgb = np.array(pic.convert("RGB"))
                    image_tensors.append(ov.Tensor(rgb))

            history = genai.ChatHistory(list(messages))
            if enable_thinking is not None:
                # extra_context 作为模板变量传给 chat template：
                # Qwen3.5 模板据此走 <think>\n\n</think> 空思考分支
                history.set_extra_context({"enable_thinking": bool(enable_thinking)})
            cfg = genai.GenerationConfig()
            cfg.max_new_tokens = int(max_tokens)
            cfg.temperature = float(temperature)
            cfg.top_p = 0.95
            # 温度接近 0 时关采样（走贪心），否则 GenerationConfig 的
            # do_sample 默认 false 会把 temperature 忽略掉
            cfg.do_sample = bool(float(temperature) > 0.05)

            t0 = time.perf_counter()
            if engine["kind"] == "vlm":
                res = engine["pipe"].generate(
                    history, images=image_tensors, generation_config=cfg)
            else:
                res = engine["pipe"].generate(history, generation_config=cfg)
            gen_ms = (time.perf_counter() - t0) * 1000.0

            text = (res.texts or [""])[0] if getattr(res, "texts", None) else ""
            return text, res, gen_ms, engine["kind"]

        return self.run(_do, timeout=max(60.0, max_tokens * 0.5))


_ov_executor: _OpenVINOExecutor | None = None
_ov_exec_lock = threading.Lock()


def get_openvino_executor() -> _OpenVINOExecutor:
    global _ov_executor
    with _ov_exec_lock:
        if _ov_executor is None:
            _ov_executor = _OpenVINOExecutor()
        return _ov_executor


# ── 模型切换器：注册表 + 全局现行模型 ───────────────────────────────
_ACTIVE_MODEL: str = ""
_ACTIVE_PROVIDER: str = ""  # 当前 provider.type，切模型时据此决定清哪个执行器
_MODEL_REGISTRY: List[str] = []
_DEFAULT_MODEL_DIR = "/Users/dango_studio/.lmstudio/models/lmstudio-community"


def _looks_like_model_dir(path) -> bool:
    """目录内含 config.json 且带权重文件（safetensors/npz/gguf）即视为一个模型。

    OpenVINO IR 目录另有一套约定（openvino_model.xml / openvino_language_model.xml），
    直接识别，不要求 HF 的 config.json。
    """
    if is_openvino_model_dir(path):
        return True
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
    """切换全局现行模型（清空当前 provider 的引擎缓存，下次请求惰性加载释放内存）。"""
    global _ACTIVE_MODEL
    path = (path or "").strip()
    if not path:
        raise ValueError("model 不能为空")
    _ACTIVE_MODEL = path
    _clear_engines()
    return _ACTIVE_MODEL


def _clear_engines():
    """按当前 provider 释放引擎缓存（执行器未创建则什么都不做）。"""
    if _ACTIVE_PROVIDER == "openvino":
        targets = (_ov_executor,)
    else:  # api/local 或尚未初始化
        targets = (_ov_executor,)
    for exe in targets:
        if exe is not None:
            exe.clear()


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
    """判断某模型是否要走多模态引擎（openvino VLMPipeline）。

    - setting: True 强制多模态；False 强制纯文本；"auto" 按权重试探。
    - OpenVINO IR：按目录有无 openvino_vision_embeddings_model.xml 判定（有视觉塔即可）。
    """
    if setting is True or str(setting).lower() == "true":
        return True
    try:
        if str(setting).lower() in ("false", "none", "0"):
            return False
    except Exception:
        return False

    if is_openvino_model_dir(model_path):
        return ov_has_vision(model_path)

    return False


# Gemma4 等 channel 模型：思考段 <|channel>…<channel|>，其后才是正文
_THINK_MARK = "<|channel>"
_THINK_END = "<channel|>"
_TURN_MARKS = ("<turn|>", "<|turn>", "<eos>")


def _strip_channel_reply(text):
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


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _strip_think_segments(text):
    """去掉 <think>…</think> 思考段，只留正文。

    未闭合的思考段（生成被截断）视为无正文，丢弃剩余全部。
    """
    if not text:
        return ""
    out = []
    pos = 0
    while True:
        i = text.find(_THINK_OPEN, pos)
        if i < 0:
            out.append(text[pos:])
            break
        out.append(text[pos:i])
        j = text.find(_THINK_CLOSE, i)
        if j < 0:
            break
        pos = j + len(_THINK_CLOSE)
    return "".join(out).strip()


def _strip_vlm_reply(text):
    """Qwen3.5 等多模态模型：去残留思考段/控制符，只留正文。

    思考段以 "thinking\n…\nresponse\n" 分隔；仅当文本以 thinking 开头，
    且出现 response 段头时才裁剪。enable_thinking=false 时输出本就很干净。
    """
    if not text:
        return ""
    text = _strip_channel_reply(text)
    text = _strip_think_segments(text)
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
        # openvino + 支持 thinking 的模型：对话默认关思考（省 token、避免只吐思考截断成空回复）
        self.enable_thinking = bool(provider.get("enable_thinking", False))
        self.multimodal_setting = provider.get("multimodal", "auto")
        self.device = (provider.get("device") or "CPU").upper()
        # 直通 openvino-genai 的设备属性，如 {"CACHE_DIR": "ov_cache"}
        self.properties = dict(provider.get("properties") or {})

        global _ACTIVE_PROVIDER
        _ACTIVE_PROVIDER = self.provider_type

        if self.provider_type == "openvino":
            self.client = None
            self.multimodal = detect_multimodal(self.model, self.multimodal_setting)
            get_openvino_executor().load(self.model, multimodal=self.multimodal,
                                         device=self.device,
                                         properties=self.properties)
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
            if self.provider_type == "openvino":
                self.multimodal = detect_multimodal(model, self.multimodal_setting)
        t0 = time.perf_counter()
        metrics.inc("llm_calls")
        try:
            if self.provider_type == "openvino":
                reply = self._chat_openvino(messages, temp, images=images,
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

    # ── openvino provider ────────────────────────────────────────────
    @staticmethod
    def _ov_image_placeholders(messages, n_images):
        """把 <ov_genai_image_i> 占位符加到「本轮」最后一条 user 消息末尾。

        只改本轮副本，历史里已落盘的旧轮保持纯文本——genai 不支持引用
        历史轮次的图片，旧轮里残留占位符会解析失败。
        """
        if n_images <= 0:
            return list(messages)
        msgs = [dict(m) for m in messages]
        tags = "\n" + "".join(f"<ov_genai_image_{i}>" for i in range(n_images))
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs[i]["content"] = (msgs[i].get("content") or "") + tags
                break
        return msgs

    def _chat_openvino(self, messages, temperature, images=None, enable_thinking=None):
        """openvino-genai 进程内推理。

        - 模板不在外面渲染，ChatHistory 直接交给 pipeline，由 genai 内部套 chat template
          （省掉一次「自己拼 prompt」的口径分歧）；
        - 图片：base64 → 临时文件 → ov.Tensor(HWC uint8 RGB)，本轮末尾挂占位符。
        """
        executor = get_openvino_executor()
        tmp_files = []
        if images:
            if not self.multimodal:
                raise RuntimeError(
                    "当前模型不支持多模态（provider.multimodal 未开启），"
                    "上传的图片无法识别"
                )
            import tempfile

            for img in images:
                fd, path = tempfile.mkstemp(suffix=".img")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(base64.b64decode(img["data"]))
                tmp_files.append(path)

        model_path = self.model  # 已在 chat() 同步为现行模型
        # 每次都过一遍 load：模型/设备/properties 有变（如 POST /model 换模型）时它自己重载，
        # 已缓存则只是一次线程往返
        executor.load(model_path, multimodal=self.multimodal, device=self.device,
                      properties=self.properties)

        req_messages = self._ov_image_placeholders(messages, len(tmp_files))
        thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        try:
            raw, res, gen_ms, kind = executor.generate(
                model_path, req_messages, self.max_tokens, temperature,
                image_paths=tmp_files or None, enable_thinking=thinking,
            )
        finally:
            for p in tmp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass

        reply = _strip_think_segments(_strip_channel_reply(raw))

        prompt_tokens = gen_tokens = gen_tps = prompt_tps = ttft_ms = None
        try:
            perf = res.perf_metrics
            prompt_tokens = perf.get_num_input_tokens()
            gen_tokens = perf.get_num_generated_tokens()
            gen_tps = perf.get_throughput().mean
            ttft_ms = perf.get_ttft().mean  # 首 token 时间 ≈ 预填充（prefill）耗时
            prompt_tps = prompt_tokens / (ttft_ms / 1000.0) if ttft_ms else None
        except Exception:  # 指标拿不到不该影响对话
            pass

        metrics.observe_ms("ov_generate", gen_ms)
        if gen_tps is not None:
            metrics.set_gauge("ov_gen_tps", round(float(gen_tps), 2))
        if ttft_ms is not None:
            metrics.set_gauge("ov_ttft_ms", round(float(ttft_ms), 1))
            metrics.observe_ms("ov_prefill", float(ttft_ms))
        if prompt_tokens:
            metrics.inc("ov_prompt_tokens", int(prompt_tokens))
        if gen_tokens:
            metrics.inc("ov_gen_tokens", int(gen_tokens))
        metrics.emit(
            "ov_generate",
            provider="openvino",
            model=model_path,
            device=self.device,
            engine=kind,
            ms=round(gen_ms, 1),
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            ttft_ms=round(float(ttft_ms), 1) if ttft_ms else None,
            prompt_tps=round(float(prompt_tps), 2) if prompt_tps else None,
            gen_tps=round(float(gen_tps), 2) if gen_tps else None,
            reply_chars=len(reply),
        )

        if not reply:
            metrics.inc("ov_empty_replies")
            metrics.emit("ov_empty_reply", max_tokens=self.max_tokens,
                         raw_head=(raw or "")[:120])
            raise RuntimeError(
                f"OpenVINO 回复为空（max_tokens={self.max_tokens}）：{raw[:200]!r}"
            )
        return reply
