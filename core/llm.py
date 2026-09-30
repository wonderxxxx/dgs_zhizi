"""统一调用层：模型无关。

provider.type：
- api / local：OpenAI 兼容协议（云端 API 或 Ollama / LM Studio）
- mlx：Apple Silicon 进程内推理（mlx-lm），不起本地服务

切换调用方式只改 config.yaml 的 provider 段，本模块对外仍是 chat()。
"""

import base64
import hashlib
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
        self._no_vision = set()  # 声称多模态但视觉塔缺权重的模型
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
                        if _vision_disabled(engine[1]):
                            self._no_vision.add(model_path)
                            metrics.emit("vision_weights_missing",
                                         model=model_path, stage="load")
                            logger.warning(
                                "mlx 视觉塔无权重 model=%s（transformers 已禁用 "
                                "vision_tower），按纯文本处理", model_path)
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
            self._no_vision.clear()

    def vision_missing(self, model_path) -> bool:
        """已加载的模型是否缺视觉塔权重（mlx_vlm 加载后复查的结果）。"""
        return model_path in self._no_vision

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
_DEFAULT_MODEL_DIR = "/Users/dango_studio/.lmstudio/models"


def _looks_like_model_dir(path) -> bool:
    """目录内含权重文件即视为一个模型。

    - HF/MLX 权重目录：有 config.json 且带 safetensors/npz
    - GGUF 目录：无 config.json，只要有一个非 mmproj/imatrix 的 .gguf 即可
    """
    try:
        if not os.path.isdir(path):
            return False
        names = os.listdir(path)
        has_config = os.path.isfile(os.path.join(path, "config.json"))
        for name in names:
            if name.endswith((".safetensors", ".npz")) and has_config:
                return True
            if name.endswith(".gguf") and not name.startswith(("mmproj-", "imatrix-")):
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


def model_label(path: str) -> str:
    """下拉框展示名：多级路径带上发布者目录，避免同名模型看不出区别。"""
    parts = [p for p in (path or "").strip().rstrip("/").split("/") if p]
    if not parts:
        return ""
    return "/".join(parts[-2:]) if len(parts) >= 3 else parts[-1]


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
    return vision_capability(_ACTIVE_MODEL, "auto")


# ── 视觉能力探针 ───────────────────────────────────────────────────
# config.json 里的 image_token_id / vision_config 只说明「架构声称多模态」，
# 不代表权重里真有视觉塔：转换时漏掉视觉塔的模型（config 照旧、权重只有语言
# 模型）mlx_vlm 照样加载成功，直到带图推理才在 vision_tower=None 上炸成
# 'NoneType' object is not callable。所以判据是权重，不是描述文件。
_VISION_WEIGHT_MARKERS = (
    "vision", "visual", "vpm", "image_tower", "img_tower",
    "mm_projector", "multi_modal_projector", "resampler", "perceiver",
)
_CONFIG_MODAL_KEYS = ("vision_config", "image_token_id", "video_token_id", "audio_config")
_vision_probe_cache: dict = {}
# 缺视觉塔的模型每个进程只警告一次：模型切换器每列一次模型、每次带图请求
# 都会重新判一遍，不去重的话同一条警告会反复刷屏，metrics.emit 还会往
# 内存 events 列表里无限追加。指纹变了（模型被修好/重下）则重新提示。
_vision_missing_warned: dict = {}
_vision_missing_lock = threading.Lock()

# 「已提示过的坏 config」跨重启也只提示一次，落盘记账。纯提示抑制机制，
# 读不到/写不了/文件损坏都必须静默退化，绝不能影响视觉判据本身。
_STATE_PATH = os.environ.get("ZHIZI_STATE_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".zhizi_state.json"
)
_state_cache: Optional[dict] = None
_state_lock = threading.Lock()


def _load_state() -> dict:
    """读运行时状态；只读一次，坏了当空。"""
    global _state_cache
    with _state_lock:
        if _state_cache is not None:
            return _state_cache
        try:
            with open(_STATE_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
            _state_cache = data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            _state_cache = {}
        return _state_cache


def _save_state() -> None:
    """原子写。失败只记 debug，绝不抛——提示抑制不该拖垮视觉判据。"""
    with _state_lock:
        if _state_cache is None:
            return
        try:
            os.makedirs(os.path.dirname(_STATE_PATH) or ".", exist_ok=True)
            tmp = f"{_STATE_PATH}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(_state_cache, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, _STATE_PATH)  # 原子替换，避免半截文件
        except OSError as exc:
            logger.debug("state 写入失败（忽略）: %s", exc)


def _vision_fingerprint(model_path) -> str:
    """config 内容 + 权重文件清单的指纹。

    两者任一变化（改了 config、重新下载/转换权重）指纹就变，届时重新提示，
    这样「模型修好了」能被看见，而反复启动不会一直刷同一条坏 config。
    """
    h = hashlib.sha256()
    try:
        with open(os.path.join(model_path, "config.json"), "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    except OSError:
        h.update(b"no-config")
    try:
        for name in sorted(os.listdir(model_path)):
            if not name.endswith((".safetensors", ".npz", ".gguf", ".bin")):
                continue
            st = os.stat(os.path.join(model_path, name))
            h.update(f"{name}:{st.st_size}:{st.st_mtime_ns}".encode())
    except OSError:
        h.update(b"no-weights")
    return h.hexdigest()[:16]


def _record_vision_check(model_path, fingerprint) -> bool:
    """这条坏 config 之前提示过没有。顺便负责落盘。"""
    checks = _load_state().setdefault("vision_checks", {})
    if not isinstance(checks, dict):
        checks = {}
        _load_state()["vision_checks"] = checks
    rec = checks.get(model_path)
    if isinstance(rec, dict) and rec.get("fingerprint") == fingerprint:
        return True
    checks[model_path] = {
        "fingerprint": fingerprint,
        "first_seen": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_state()
    return False


def _forget_vision_check(model_path) -> None:
    """模型已修好（判出视觉塔）就把记录清掉，将来再坏才会重新提示。"""
    checks = _load_state().get("vision_checks")
    if isinstance(checks, dict) and model_path in checks:
        del checks[model_path]
        _save_state()


def _vision_disabled(model) -> bool:
    """加载后复查：transformers 找不到视觉塔权重时会把子模块置 None
    （gemma4 是 vision_tower / embed_vision），此时带图推理必崩。
    类属性与实例属性都查（mlx Module 挂在实例上，替身/降级实现可能挂类上）。"""
    names = set(vars(model)) if hasattr(model, "__dict__") else set()
    names.update(n for n in dir(type(model)) if not n.startswith("_"))
    for name in names:
        if not any(m in name for m in ("vision", "visual", "vpm")):
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if value is None:
            return True
    return False


def _tensor_names(model_path) -> Optional[List[str]]:
    """列出权重张量名；读不到返回 None（调用方据此放弃判断）。"""
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index, encoding="utf-8") as fh:
                return list(json.load(fh).get("weight_map", {}).keys())
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
    try:
        names = os.listdir(model_path)
    except OSError:
        return None
    for name in sorted(names):  # 单文件 safetensors：只读文件头，不加载权重
        if name.endswith(".safetensors"):
            try:
                from safetensors import safe_open

                with safe_open(os.path.join(model_path, name), framework="numpy") as fh:
                    return list(fh.keys())
            except Exception:
                return None
    for name in sorted(names):
        if name.endswith(".npz"):
            try:
                import zipfile

                with zipfile.ZipFile(os.path.join(model_path, name)) as zf:
                    return [n[:-4] for n in zf.namelist() if n.endswith(".npy")]
            except Exception:
                return None
    return None


def _has_vision_weights(model_path) -> Optional[bool]:
    """权重里是否含视觉塔张量。None = 判不出来（GGUF / 远端仓库等）。"""
    if model_path in _vision_probe_cache:
        return _vision_probe_cache[model_path]
    names = _tensor_names(model_path)
    if names is None:
        return None
    # 视觉塔挂在模型根下（vision_tower.* / model.visual.* / language_model.vision_tower.*）
    hit = any(
        any(m in seg for m in _VISION_WEIGHT_MARKERS)
        for key in names
        for seg in key.split(".")[:3]
    )
    _vision_probe_cache[model_path] = hit
    return hit


def _forced_multimodal(setting) -> Optional[bool]:
    """True/False = 用户显式指定；None = auto。"""
    if setting is True:
        return True
    text = str(setting).strip().lower()
    if text == "true":
        return True
    if text in ("false", "none", "0"):
        return False
    return None


def _cfg_claims_modal(model_path) -> bool:
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(cfg, dict) and any(k in cfg for k in _CONFIG_MODAL_KEYS)


def _warn_vision_missing_once(model_path) -> bool:
    """首次发现该模型缺视觉塔才返回 True（并负责记指标+告警）。

    两级去重：进程内按指纹去重（挡同一进程里的反复调用），落盘按指纹记账
    （跨重启也只提示一次）。指纹变了说明 config 或权重动过，重新提示。
    """
    fp = _vision_fingerprint(model_path)
    with _vision_missing_lock:
        if _vision_missing_warned.get(model_path) == fp:
            return False
        _vision_missing_warned[model_path] = fp
    if _record_vision_check(model_path, fp):
        return False  # 之前（别的进程）已经提示过同一份坏 config
    metrics.inc("mlx_vision_weights_missing")
    metrics.emit("vision_weights_missing", model=model_path, stage="probe")
    logger.warning(
        "%s 的 config 声明多模态，但权重里没有视觉塔（转换时漏了）——"
        "已按纯文本模型处理，发图会报错。%s",
        model_label(model_path), _vision_sibling_hint(model_path),
    )
    return True


def vision_capability(model_path, setting="auto") -> bool:
    """该模型是否真能看图。config 只当线索，权重才是判据。"""
    forced = _forced_multimodal(setting)
    if forced is not None:
        return forced
    if not model_path or not _cfg_claims_modal(model_path):
        return False
    has = _has_vision_weights(model_path)
    if has is None:  # 判不出来（远端仓库等）：沿用 config 判据
        return True
    if has:
        _forget_vision_check(model_path)  # 修好了就销账，将来再坏才重新提示
        return True
    _warn_vision_missing_once(model_path)
    return False


def _vision_sibling_hint(model_path) -> str:
    """同名但带视觉塔的兄弟版本（多在别的发布者目录下），给个可点的提示。"""
    want = os.path.basename(model_path.rstrip("/"))
    for other in get_model_registry():
        if other == model_path or os.path.basename(other) != want:
            continue
        if vision_capability(other, "auto"):
            return f"可改用 {other}"
    return "同类可换带视觉塔的版本。"


def client_vision(client) -> bool:
    """某个 LLMClient 能否看图（含 provider.multimodal 强制开关与加载后复查）。"""
    if client is None:
        return False
    flag = getattr(client, "vision", None)
    if flag is None:  # 非 mlx provider：由服务端决定
        return bool(getattr(client, "multimodal", False))
    if not flag or getattr(client, "provider_type", "") != "mlx":
        return bool(flag)
    return not get_mlx_executor().vision_missing(getattr(client, "model", ""))


def detect_multimodal(model_path, setting="auto") -> bool:
    """判断某 MLX 模型是否走多模态引擎（mlx_vlm）。

    这里只管「引擎选择」：架构是 multimodal 类就得用 mlx_vlm 加载（纯文本
    推理也依赖它），哪怕这份权重没带视觉塔。带图能力由 vision_capability 判定。

    - setting: True 强制多模态；False 强制纯文本；"auto" 看 config 是否声明模态。
    """
    forced = _forced_multimodal(setting)
    if forced is not None:
        return forced
    try:
        importlib.import_module("mlx_vlm")
    except ImportError:
        return False
    return _cfg_claims_modal(model_path)


# Gemma4 等 channel 模型有两套思考标记，都要认：
# - 模板态：<|channel|>thought\n<channel|>正文       （split_thinking 宏的切法）
# - 生成态：<|channel|>analysis<|message|>思考<|end|>…<|channel|>final<|message|>正文
# 后者带 <|message|> 且用 <|end|> 收尾，没有无管道的 <channel|>，只按模板那套切
# 会一个标记都匹配不到，整段思考原文直接当回复落进记忆。
_CHANNEL_OPEN = "<|channel|>"
_CHANNEL_BODY = "<|message|>"
_CHANNEL_END = "<|end|>"
_THINK_END = "<channel|>"
_THINK_CHANNELS = ("analysis", "thought", "thinking", "reasoning",
                   "commentary", "scratchpad", "reflection", "plan")
_CONTROL_MARKS = (
    "<channel|>", "<|channel|>", "<|message|>", "<|start|>", "<|end|>",
    "<turn|>", "<|turn>", "<eos>", "<|eos|>", "<end_of_turn>", "<|end_of_turn|>",
)


def _strip_channel_segments(text):
    """按 <|channel|>名称<|message|>正文<|end|> 拆段，只留非思考通道的正文。

    段与段之间的 <|start|>assistant 之类的结构标记直接丢掉。正文永远在最后一个
    通道里，但按通道名判断更稳——final 之外的（如 tool_call）也该留。
    """
    if _CHANNEL_OPEN not in text or _CHANNEL_BODY not in text:
        return text
    kept, pos = [], 0
    while True:
        head = text.find(_CHANNEL_OPEN, pos)
        if head < 0:
            break
        name_start = head + len(_CHANNEL_OPEN)
        name_end = min(
            (i for i in (text.find(_CHANNEL_BODY, name_start),
                         text.find(_CHANNEL_END, name_start))
             if i >= 0),
            default=len(text),
        )
        name = text[name_start:name_end].strip().lower()
        close = text.find(_CHANNEL_END, name_end)
        body_end = len(text) if close < 0 else close
        if name not in _THINK_CHANNELS:
            kept.append(text[name_end:body_end])
        pos = body_end + len(_CHANNEL_END) if close >= 0 else body_end
    return "".join(kept)


def _strip_mlx_reply(text):
    """去掉思考通道 / 残留控制符，只留回复正文。"""
    if not text:
        return ""
    out = _strip_channel_segments(text)
    parts = out.split(_THINK_END)
    kept = []
    for part in parts:
        if _CHANNEL_OPEN in part:
            kept.append(part.split(_CHANNEL_OPEN, 1)[0])
        else:
            kept.append(part)
    out = "".join(kept)
    for mark in _CONTROL_MARKS:
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
            self.vision = vision_capability(self.model, self.multimodal_setting)
            get_mlx_executor().load(self.model, multimodal=self.multimodal)
            return

        self.multimodal = True  # OpenAI 兼容服务端透传内容块，由服务端决定是否支持
        self.vision = True
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
                self.vision = vision_capability(model, self.multimodal_setting)
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
        model_path = self.model  # 已在 chat() 同步为现行模型
        image_paths = None
        tmp_files = []
        if images:
            if not self.multimodal:
                raise RuntimeError(
                    "当前模型不支持多模态（provider.multimodal 未开启），"
                    "上传的图片无法识别"
                )
            if not client_vision(self):
                raise RuntimeError(
                    f"{model_label(model_path)} 看不了图：权重里没有视觉塔"
                    f"（config 虽声明了多模态，转换时只带了语言模型）。"
                    f"{_vision_sibling_hint(model_path)}"
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
        if executor._engines.get(model_path) is None:
            executor.load(model_path, multimodal=self.multimodal)
        if image_paths and executor.vision_missing(model_path):
            # 探针判不出来时的兜底：加载后复查发现视觉塔确实缺权重
            raise RuntimeError(
                f"{model_label(model_path)} 看不了图：加载后视觉塔为空，"
                f"这份权重只带了语言模型。{_vision_sibling_hint(model_path)}"
            )

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
