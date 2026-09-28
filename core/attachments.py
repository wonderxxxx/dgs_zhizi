"""图片附件：落盘 + 内容说明。

两条线，各管一件事：
- **看得见**：图片字节落到 `notes/<user_id>/<角色>/images/`，聊天记录只存引用
  （文件名 + mime）。base64 不进 JSON、不进上下文，刷新页面还能翻出原图。
- **记得住**：图片内容用一句话说明（VLM 描述），随该轮消息写进五维记忆
  （工作记忆存 `caption` 字段，近期/事实/反思按带说明的文本检索），
  她下一轮能接上"你刚才给我看的那张图"。

有自视身份卡的角色（智子）复用自识别阶段已做的那次 VLM 描述，不额外调用；
无身份卡（白鸥）或模型不支持多模态时降级为 `[图片]` 标记，说明为空。
"""

import base64
import binascii
import json
import mimetypes
import os
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from .metrics import logger, metrics

IMG_DIR = "images"
_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,120}")
_CAPTION_MAX = 120
_MAX_BYTES = 12 * 1024 * 1024   # 单图上限，别让一轮贴图把磁盘写满
_ATTR_ORDER = ("summary", "clothing", "appearance_age", "hair", "eyes", "style")

_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
        "image/gif": ".gif", "image/webp": ".webp", "image/bmp": ".bmp"}

_CAPTION_PROMPT = (
    "用一句不超过 30 字的中文，客观描述这些图片里的主要内容和场景"
    "（人物外貌、动作、环境等）。只描述看得见的东西，不要猜测人物身份或关系。"
)


def image_dir(note_root: str) -> str:
    """某用户 × 某角色的图片目录（note_root = notes/<user_id>/<角色>）。"""
    return os.path.join(note_root or "", IMG_DIR)


def save_images(note_root: str, images: List[dict]) -> List[Dict[str, str]]:
    """base64 图片 → 落盘，返回引用 [{name, mime}]（失败的图静默跳过）。"""
    if not note_root or not images:
        return []
    out = []
    for img in images:
        try:
            data = base64.b64decode(img.get("data") or "", validate=True)
            if not data:
                continue
            if len(data) > _MAX_BYTES:
                logger.warning("图片过大（%.1f MB），已跳过: %s",
                               len(data) / 1048576.0, len(data))
                continue
            mime = str(img.get("mime") or "image/png").lower()
            ext = _EXT.get(mime) or (os.path.splitext(mime)[1] or ".png")
            if not re.fullmatch(r"\.[A-Za-z0-9]{1,5}", ext):
                ext = ".png"
            name = (f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}{ext}")
            os.makedirs(image_dir(note_root), exist_ok=True)
            with open(os.path.join(image_dir(note_root), name), "wb") as fh:
                fh.write(data)
            out.append({"name": name, "mime": mime})
        except (binascii.Error, ValueError, OSError) as exc:
            logger.warning("图片落盘失败，已跳过: %s", exc)
    metrics.inc("images_saved", len(out))
    return out


def resolve(note_root: str, name: str) -> Optional[str]:
    """引用 → 绝对路径。名字只认 [A-Za-z0-9._-]，越界/穿越一律 None。"""
    if not note_root or not name or not _NAME_RE.fullmatch(str(name)):
        return None
    path = os.path.join(image_dir(note_root), str(name))
    return path if os.path.isfile(path) else None


def read_image(note_root: str, name: str):
    """读回图片字节 → (bytes, mime)；不存在返回 None。"""
    path = resolve(note_root, name)
    if not path:
        return None
    mime = mimetypes.guess_type(path)[0] or "image/png"
    try:
        with open(path, "rb") as fh:
            return fh.read(), mime
    except OSError:
        return None


def clear(note_root: str) -> int:
    """删掉该用户该角色的全部图片（清记忆时顺手清，避免留孤儿文件）。"""
    root = image_dir(note_root)
    if not os.path.isdir(root):
        return 0
    n = 0
    for name in os.listdir(root):
        if not _NAME_RE.fullmatch(name):
            continue
        try:
            os.remove(os.path.join(root, name))
            n += 1
        except OSError:
            continue
    return n


def caption_from(described: Optional[dict]) -> str:
    """自识别阶段的中立描述 → 一句话说明（零额外 LLM 调用）。"""
    if not isinstance(described, dict):
        return ""
    parts = []
    for key in _ATTR_ORDER:
        val = str(described.get(key) or "").strip()
        if val and val not in parts:
            parts.append(val)
    return "，".join(parts)[:_CAPTION_MAX]


def _plain_sentence(raw: str) -> str:
    """把模型回复收拾成一句话：去围栏、JSON 取 summary、只留首行。

    说明是要写进她记忆的文本，不能塞结构化原文（有模型会顺着描述任务回 JSON）。
    """
    text = (raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"^```[A-Za-z]*\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            for key in ("summary", "caption", "description", "text"):
                val = str(obj.get(key) or "").strip()
                if val:
                    return val[:_CAPTION_MAX]
    return text.splitlines()[0].strip()[:_CAPTION_MAX]


def caption(llm, images: List[dict], described: Optional[dict] = None) -> str:
    """图片内容的一句话说明（进记忆/上下文用）；失败降级为空串。"""
    if not images:
        return ""
    reuse = caption_from(described)
    if reuse:
        return reuse
    if llm is None or not getattr(llm, "multimodal", False):
        return ""
    t0 = time.perf_counter()
    try:
        raw = llm.chat(
            [{"role": "system", "content": "你是图像描述助手，只描述客观可见内容。"},
             {"role": "user", "content": _CAPTION_PROMPT}],
            images=images, temperature=0.2, enable_thinking=False,
        )
    except Exception as exc:  # 描述失败不该拖垮对话
        metrics.inc("image_caption_errors")
        logger.warning("图片说明失败: %s", exc)
        return ""
    metrics.observe_ms("image_caption", (time.perf_counter() - t0) * 1000.0)
    text = _plain_sentence(raw)
    if not text:
        return ""
    metrics.inc("image_captions")
    return text


def with_caption(content: str, caption: str) -> str:
    """检索/上下文用的文本：原话 + 一行图片内容说明。"""
    if not caption:
        return content or ""
    return f"{content or ''}\n[图片内容] {caption}".strip()
