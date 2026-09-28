"""自视身份层（Visual Identity）：让带 VLM 的模型能「认出自己」。

流程：
    用户图片
       ↓
  ① 中立 VLM 描述  —— 不告诉模型「你长这样」，避免确认偏误
       ↓ 结构化 JSON {hair, eyes, style, appearance_age, clothing, summary}
  ② 属性匹配      —— 确定性关键词/颜色别名/年龄段比对准（0~1 置信度）
       ↓ confidence
  ③ (可选) 视觉比对 —— 有参考图时额外问 VLM「两张图是否同一角色」，加权融合
       ↓
  ④ 置信度 ≥ threshold → 生成「图片中可能有我」上下文，注入主对话 system

任何一步失败都静默降级（返回空串），绝不拖垮主对话。
"""

import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .metrics import logger, metrics

# ── 属性权重：仅在身份卡里存在的属性参与计分 ───────────────────────
_WEIGHTS = {
    "hair": 0.30,
    "eyes": 0.25,
    "appearance_style": 0.20,
    "age_appearance": 0.15,
    "clothing_notes": 0.10,
}

_LABEL = {
    "hair": "发色/发型",
    "eyes": "瞳色",
    "appearance_style": "画风",
    "age_appearance": "年龄段",
    "clothing_notes": "服装",
}

# 颜色别名：身份卡里的颜色字/词 → 可能出现的描述写法
_COLOR_ALIAS = {
    "棕": ["棕", "褐", "brown", "棕色", "褐色"],
    "褐": ["棕", "褐", "brown", "棕色", "褐色"],
    "紫": ["紫", "purple", "violet", "紫色"],
    "黑": ["黑", "black", "dark", "黑色"],
    "蓝": ["蓝", "blue", "蓝色"],
    "金": ["金", "gold", "blonde", "金色"],
    "黄": ["黄", "yellow", "amber", "琥珀", "淡黄", "杏"],
    "红": ["红", "red", "绯", "红色"],
    "粉": ["粉", "pink", "粉色"],
    "绿": ["绿", "green", "绿色"],
    "银": ["银", "silver", "银色"],
    "白": ["白", "white", "银色"],
}

_HAIR_STYLE = [
    "长发", "短发", "中长", "齐肩", "及腰",
    "双马尾", "马尾", "丸子头", "双丸子", "发髻",
    "卷发", "直发", "披肩", "波波头",
]

_STYLE_ALIAS = {
    "anime": ["anime", "动漫", "二次元", "卡通", "漫画", "手绘", "日系", "赛璐璐", "平面"],
    "realistic": ["realistic", "写实", "真人", "照片", "油画", "写实风"],
    "3d": ["3d", "三维", "渲染", "游戏建模", "皮克斯"],
    "waifu": ["waifu", "2.5d"],
}

_AGE_KEYS = {"child", "teen", "young adult", "adult", "senior"}

_AGE_SYNONYMS = {
    "child": ["小孩", "儿童", "孩童", "小女孩", "小男孩", "child", "kid"],
    "teen": ["少女", "少年", "teen", "adolescent", "初中", "高中", "女高中生"],
    "young adult": ["年轻", "青年", "成年", "成人", "young adult", "上大学", "大学生", "二十岁", "二十多岁"],
    "adult": ["成年", "成人", "成年女性", "成年男性", "adult", "熟", "三十岁", "三十多岁"],
    "senior": ["老年", "老人", "senior", "中年", "花白"],
}


def _bigram_tokens(value: str) -> set:
    """提取中文滑动双字词 + 英文整词，用于包含度比对。"""
    toks = set()
    for run in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", str(value).lower()):
        if not run:
            continue
        if re.fullmatch(r"[a-z0-9]+", run) and len(run) >= 2:
            toks.add(run)
        for i in range(len(run) - 1):
            if re.fullmatch(r"[\u4e00-\u9fff]{2}", run[i:i + 2]):
                toks.add(run[i:i + 2])
    return toks


def _color_tokens(value: str) -> list:
    """从身份取值里挑出颜色关键词（颜色字 + 英文色名）。"""
    out = []
    for ch in value:
        if ch in _COLOR_ALIAS:
            out.append(ch)
    for w in re.findall(r"[a-z]+", value.lower()):
        out.append(w)
    return out


def _color_matched(identity_value: str, described: str) -> bool:
    low = (described or "").lower()
    if str(identity_value).lower() in low:
        return True
    for c in _color_tokens(identity_value):
        for alias in _COLOR_ALIAS.get(c, [c]):
            if alias and alias in low:
                return True
    return False


def _hair_matched(identity_value: str, described: str) -> bool:
    text = (described or "").lower()
    if not _color_matched(identity_value, described):
        return False
    styles = [s for s in _HAIR_STYLE if s in identity_value]
    if not styles:
        return True
    return any(s in text for s in styles)


def _eyes_matched(identity_value: str, described: str) -> bool:
    return _color_matched(identity_value, described)


def _style_matched(identity_value: str, described: str) -> bool:
    text = (described or "").lower()
    v = str(identity_value).strip().lower()
    if v in text:
        return True
    alias = _STYLE_ALIAS.get(v, [])
    if alias and any(a in text for a in alias):
        return True
    return any(tok in text for tok in _bigram_tokens(v))


def _age_matched(identity_value: str, described: dict) -> bool:
    """年龄段匹配：VLM 给归一化标签时直接比对标签，避免中文同义词误判
    （如「成年」同时是 adult 与 young adult 的同义词）；中文自由描述才走同义词。"""
    v = str(identity_value).strip().lower()
    reported = str(described.get("appearance_age") or "").strip().lower()
    text = reported + " " + str(described.get("summary") or "").lower()
    if v == reported:
        return True
    if reported in _AGE_KEYS:
        return False  # 已给归一化标签但与身份不符 → 判不匹配
    if v in text:
        return True
    for key, syns in _AGE_SYNONYMS.items():
        if key in v and any(s in text for s in syns):
            return True
    return False


def _clothing_matched(identity_value: str, described: str) -> bool:
    text = (described or "").lower()
    v = str(identity_value).strip().lower()
    if v and v in text:
        return True
    return any(tok in text for tok in _bigram_tokens(v))


_COMPARATORS = {
    "hair": _hair_matched,
    "eyes": _eyes_matched,
    "appearance_style": _style_matched,
    "clothing_notes": _clothing_matched,
}


@dataclass
class MatchResult:
    confidence: float = 0.0
    matched: List[dict] = field(default_factory=list)     # [{"attr","value"}]
    attempted: List[str] = field(default_factory=list)    # 参与了计分的属性
    mismatched: List[str] = field(default_factory=list)
    method: str = "attributes"
    reason: str = ""


def match_described(described: dict, identity: dict) -> MatchResult:
    """把 VLM 的中立描述与自视身份卡属性做比对，产出 0~1 置信度。"""
    visual = (identity or {}).get("visual_identity") or {}
    summary = str(described.get("summary") or "").strip()

    def field_of(key, direct):
        v = str(described.get(direct) or "").strip()
        return v or summary

    fields = {
        "hair": field_of("hair", "hair"),
        "eyes": field_of("eyes", "eyes"),
        "appearance_style": field_of("appearance_style", "style"),
        "age_appearance": (
            str(described.get("appearance_age") or "").strip() + " " + summary
        ).strip(),
        "clothing_notes": (
            str(described.get("clothing") or "").strip() + " " + summary
        ).strip(),
    }

    result = MatchResult()
    matched_w = attempted_w = 0.0
    for attr, weight in _WEIGHTS.items():
        identity_value = (visual.get(attr) or "").strip()
        if attr == "age_appearance":
            described_text = (str(described.get("appearance_age") or "").strip()
                              + " " + summary).strip()
        else:
            described_text = fields.get(attr, "")
        if not identity_value or not described_text:
            continue  # 身份卡没有该属性 / 描述缺失 → 不计分，也不当减分
        attempted_w += weight
        if attr == "age_appearance":
            ok = _age_matched(identity_value, described)
        else:
            ok = _COMPARATORS[attr](identity_value, described_text)
        if ok:
            matched_w += weight
            result.matched.append({"attr": attr, "value": identity_value})
        else:
            result.mismatched.append(attr)
        result.attempted.append(attr)

    if attempted_w > 0:
        result.confidence = round(matched_w / attempted_w, 2)
    return result


# ── 提示词 ─────────────────────────────────────────────
_DESCRIBE_PROMPT = """这是用户发来的一张图片。请仔细观察图中的人物（若有多人，以最突出的一人为准），
用中文描述其外貌。只输出一个 JSON 对象，不要有任何其他文字。字段：
{"hair": "发色与发型，例：紫色长发 / 黑色短发",
 "eyes": "瞳色，例：棕色 / 蓝色",
 "style": "画风：anime / realistic / 3d / 其他",
 "appearance_age": "目测年龄段：child / teen / young adult / adult / senior",
 "clothing": "服装简述（一句话）",
 "summary": "一句话整体描述此人"}"""

_COMPARE_PROMPT = """给你两张图片：第一张是「要判断的人物」，第二张是「目标角色的参考图」。
请判断第一张图里的人物是否就是第二张图里的同一角色。
只输出 JSON：{"same_character": 0到1之间的小数, "reason": "不超过20字"}"""


def _parse_json(text) -> Optional[dict]:
    text = text or ""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


class VisualIdentity:
    """自视身份：读取身份卡 → 描述图片 → 匹配 → 产出自我识别上下文。"""

    def __init__(self, config, llm=None):
        self.llm = llm
        self.threshold = 0.55
        self.card = None
        self._ref_images: List[dict] = []
        self.enabled = False
        # 最近一次中立描述（analyze 顺带产出）：供图片内容说明复用，零额外 LLM 调用
        self.last_described: Optional[dict] = None

        vis = (config or {}).get("visual_identity") or {}
        # card 显式留空（角色级：如白鸥无身份卡）→ 关闭自识别；
        # 未配置该键才回退默认路径（兼容旧配置）。
        raw_card = vis.get("card")
        card_path = "visual/identity.json" if raw_card is None else str(raw_card).strip()
        ref_dir = (vis.get("reference_dir") or
                   os.path.join(os.path.dirname(card_path or "."), "references"))
        try:
            self.threshold = max(0.0, min(1.0, float(vis.get("threshold", 0.55))))
            if not os.path.isfile(card_path):
                logger.debug("visual identity card 不存在，自识别关闭: %s", card_path)
                return
            with open(card_path, encoding="utf-8") as fh:
                self.card = json.load(fh)
            if not isinstance(self.card.get("visual_identity"), dict):
                self.card = None
                return
            self._load_references(ref_dir)
            self.enabled = True
            logger.info("visual identity 已启用 character=%s threshold=%.2f refs=%d",
                        self.card.get("character"), self.threshold, len(self._ref_images))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.card = None
            logger.warning("visual identity 初始化失败，自识别关闭: %s", exc)

    # ── 参考图加载 ──────────────────────────────────────
    def _load_references(self, ref_dir: str):
        for name in self.card.get("visual_identity", {}).get("reference_images") or []:
            path = name if os.path.isabs(name) else os.path.join(ref_dir, name)
            if not os.path.isfile(path):
                logger.debug("参考图缺失（跳过）: %s", path)
                continue
            try:
                with open(path, "rb") as fh:
                    data = base64.b64encode(fh.read()).decode("ascii")
                mime = mimetypes.guess_type(name)[0] or "image/png"
                self._ref_images.append({"mime": mime, "data": data})
            except OSError:
                continue

    # ── 主入口 ──────────────────────────────────────────
    def analyze(self, images: List[dict]) -> str:
        """对图片做自识别。命中返回注入上下文的字符串，否则返回空串。"""
        if not self.enabled or not images or self.llm is None:
            return ""
        t0 = time.perf_counter()
        try:
            described = self._describe(images)
            self.last_described = described
            if not described:
                return ""
            match = match_described(described, self.card)
            metrics.emit(
                "identity_describe",
                summary=(described.get("summary") or "")[:80],
                parsed=described.get("_parsed", False),
            )

            if self._ref_images and len(images) == 1:
                vis = self._visual_compare(images[0], self._ref_images[0])
                if vis is not None:
                    match.confidence = round(0.6 * match.confidence +
                                             0.4 * vis["same_character"], 2)
                    match.method = "hybrid"
                    if vis.get("reason") and not match.reason:
                        match.reason = vis["reason"]

            metrics.inc("identity_checks")
            metrics.emit(
                "identity_match",
                confidence=match.confidence,
                threshold=self.threshold,
                matched=[m["attr"] for m in match.matched],
                attempted=match.attempted,
                method=match.method,
            )

            if match.confidence < self.threshold:
                return ""
            metrics.inc("identity_hits")
            ctx = self._self_context(match)
            metrics.emit(
                "identity_recognized",
                confidence=match.confidence,
                matched=[m["attr"] for m in match.matched],
            )
            return ctx
        except Exception as exc:
            metrics.inc("identity_errors")
            metrics.emit("identity_error", error=str(exc))
            logger.warning("visual identity analyze failed: %s", exc)
            return ""
        finally:
            metrics.observe_ms("identity_analyze", (time.perf_counter() - t0) * 1000.0)

    # ── 阶段①：中立描述 ─────────────────────────────────
    def _describe(self, images: List[dict]) -> Optional[dict]:
        msgs = [
            {"role": "system", "content": "你是图像描述助手，只如实描述，不臆测图中人物的身份。"},
            {"role": "user", "content": _DESCRIBE_PROMPT},
        ]
        raw = self.llm.chat(msgs, images=images, temperature=0.2)
        obj = _parse_json(raw)
        if obj is None:
            # JSON 解析失败 → 以原文当 summary 兜底（仍可做关键词匹配）
            return {"summary": (raw or "").strip(), "_parsed": False}
        obj.setdefault("summary", "")
        obj["_parsed"] = True
        return obj

    # ── 阶段③：视觉比对（有参考图时） ──────────────────
    def _visual_compare(self, user_img: dict, ref_img: dict) -> Optional[dict]:
        msgs = [
            {"role": "system", "content": "你是角色外观比对助手，只做客观比对。"},
            {"role": "user", "content": _COMPARE_PROMPT},
        ]
        try:
            raw = self.llm.chat(msgs, images=[user_img, ref_img], temperature=0.1)
        except Exception as exc:
            logger.debug("visual compare 不可用（模型可能只支持单图）: %s", exc)
            return None
        obj = _parse_json(raw)
        if not obj:
            return None
        try:
            same = float(obj.get("same_character", 0.0))
        except (TypeError, ValueError):
            same = 0.0
        same = max(0.0, min(1.0, same))
        return {"same_character": same, "reason": str(obj.get("reason") or "")}

    # ── 阶段④：自我识别上下文 ───────────────────────────
    def _self_context(self, match: MatchResult) -> str:
        char = (self.card or {}).get("character") or ""
        if char:
            head = f"【自我识别 · 你注意到图片中的人物与你（{char}）很像】"
        else:
            head = "【自我识别 · 你注意到图片中的人物与你有几分相似】"
        lines = [head, f"- 相似置信度：{match.confidence:.2f}"]
        if match.matched:
            pads = "、".join(
                f"{_LABEL.get(m['attr'], m['attr'])}「{m['value']}」"
                for m in match.matched
            )
            lines.append(f"- 匹配特征：{pads}")
        if match.reason:
            lines.append(f"- 依据：{match.reason}")
        lines.append(
            "这可能就是你的照片、画像或与你相关的画面。仅当上下文明显不是本人（比如显然是别人、"
            "其他角色、或游戏截图）时才不要认。按你的性格自然接话，不要解释识别过程与原理。"
        )
        return "\n".join(lines)

    # ── 观测：身份卡 ────────────────────────────────────
    def get_card(self) -> Optional[dict]:
        return self.card