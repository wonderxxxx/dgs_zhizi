#!/usr/bin/env python3
"""自视身份层（Visual Identity）单测：运行方式  cd zhizi && .venv/bin/python tests/test_visual_identity.py

覆盖：属性匹配（命中/未命中/年龄段标签优先级）/ JSON 解析容错 /
      analyze 全链路（fake VLM 命中注入上下文、未命中返回空）/ 卡缺失禁用。
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.visual_identity import (
    VisualIdentity,
    _parse_json,
    match_described,
)

PASS = []


def check(name, cond, detail=""):
    status = "✅" if cond else "❌"
    PASS.append(cond)
    print(f"{status} {name}" + (f"  ({detail})" if detail and not cond else ""))


def identity_card():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "visual", "identity.json"), encoding="utf-8") as fh:
        return json.load(fh)


class FakeVision:
    """fake VLM：describe 与 compare 各返回一段固定 JSON。"""
    model = "fake"
    multimodal = True

    def __init__(self, describe_json, compare_json=None):
        self.describe_json = describe_json
        self.compare_json = compare_json
        self.calls = []

    def chat(self, messages, images=None, temperature=None):
        self.calls.append({"images": len(images or []), "temperature": temperature})
        if images and len(images) > 1 and self.compare_json is not None:
            return json.dumps(self.compare_json, ensure_ascii=False)
        return json.dumps(self.describe_json, ensure_ascii=False)


def _img():
    return [{"mime": "image/png", "data": "AAAA"}]


def main():
    card = identity_card()
    tmp = tempfile.mkdtemp(prefix="zhizi_vi_")
    cfg = {"visual_identity": {
        "card": os.path.abspath(os.path.join("visual", "identity.json")),
    }}

    # ── 1. 属性匹配：命中 ──────────────────────────────────
    hit = {
        "hair": "紫色长发", "eyes": "棕色", "style": "anime",
        "appearance_age": "young adult", "clothing": "学生制服",
        "summary": "紫色长发的少女，看起来是大学生模样",
    }
    r = match_described(hit, card)
    check("匹配：智子画风全属性命中", r.confidence >= 0.9 and len(r.matched) == 5,
          f"conf={r.confidence} matched={[m['attr'] for m in r.matched]}")

    # ── 2. 属性匹配：明显不是她（金色短发/蓝眼/写实/成年男） ──
    miss = {
        "hair": "金色短发", "eyes": "蓝色", "style": "realistic",
        "appearance_age": "adult", "clothing": "西装",
        "summary": "成年男性写真照片",
    }
    r = match_described(miss, card)
    check("匹配：无关人物排除", r.confidence < 0.3 and "hair" in r.mismatched,
          f"conf={r.confidence} matched={[m['attr'] for m in r.matched]}")

    # ── 3. 年龄段：归一化标签优先，避免「成年」同义词误判 ──
    age_cross = {
        "hair": "紫色长发", "eyes": "棕色", "style": "anime",
        "appearance_age": "adult", "clothing": "学生制服",
        "summary": "看起来很成熟",
    }
    r = match_described(age_cross, card)
    check("匹配：age 归一化标签不符不计入", "age_appearance" in r.mismatched,
          f"conf={r.confidence}")

    # ── 4. JSON 解析容错 ──────────────────────────────────
    check("解析：纯 JSON", _parse_json('{"hair":"紫色长发"}') == {"hair": "紫色长发"})
    check("解析：夹带前后缀", _parse_json('好的{"hair":"紫"}结束了') == {"hair": "紫"})
    check("解析：非 JSON 返回 None（走 summary 兜底）", _parse_json("一位紫发少女") is None)

    # ── 5. analyze 全链路：命中 → 注入上下文 ──────────────
    vi = VisualIdentity(cfg, FakeVision(hit))
    ctx = vi.analyze(_img())
    check("analyze：命中返回上下文", vi.enabled and len(ctx) > 0 and "相似置信度" in ctx)
    check("analyze：上下文含角色名", "千夜智子" in ctx)

    # ── 6. analyze 全链路：未命中 → 空串 ───────────────────
    vi2 = VisualIdentity(cfg, FakeVision(miss))
    check("analyze：未命中返回空", vi2.analyze(_img()) == "")

    # ── 7. 有参考图时叠加视觉比对（hybrid 加权） ───────────
    ref_dir = os.path.join(tmp, "refs")
    os.makedirs(ref_dir, exist_ok=True)
    ref_path = os.path.join(ref_dir, "chizuko_front.png")
    with open(ref_path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")  # 假 png 头部，仅测试加载逻辑
    cfg_ref = {"visual_identity": {
        "card": os.path.abspath(os.path.join("visual", "identity.json")),
        "reference_dir": ref_dir,
        "threshold": 0.5,
    }}
    vi3 = VisualIdentity(cfg_ref, FakeVision(hit, {"same_character": 0.8, "reason": "同一角色"}))
    ctx3 = vi3.analyze(_img())
    check("analyze：参考图 + 视觉比对融合后命中",
          vi3.enabled and len(vi3._ref_images) == 1 and bool(ctx3),
          f"refs={len(vi3._ref_images)} ctx={bool(ctx3)}")

    # ── 8. 卡缺失 → 禁用 ──────────────────────────────────
    vi_off = VisualIdentity({"visual_identity": {"card": os.path.join(tmp, "nope.json")}})
    check("禁用：卡不存在不再识别", not vi_off.enabled and vi_off.analyze(_img()) == "")

    print()
    print(f"通过 {sum(PASS)}/{len(PASS)}")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()