#!/usr/bin/env python3
"""时间锚点（第 1 步）：让她知道「现在几点」与「距上次对话多久」。

覆盖：time_anchor 纯函数（绝对时间恒在 / 相对时间按阈值开关 / 坏时间戳降级 /
      间隔措辞分档）/ build_messages 注入位置 / 无历史只给绝对时间。
运行方式：cd zhizi && .venv/bin/python tests/test_timeflow.py
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import persona as persona_mod  # noqa: E402
from core.persona import Persona, time_anchor, _humanize_gap  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZHIZI_MD = os.path.join(BASE, "prompts", "zhizi_v1.md")

PASS = []


def check(name, cond, detail=""):
    status = "✅" if cond else "❌"
    PASS.append(bool(cond))
    print(f"{status} {name}" + (f"  ({detail})" if not cond else ""))


class ScriptLLM:
    model = "fake"
    multimodal = True

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def chat(self, messages, temperature=None, images=None, enable_thinking=None):
        system = messages[0].get("content", "") if messages else ""
        if len(messages) <= 1:
            return "[]"
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else "嗯。"


def make_persona():
    cfg = {
        "system_prompt": ZHIZI_MD,
        "web_search": {"enabled": False},
        "provider": {"type": "local", "base_url": "http://x", "model": "fake",
                     "api_key": "x", "temperature": 0.85},
    }
    return Persona(cfg, llm=ScriptLLM(["好。"]))


def main():
    now = datetime(2026, 10, 13, 23, 10, 0)  # 周二

    # ── 1. 绝对时间恒在 + 星期 ──────────────────────────────
    only = time_anchor(now)
    check("绝对时间：始终给出年月日·星期·时刻",
          "2026-10-13" in only and "周二" in only and "23:10" in only,
          f"got={only!r}")
    check("无历史：不给相对时间", "已过" not in only)

    # ── 2. 阈值：短间隔不给相对时间 ─────────────────────────
    near = time_anchor(now, (now - timedelta(minutes=3)).isoformat())
    check("阈值内（3 分钟）：不加相对时间", "已过" not in near, f"got={near!r}")
    edge = time_anchor(now, (now - timedelta(minutes=10)).isoformat())
    check("阈值边界（10 分钟）：开始给相对时间", "已过" in edge, f"got={edge!r}")

    # ── 3. 间隔措辞分档 ───────────────────────────────────
    h = time_anchor(now, (now - timedelta(hours=3, minutes=20)).isoformat())
    check("小时档：小时+分钟", "3 小时 20 分钟" in h, f"got={h!r}")
    d = time_anchor(now, (now - timedelta(days=2)).isoformat())
    check("天档：按天计", "2 天" in d, f"got={d!r}")
    check("_humanize_gap：不到 1 分钟", _humanize_gap(timedelta(seconds=30)) == "不到 1 分钟")
    check("_humanize_gap：纯小时不漏分钟", _humanize_gap(timedelta(hours=8)) == "8 小时")

    # ── 4. 坏时间戳降级：只给绝对时间，不抛 ─────────────────
    bad = time_anchor(now, "不是时间")
    check("坏时间戳：降级为只给绝对时间", "已过" not in bad and "2026-10-13" in bad)

    # ── 5. build_messages 注入：无历史 → 只绝对时间 ─────────
    p = make_persona()
    p.llm.replies = ["（抬眼）嗯。"]
    built = p.build_messages("在吗")
    sys_content = built[0]["content"]
    check("注入：system 首条含【此刻】",
          built[0]["role"] == "system" and "【此刻】" in sys_content)
    check("注入：无历史不含相对时间", "已过" not in sys_content)

    # ── 6. 有历史（隔了很久）→ 带相对时间 ───────────────────
    p._last_turn_time = lambda: "2026-10-13T18:00:00"
    p.time_anchor_block = lambda now=None: time_anchor(
        datetime(2026, 10, 13, 23, 10), p._last_turn_time())
    sys2 = p.build_messages("我回来了")[0]["content"]
    check("注入：有历史时给出相对时间", "已过 5 小时" in sys2, f"got={sys2[-120:]!r}")

    print()
    print(f"通过 {sum(PASS)}/{len(PASS)}")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
