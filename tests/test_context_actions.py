#!/usr/bin/env python3
"""注入上下文的动作剥离：模型只看得到自己的台词，看不到旧的括号动作。

背景：工作记忆回灌「最近对话」时若原样带上 assistant 原文，历史里的招牌
动作（连着几轮都"捏薯片袋"）会被模型当范式照抄，形成"每条回复都在吃薯片"
的自我强化复读。修复：get_context 回灌前剥掉 assistant 的括号动作；
落盘与展示层（get_messages / Persona.get_history）不受影响；用户消息里的
动作是读者的戏份，原样保留；剥空了（纯动作回复）退回原文兜底。

覆盖：assistant 剥离 / user 保留 / 纯动作回复兜底 / 展示层原文 /
      manager 级「最近对话」注入同样剥离 / system 摘要行不受影响。
运行方式：cd zhizi && .venv/bin/python tests/test_context_actions.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory.manager import MemoryManager  # noqa: E402
from core.memory.recent import RecentMemory  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    cond = bool(cond)
    PASS.append(cond)
    print(f"{'✅' if cond else '❌'} {name}" + (f"  ({detail})" if not cond else ""))


def main():
    tmp = tempfile.mkdtemp(prefix="zhizi_ctx_")

    # ── 1. RecentMemory.get_context：assistant 剥动作，user 保留 ──
    r = RecentMemory(memory_dir=os.path.join(tmp, "recent"))
    r.add_message("default", "user", "（从背包里掏出两包薯片）给你的")
    r.add_message("default", "assistant", "（把薯片袋捏得咔啦响）哥哥偏心。")
    r.add_message("default", "user", "今天冷不冷")
    r.add_message("default", "assistant", "（蜷在沙发角落）还行，风不大。")
    ctx = r.get_context("default")

    check("回灌：assistant 括号动作被剥掉", "薯片袋捏得" not in ctx
          and "蜷在沙发角落" not in ctx, f"ctx={ctx!r}")
    check("回灌：assistant 台词保留", "哥哥偏心。" in ctx and "还行，风不大。" in ctx,
          f"ctx={ctx!r}")
    check("回灌：user 动作原样保留（读者戏份）",
          "从背包里掏出两包薯片" in ctx, f"ctx={ctx!r}")

    # ── 2. 纯动作回复：剥空了退回原文，不留空行 ────────────────
    r.add_message("default", "assistant", "（僵住）")
    ctx2 = r.get_context("default")
    check("纯动作回复：兜底回原文", "assistant: （僵住）" in ctx2, f"ctx={ctx2!r}")

    # ── 3. 展示层与落盘不受影响 ────────────────────────────────
    msgs = r.get_messages("default")
    check("展示层：get_messages 仍是原文",
          msgs[1]["content"] == "（把薯片袋捏得咔啦响）哥哥偏心。"
          and msgs[3]["content"] == "（蜷在沙发角落）还行，风不大。",
          f"got={msgs[1]['content']!r}")

    # ── 4. manager 级：注入 system 的「最近对话」同样剥离 ────────
    mm = MemoryManager(memory_dir=os.path.join(tmp, "memory"), llm_client=None)
    mm.recent.add_message("default", "user", "我买了两包薯片")
    mm.recent.add_message("default", "assistant", "（皱眉）又吃薯片。")
    context, trace = mm.get_context_traced(user_id="default", query="薯片")
    check("注入上下文：最近对话段无 assistant 动作", "皱眉" not in context,
          f"ctx={context!r}")
    check("注入上下文：台词仍在", "又吃薯片。" in context, f"ctx={context!r}")
    recent_dim = trace["dimensions"][0]
    check("召回 trace：观测界面仍显示原文",
          recent_dim["items"][-1]["content"] == "（皱眉）又吃薯片。",
          f"got={recent_dim['items'][-1]!r}")

    # ── 5. system 摘要行（压缩产物）不被误伤 ────────────────────
    r.add_message("default", "system", "之前的对话摘要: 她在吃薯片（动作描写）")
    ctx3 = r.get_context("default")
    check("system 摘要行：原样注入", "她在吃薯片（动作描写）" in ctx3, f"ctx={ctx3!r}")

    print(f"\n通过 {sum(PASS)}/{len(PASS)}")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
