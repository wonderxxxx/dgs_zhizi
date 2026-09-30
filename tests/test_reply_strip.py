#!/usr/bin/env python3
"""回复正文清洗：思考通道 / 控制符不能漏进上下文和记忆。

背景：gemma-4 的 channel 模型有两套思考标记，两套都得认——
- 模板态 <|channel|>thought\\n<channel|>正文（chat_template 的 strip_thinking 宏那套）
- 生成态 <|channel|>analysis<|message|>思考<|end|>…<|channel|>final<|message|>正文
生成态带 <|message|> 且用 <|end|> 收尾，没有无管道的 <channel|>，只按模板态
那套切会一个标记都匹配不到，2000 字英文思考原文会整段当回复落进工作记忆和
时间索引（真实污染样本见 tests 末尾）。

覆盖：两套标记的剥离 / 非思考通道保留 / 不误伤正常正文与 <message|> /
      Qwen 的 thinking…response 变体 / 空回复（只有思考没有正文）。
运行方式：.venv/bin/python tests/test_reply_strip.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import llm as L  # noqa: E402

PASS = []

O, M, E = L._CHANNEL_OPEN, L._CHANNEL_BODY, L._CHANNEL_END
T = L._THINK_END


def check(name, cond, detail=""):
    cond = bool(cond)
    PASS.append(cond)
    print(f"{'✅' if cond else '❌'} {name}" + (f"  ({detail})" if detail and not cond else ""))


def main():
    cases = [
        # ── 模板态 ────────────────────────────────────────────
        ("模板态 thought", f"{O}thought\n{T}（点头）嗯。", "（点头）嗯。"),
        ("模板态多段交替", f"{O}thought\n{T}甲{T}乙", "甲乙"),
        # ── 生成态（带 <|message|> / <|end|> 的完整协议）────
        ("生成态 analysis→final",
         f"{O}analysis{M}我在想…{E}{O}start{O}final{M}答。", "答。"),
        ("生成态 通道名 thought",
         f"{O}thought{M}思考{E}{O}final{M}正文在这", "正文在这"),
        ("非思考通道也保留",
         f"{O}analysis{M}思考{E}{O}tool_call{M}{{}}{E}{O}final{M}答。", "{}答。"),
        ("尾随结构标记", f"{O}final{M}好。{E}{O}start", "好。"),
        ("只有思考没有正文", f"{O}analysis{M}想了半天{E}", ""),
        # ── 不该误伤 ─────────────────────────────────────────
        ("纯正文", "（把薯片袋捏得咯吱作响）好吃。", "（把薯片袋捏得咯吱作响）好吃。"),
        ("残留 turn/eos", "（挑眉）<turn|><eos>", "（挑眉）"),
        ("正文含 <message|>", "（歪头）<message|> huh", "（歪头）<message|> huh"),
        ("空", "", ""),
        ("None", None, ""),
    ]
    for name, src, want in cases:
        got = L._strip_mlx_reply(src)
        check(f"剥离：{name}", got == want, f"got={got!r} want={want!r}")

    # 标记常量本身：曾把 <|channel|> 写成少一个尾管的 10 字符版本，
    # 导致模板态的思考段一个都匹配不到（旧 bug 的根因）
    check("常量：<|channel|> 必须带尾管（11 字符）",
          L._CHANNEL_OPEN == f"<{'|'}channel{'|'}>" and len(L._CHANNEL_OPEN) == 11,
          f"{L._CHANNEL_OPEN!r} len={len(L._CHANNEL_OPEN)}")
    check("常量：<channel|> 是 10 字符的闭合标记", len(L._THINK_END) == 10)

    # ── Qwen 的 thinking…response 变体（多模态路径）──────────
    check("Qwen：thinking/response 分段",
          L._strip_vlm_reply("thinking\n先想想\nresponse\n\n这是正文。") == "这是正文。")
    check("Qwen：只有思考没有正文 → 空",
          L._strip_vlm_reply("thinking\n只有思考没正文") == "")
    check("Qwen：普通回复原样返回",
          L._strip_vlm_reply("（挑眉）嗯。") == "（挑眉）嗯。")

    # ── 真实污染样本的结构 ──────────────────────────────────
    leaked = (f"{O}analysis{M}We need to respond as 千夜智子, short, natural. "
              f"She is a sister living with the brother.{E}"
              f"{O}start{O}assistant{O}final{M}"
              f"（把薯片放进嘴里）\n“我这边没看到啊。”")
    got = L._strip_mlx_reply(leaked)
    check("污染样本：英文思考原文不进正文", "We need to" not in got, f"got={got!r}")
    check("污染样本：只剩 final 正文",
          got == "（把薯片放进嘴里）\n“我这边没看到啊。”", f"got={got!r}")
    check("污染样本：正文不残留控制符",
          not any(m in got for m in ("<|", f"<{'|'}channel{'|'}>", "<turn|>")))

    print(f"\n通过 {sum(PASS)}/{len(PASS)}")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
