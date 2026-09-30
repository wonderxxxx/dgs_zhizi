#!/usr/bin/env python3
"""清洗工作记忆与时间索引里被思考原文污染的助手回复。

命中条件：内容含 channel 协议标记（<|channel|> / <channel|> / <|message|>）
或典型思考口吻（英文 "We need to" 开头的推理段落）。用当前 core.llm 的
_strip_mlx_reply 重算一遍，剥得掉就换成干净正文，剥成空则整条删除。

时间索引按内容删（旧内容），重写后把干净正文重新入索引，保持两端一致。

用法：.venv/bin/python tools/scrub_thinking_leak.py [--dry]
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import _strip_mlx_reply  # noqa: E402
from core.memory.timeindex import TimeIndexedMemory  # noqa: E402

NOTES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")
LEAK_MARKS = ("<|channel|>", "<channel|>", "<|message|>", "We need to respond",
              "Let's produce", "Ok let me")


def looks_leaked(text):
    return any(m in text for m in LEAK_MARKS)


def main():
    dry = "--dry" in sys.argv
    fixed = dropped = 0
    for user in sorted(os.listdir(NOTES)):
        note_dir = os.path.join(NOTES, user)
        rpath = os.path.join(note_dir, "memory", "recent", "recent_default.json")
        tdir = os.path.join(note_dir, "memory", "timeindex")
        if not os.path.isfile(rpath) or not os.path.isfile(os.path.join(tdir, "timeindex.db")):
            continue
        with open(rpath, encoding="utf-8") as fh:
            msgs = json.load(fh)
        hits = [m for m in msgs
                if m.get("role") == "assistant" and looks_leaked(m.get("content") or "")]
        if not hits:
            continue
        index = TimeIndexedMemory(tdir)
        try:
            print(f"\n{user}: 命中 {len(hits)} 条")
            keep = []
            for m in msgs:
                if m.get("role") != "assistant" or not looks_leaked(m.get("content") or ""):
                    keep.append(m)
                    continue
                raw = m.get("content") or ""
                clean = _strip_mlx_reply(raw)
                if not clean:
                    print(f"  删除 {len(raw)} 字：剥后为空，整条都是思考")
                    if not dry:
                        index.forget("default", raw, category="conversation")
                    dropped += 1
                    continue
                print(f"  重写 {len(raw)} → {len(clean)} 字: {clean[:48]!r}")
                if not dry:
                    index.forget("default", raw, category="conversation")
                    index.add("default", clean, category="conversation", importance=3)
                fixed += 1
                keep.append({**m, "content": clean})
            if not dry and (fixed or dropped):
                with open(rpath, "w", encoding="utf-8") as fh:
                    json.dump(keep, fh, ensure_ascii=False, indent=2)
                print(f"  已写回 {rpath}")
            index.close()
        except Exception as exc:
            print(f"  出错：{exc}")
            try:
                index.close()
            except Exception:
                pass
    if not fixed and not dropped:
        print("无污染记录")
    else:
        verb = "预计" if dry else "共"
        print(f"\n{verb}重写 {fixed} 条 / 删除 {dropped} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
