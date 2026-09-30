#!/usr/bin/env python3
"""清理时间索引里的孤儿条目。

孤儿 = category='conversation' 但内容已不在工作记忆（recent_default.json）里的条目。
成因：重新生成只回滚了工作记忆（旧实现），以及工作记忆超过 max_items 被压缩时
丢弃的旧消息。它们仍参与 BM25 召回——聊到相关话题时召回来的可能是用户早已
放弃的回答，措辞和聊天记录对不上。

判定用「内容前缀包含」而非严格相等：工作记忆存原文，时间索引存
with_caption(原文, caption)，带图轮次两者不完全一致。

用法：.venv/bin/python tools/purge_timeindex_orphans.py [--dry]
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory.timeindex import TimeIndexedMemory  # noqa: E402

NOTES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")
PREFIX = 25   # 与内容去重粒度一致：太短的正文容易撞车


def live_contents(note_dir):
    """该用户工作记忆里所有仍在的正文。"""
    path = os.path.join(note_dir, "memory", "recent", "recent_default.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            msgs = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    return [(m.get("content") or "").strip() for m in msgs]


def is_orphan(content, live):
    text = (content or "").strip()
    if not text:
        return False
    for other in live:
        if text == other:
            return False
        if len(text) > PREFIX and text[:PREFIX] in other:
            return False
        if len(other) > PREFIX and other[:PREFIX] in text:
            return False
    return True


def main():
    dry = "--dry" in sys.argv
    total = 0
    for user in sorted(os.listdir(NOTES)):
        note_dir = os.path.join(NOTES, user)
        ti_dir = os.path.join(note_dir, "memory", "timeindex")
        if not os.path.isfile(os.path.join(ti_dir, "timeindex.db")):
            continue
        live = live_contents(note_dir)
        index = TimeIndexedMemory(ti_dir)
        try:
            docs = index.get_by_category("default", "conversation")
            orphans = [d for d in docs if is_orphan(d.get("content"), live)]
            if not orphans:
                index.close()
                continue
            print(f"\n{user}: conversation {len(docs)} 条 → 孤儿 {len(orphans)} 条")
            gone = 0
            for doc in orphans:
                mark = "预删" if dry else "已删"
                if not dry and index.forget("default", doc["content"],
                                            category="conversation"):
                    gone += 1
                head = (doc.get("content") or "").replace("\n", " ")[:56]
                print(f"  {mark} #{doc['id']} {doc.get('time', '')[:16]} {head!r}")
            total += gone if not dry else len(orphans)
            index.close()
        except Exception as exc:
            print(f"{user}: 出错 {exc}")
            try:
                index.close()
            except Exception:
                pass
    verb = "预计清理" if dry else "共清理"
    print(f"\n{verb} {total} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
