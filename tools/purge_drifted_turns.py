#!/usr/bin/env python3
"""定向删除 10-14 被污染的两轮对话（工作记忆 + 时间索引联动回滚）。

背景：那两轮回复犯了两类错——
  17:15 轮：身份漂移（「公司裁员没裁到我头上」「去人事部」）+ 4 个括号分镜
  17:51 轮：再次擦边（「黑掉 HR 的裁员名单」「工牌自然消失」）
它们留在工作记忆里会每轮原样注入 system，等于持续 few-shot 示范错误写法，
新红线被历史例子压住；时间索引里同内容还会以「相关往事」被 BM25 召回。

facts/reflections 不动：全是关于用户的准确事实，正是事实召回链路的测试素材。

用法：
  .venv/bin/python tools/purge_drifted_turns.py --dry    # 预览将删内容
  .venv/bin/python tools/purge_drifted_turns.py          # 执行（自动备份）
"""

import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory.timeindex import TimeIndexedMemory  # noqa: E402

NOTE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "notes", "dgs3")
USER_ID = "default"
# 只删这两轮：识别条件是 10-14 当天的全部消息（该日仅这两轮）
TARGET_DATE = "2026-10-14"


def preview(text, n=70):
    one = " ".join((text or "").split())
    return one[:n] + ("…" if len(one) > n else "")


def main():
    dry = "--dry" in sys.argv
    recent_path = os.path.join(NOTE_DIR, "memory", "recent", "recent_default.json")
    with open(recent_path, encoding="utf-8") as fh:
        msgs = json.load(fh)

    doomed = [m for m in msgs
              if (m.get("timestamp") or "").startswith(TARGET_DATE)]
    if not doomed:
        print(f"没有 {TARGET_DATE} 的消息，无需清理。")
        return
    if len(doomed) != 4 or doomed[0].get("role") != "user" \
            or doomed[-1].get("role") != "assistant":
        print(f"!! {TARGET_DATE} 共 {len(doomed)} 条，与预期的 4 条（两问两答）不符，中止。")
        sys.exit(1)

    print(f"将删除 {len(doomed)} 条（两轮）：")
    for m in doomed:
        print(f"  [{m['role']:9}] {m.get('timestamp', '')[:19]}  {preview(m['content'])}")

    if dry:
        print("\n--dry：未做任何修改。")
        return

    # 备份再写：一次性脚本也不裸奔
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = recent_path + f".bak-{stamp}"
    shutil.copy2(recent_path, backup)
    keep = [m for m in msgs if m not in doomed]
    with open(recent_path, "w", encoding="utf-8") as fh:
        json.dump(keep, fh, ensure_ascii=False, indent=1)
    print(f"\n工作记忆：{len(msgs)} → {len(keep)} 条（备份 {os.path.basename(backup)}）")

    # 时间索引按内容回滚（与 persona._forget_timeindex 同一语义：
    # 剩余工作记忆里还有的内容不删——同内容可能被别的轮次引用）
    ti = TimeIndexedMemory(os.path.join(NOTE_DIR, "memory", "timeindex"))
    live = {(m.get("content") or "").strip() for m in keep}
    removed = 0
    for m in doomed:
        text = (m.get("content") or "").strip()
        if not text or text in live:
            continue
        if ti.forget(USER_ID, text, category="conversation"):
            removed += 1
    ti.close()
    print(f"时间索引：回滚 {removed} 条")
    print("facts / reflections 未改动。")


if __name__ == "__main__":
    main()
