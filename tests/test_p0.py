#!/usr/bin/env python3
"""P0 记忆升级验证脚本：运行方式  cd zhizi && .venv/bin/python tests/test_p0.py

覆盖：中文检索 / 去重 / importance / 时间过滤 / JSONL 迁移 / 用户隔离 / 持久化 /
      _parse_note 解析 / persona 全链路冒烟（fake LLM）。
全部使用临时目录，不污染工程数据。
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import NoteStore
from core.persona import Persona, _parse_note

PASS = []


def check(name, cond, detail=""):
    status = "✅" if cond else "❌"
    PASS.append(cond)
    print(f"{status} {name}" + (f"  ({detail})" if detail and not cond else ""))


def fake_llm_factory(notes):
    """fake LLM：chat 返回固定回复；额外提供笔记回放（模拟智子写笔记）。"""
    class FakeLLM:
        model = "fake"
        def __init__(self, config=None):
            self.replay = iter(notes)
        def chat(self, messages, temperature=None):
            try:
                return next(self.replay)
            except StopIteration:
                return "（沉默片刻）嗯。"
    return FakeLLM()


def main():
    tmp = tempfile.mkdtemp(prefix="zhizi_p0_")

    # ── 1. 中文检索 + BM25 ──────────────────────────────────────
    store = NoteStore(os.path.join(tmp, "u1"))
    store.add("哥哥喜欢薯片，薯片不离手", context="聊到零食")
    store.add("哥哥在看苏小野的直播，看得很入迷", context="智子一脸嫌弃")
    store.add("哥哥承诺周末带我去吃火锅", context="约定")
    store.add("熬夜写代码，凌晨两点才睡", context="哥哥的工作状态")

    r = store.search("薯片")
    check("中文检索：查「薯片」命中", len(r) >= 1 and "薯片" in r[0]["text"], f"top1={r[0]['text'] if r else '无'}")
    r2 = store.search("直播")
    check("中文检索：查「直播」命中", len(r2) >= 1 and "直播" in r2[0]["text"], f"top1={r2[0]['text'] if r2 else '无'}")
    r3 = store.search("火锅 周末")
    check("多词查询「火锅 周末」命中约定", any("火锅" in n["text"] for n in r3), f"top={[n['text'] for n in r3]}")
    r4 = store.search("不相关话题xyz")
    check("无命中返回空列表", r4 == [])
    check("search 返回字段含 importance", "importance" in r[0] and "time" in r[0] and "context" in r[0])

    # ── 2. importance：clamp + 排名微调 ─────────────────────────
    store.add("极其重要：哥哥的生日是 9 月 20 日", importance=99)  # 越界截断
    note = store.search("生日")[0]
    check("importance 越界截断到 10", note["importance"] == 10, f"got={note['importance']}")
    store.add("随手一记", importance=0)
    note = store.search("随手")[0]
    check("importance 下限截断到 1", note["importance"] == 1, f"got={note['importance']}")

    # ── 3. hash 去重：同文本只保留一条，重要度取 max ───────────
    before = len(store)
    store.add("哥哥承诺周末带我去吃火锅", importance=3)  # 文本规范化后相同
    after = len(store)
    check("精确去重：重复文本不新增", after == before, f"{before}→{after}")
    dup = store.search("火锅")[0]
    check("重复后重要度取 max", dup["importance"] >= 5, f"got={dup['importance']}")

    # ── 4. 时间过滤 ────────────────────────────────────────────
    t0 = store.all()[0]["time"]
    filtered = store.search("哥哥", since="2099-01-01")
    check("since 过滤：未来时间无命中", filtered == [])
    check("all() 按时间倒序", store.all()[0]["time"] >= store.all()[-1]["time"])

    # ── 5. 持久化：重开实例数据还在 ────────────────────────────
    store2 = NoteStore(os.path.join(tmp, "u1"))
    check("持久化：重开后笔记数一致", len(store2) == len(store), f"{len(store2)}/{len(store)}")
    check("持久化：重开后检索可用", "薯片" in store2.search("薯片")[0]["text"])

    # ── 6. JSONL 迁移 ──────────────────────────────────────────
    old_dir = os.path.join(tmp, "u_old")
    os.makedirs(old_dir, exist_ok=True)
    with open(os.path.join(old_dir, "notes.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": "2026-09-01T10:00:00", "text": "旧笔记：哥哥喜欢熬夜", "context": "迁移测试"}, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"time": "2026-09-02T10:00:00", "text": "旧笔记：哥哥不吃香菜", "context": "迁移测试2"}, ensure_ascii=False) + "\n")
    store3 = NoteStore(old_dir)
    check("迁移：旧 JSONL 全部导入", len(store3) == 2, f"got={len(store3)}")
    check("迁移：检索旧笔记", "香菜" in store3.search("香菜")[0]["text"])
    check("迁移：原文件改名保留", os.path.exists(os.path.join(old_dir, "notes.jsonl.migrated")) and not os.path.exists(os.path.join(old_dir, "notes.jsonl")))
    store4 = NoteStore(old_dir)
    check("迁移幂等：重开不重复导入", len(store4) == 2)

    # ── 7. 用户隔离（模拟 UserManager 的目录隔离） ─────────────
    ua = NoteStore(os.path.join(tmp, "alice"))
    ub = NoteStore(os.path.join(tmp, "bob"))
    ua.add("alice 的暗号：月光")
    check("用户隔离：bob 检索不到 alice 的笔记", ub.search("月光") == [] and len(ua) == 1 and len(ub) == 0)

    # ── 8. _parse_note 解析 ────────────────────────────────────
    t, imp = _parse_note("观测笔记：哥哥说周末想吃火锅\n重要度：8")
    check("解析：文本+重要度", t == "哥哥说周末想吃火锅" and imp == 8, f"{t!r}/{imp}")
    t2, imp2 = _parse_note("观测笔记：哥哥最近压力大")
    check("解析：无重要度默认 5", t2 == "哥哥最近压力大" and imp2 == 5)
    t3, imp3 = _parse_note("重要度： 12\n观测笔记：测试")
    check("解析：重要度截断", imp3 == 10 and "测试" in t3, f"{t3!r}/{imp3}")
    t4, imp4 = _parse_note("")
    check("解析：空输入", t4 == "" and imp4 == 5)

    # ── 9. persona 全链路冒烟（fake LLM：回复 + 写笔记） ───────
    cfg = {
        "system_prompt": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "zhizi_v1.md"),
        "memory": {"note_file": os.path.join(tmp, "smoke"), "top_k": 5},
        "provider": {"base_url": "x", "model": "fake", "api_key": "x", "temperature": 0.85},
    }
    fake = fake_llm_factory([
        "（歪头）哥哥终于想起我啦！",                    # 对话回复
        "观测笔记：哥哥说想带我吃火锅\n重要度：7",       # 笔记生成
    ])
    persona = Persona(cfg, llm=fake)
    result = persona.reply_structured("哥哥回来了")
    check("全链路：reply_structured 返回结构", "reply" in result and "actions" in result and "raw" in result, f"{result.get('reply','')[:20]}")
    check("全链路：笔记已写入且带重要度", len(persona.memory) == 1 and persona.memory.all()[0]["importance"] == 7)

    store.close(); store2.close(); store3.close(); store4.close()
    ua.close(); ub.close()

    print()
    print(f"通过 {sum(PASS)}/{len(PASS)}")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
