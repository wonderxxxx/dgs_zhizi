#!/usr/bin/env python3
"""聊天记录回填验证：同一 (user_id, character) 打开会话即拉到此前往来。

覆盖：Persona.get_history 正文/动作拆分 / 压缩产生的 system 摘要行不上界面 /
      limit 取最近 N 条 / 跨进程重启仍在（落盘）/ 用户与角色二级隔离 /
      无记忆角色退回进程内窗口（persisted=False）/ GET /history 路由与鉴权。
运行方式：cd zhizi && .venv/bin/python tests/test_history.py
"""

import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.users import UserManager  # noqa: E402
from api import make_handler  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZHIZI_MD = os.path.join(BASE, "prompts", "zhizi_v1.md")
BAIOU_TXT = os.path.join(BASE, "prompts", "白鸥人物卡-万物皆数定制版.txt")
IDENTITY = os.path.join(BASE, "visual", "identity.json")

PASS = []


def check(name, cond, detail=""):
    status = "✅" if cond else "❌"
    PASS.append(cond)
    print(f"{status} {name}" + (f"  ({detail})" if detail and not cond else ""))


class FakeLLM:
    """fake LLM：回复里带全角动作描写，顺带验证历史拆分。"""

    model = "fake"
    multimodal = False

    def __init__(self, config=None):
        pass

    def chat(self, messages, temperature=None, images=None, enable_thinking=None):
        return "（托腮）嗯，哥哥回来了。（笑）"


def make_cfg(tmp):
    return {
        "characters": {
            "千夜智子": {
                "prompt": ZHIZI_MD,
                "visual_identity": IDENTITY,
                "memory": True,
                "default": True,
            },
            "白鸥": {
                "prompt": BAIOU_TXT,
                "visual_identity": "",
                "memory": False,
            },
        },
        "memory": {"note_file": os.path.join(tmp, "notes"), "top_k": 5},
        "provider": {"type": "local", "base_url": "http://x", "model": "fake",
                     "api_key": "x", "temperature": 0.85},
    }


def seed_recent(tmp, user_id, char_dir, messages):
    """直接写工作记忆 JSON（模拟历史进程留下的落盘数据）。"""
    d = os.path.join(tmp, "notes", user_id, char_dir, "memory", "recent")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "recent_default.json"), "w", encoding="utf-8") as fh:
        json.dump(messages, fh, ensure_ascii=False)


def get(url, api_key=""):
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def main():
    tmp = tempfile.mkdtemp(prefix="zhizi_hist_")
    cfg = make_cfg(tmp)
    um = UserManager(cfg)

    # ── 1. 落盘数据：正文/动作拆分 + system 摘要行不上界面 ──────
    seed_recent(tmp, "alice", "", [
        {"role": "system", "content": "之前的对话摘要: 旧内容", "timestamp": "2026-09-01T10:00:00"},
        {"role": "user", "content": "哥哥回来了吗", "timestamp": "2026-09-01T10:00:01"},
        {"role": "assistant", "content": "（抬头）嗯，在的。（笑）",
         "timestamp": "2026-09-01T10:00:09"},
        {"role": "user", "content": "我买了两包薯片", "timestamp": "2026-09-01T10:01:00"},
        {"role": "assistant", "content": "（皱眉）又吃薯片。", "timestamp": "2026-09-01T10:01:05"},
    ])
    h = um.entry("alice")["persona"].get_history()
    msgs = h["messages"]
    check("落盘历史：4 条（system 摘要行被滤掉）", len(msgs) == 4, f"got={len(msgs)}")
    check("落盘历史：正序 user/assistant 交替",
          [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"],
          f"roles={[m['role'] for m in msgs]}")
    check("落盘历史：助手回复拆出动作",
          msgs[1]["actions"] == ["抬头", "笑"] and msgs[1]["content"] == "嗯，在的。",
          f"{msgs[1]}")
    check("落盘历史：带时间戳供壳子显示", msgs[0]["time"] == "2026-09-01T10:00:01")
    check("落盘历史：persisted=True", h["persisted"] is True)

    # ── 2. limit：取最近 N 条 ────────────────────────────────
    h2 = um.entry("alice")["persona"].get_history(limit=2)
    check("limit=2：只回最近两条且仍是正序",
          [m["content"] for m in h2["messages"]] == ["我买了两包薯片", "又吃薯片。"],
          f"got={[m['content'] for m in h2['messages']]}")

    # ── 3. 跨进程重启：新的 UserManager 从落盘读回 ────────────
    um2 = UserManager(make_cfg(tmp))
    h3 = um2.entry("alice")["persona"].get_history()
    check("重启后：同一 user_id × 角色历史仍在", len(h3["messages"]) == 4,
          f"got={len(h3['messages'])}")

    # ── 4. 二级隔离：其他用户 / 其他角色看不到 ────────────────
    check("隔离：其他用户为空", um2.entry("bob")["persona"].get_history()["messages"] == [])
    baiou_z = um2.entry("alice", "白鸥")["persona"].get_history()
    check("隔离：同用户换角色为空", baiou_z["messages"] == [])
    check("隔离：白鸥不落盘（persisted=False）", baiou_z["persisted"] is False)

    # ── 5. 无记忆角色：退回进程内窗口 ────────────────────────
    b = um2.entry("alice", "白鸥")["persona"]
    b.llm = FakeLLM()          # 注入 fake LLM：进程内窗口要靠真实 reply 写入
    b.reply_structured("我现在在危险区")
    hb = b.get_history()
    check("无记忆角色：进程内窗口仍可回填 2 条", len(hb["messages"]) == 2,
          f"got={len(hb['messages'])}")
    check("无记忆角色：仍标 persisted=False 提示壳子", hb["persisted"] is False)

    # ── 6. GET /history 路由 + 鉴权 ──────────────────────────
    def serve(api_key):
        srv = ThreadingHTTPServer(("127.0.0.1", 0),
                                  make_handler(um2, api_key, dashboard_enabled=False))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, f"http://127.0.0.1:{srv.server_address[1]}/history?user_id=alice" \
                    f"&character={urllib.parse.quote('千夜智子')}&limit=10"

    # 未开鉴权：裸读即可
    srv, url = serve("")
    try:
        code, body = get(url)
        check("/history：未开鉴权 → 200 + 带回该 user × 角色的历史",
              code == 200 and len(body["messages"]) == 4
              and body["user_id"] == "alice" and body["character"] == "千夜智子",
              f"code={code} body={str(body)[:120]}")
    finally:
        srv.shutdown()
        srv.server_close()

    # 开了鉴权：带对 Key 放行，裸读 401（对话内容是私事，不比 /chat 更松）
    srv, url = serve("s3cret")
    try:
        code, body = get(url, api_key="s3cret")
        check("/history：带对 Key → 200", code == 200 and len(body["messages"]) == 4,
              f"code={code}")
        try:
            get(url)
            check("/history：开了鉴权裸读 → 401", False, "居然放行了")
        except urllib.error.HTTPError as e:
            check("/history：开了鉴权裸读 → 401", e.code == 401, f"code={e.code}")
    finally:
        srv.shutdown()
        srv.server_close()

    print()
    print(f"通过 {sum(PASS)}/{len(PASS)}")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
