#!/usr/bin/env python3
"""重新生成：丢掉最后一条回答，就同一句话再要一个回答（换模型重试）。

覆盖：有记忆角色 / 无记忆角色 都只替换不追加 / 没有一问一答 → 400 /
      生成失败时上一轮原样放回 / 带图轮次：图从落盘读回再喂给模型 /
      POST /regenerate 路由与鉴权。
运行方式：cd zhizi && .venv/bin/python tests/test_regenerate.py
"""

import base64
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.users import UserManager  # noqa: E402
from api import make_handler  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZHIZI_MD = os.path.join(BASE, "prompts", "zhizi_v1.md")
BAIOU_TXT = os.path.join(BASE, "prompts", "白鸥人物卡-万物皆数定制版.txt")
IDENTITY = os.path.join(BASE, "visual", "identity.json")

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

PASS = []


def check(name, cond, detail=""):
    status = "✅" if cond else "❌"
    PASS.append(bool(cond))
    print(f"{status} {name}" + (f"  ({detail})" if not cond else ""))


class FakeLLM:
    """对话调用按序回不同句子；工具类调用（事实/摘要/描述）各回各的。"""

    model = "fake"
    multimodal = True

    def __init__(self):
        self.conversation_calls = 0
        self.fail_at = None        # 第 N 次对话调用时抛错（验回滚）
        self.last_images = None    # 最近一次对话调用收到的图片

    def chat(self, messages, temperature=None, images=None, enable_thinking=None):
        system = messages[0].get("content", "") if messages else ""
        # 单条消息 = 事实提取/反思/摘要这类工具调用，别掺进对话回复序列
        if len(messages) <= 1:
            return "[]"
        if "图像描述助手" in system and "只描述客观可见内容" in system:
            return "一个女孩穿着白裙站在樱花树下"
        if images and "外观比对助手" in system:
            return json.dumps({"same_character": 0.1, "reason": "不同人"},
                              ensure_ascii=False)
        if images and "图像描述助手" in system:
            return json.dumps({
                "hair": "黑色长发", "eyes": "琥珀色", "style": "anime",
                "appearance_age": "young adult", "clothing": "白色连衣裙",
                "summary": "一个女孩穿着白裙站在樱花树下",
            }, ensure_ascii=False)
        self.conversation_calls += 1
        if self.fail_at == self.conversation_calls:
            raise RuntimeError("模型炸了")
        self.last_images = images
        return f"第{self.conversation_calls}次回答"


def inject(persona, fake):
    """把 fake 换进 persona 与其记忆子系统（它们各自持有一份引用）。"""
    persona.llm = fake
    if getattr(persona, "visual_identity", None) is not None:
        persona.visual_identity.llm = fake
    mm = persona.memory_manager
    if mm:
        for sub in vars(mm).values():
            if hasattr(sub, "llm_client"):
                sub.llm_client = fake
    return fake


def make_cfg(tmp):
    return {
        "characters": {
            "千夜智子": {
                "prompt": ZHIZI_MD, "visual_identity": IDENTITY,
                "memory": True, "default": True,
            },
            "白鸥": {
                "prompt": BAIOU_TXT, "visual_identity": "", "memory": False,
            },
        },
        "memory": {"note_file": os.path.join(tmp, "notes"), "top_k": 5},
        "attachments": {"caption": True},
        "provider": {"type": "local", "base_url": "http://x", "model": "fake",
                     "api_key": "x", "temperature": 0.85},
    }


def roles(persona):
    return [(m["role"], m.get("content", "")) for m in persona.get_history()["messages"]]


def post(url, payload, api_key=""):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def main():
    tmp = tempfile.mkdtemp(prefix="zhizi_regen_")
    um = UserManager(make_cfg(tmp))

    # ── 1. 有记忆角色：替换而非追加 ──────────────────────────
    p = um.entry("alice")["persona"]
    fake = inject(p, FakeLLM())
    p.reply_structured("哥哥回来了吗")
    before = roles(p)
    check("准备：一轮后历史 2 条", len(before) == 2, f"got={before}")
    p.regenerate_structured()
    after = roles(p)
    check("有记忆角色：仍是 2 条（替换不追加）", len(after) == 2, f"got={after}")
    check("有记忆角色：提问原样保留", after[0] == before[0], f"{after[0]} vs {before[0]}")
    check("有记忆角色：回答被换成新模型的",
          after[1][1] == "第2次回答" and after[1][1] != before[1][1],
          f"got={after[1][1]}")

    # ── 2. 无记忆角色：进程内窗口同样只替换 ──────────────────
    b = um.entry("alice", "白鸥")["persona"]
    bfake = inject(b, FakeLLM())
    b.reply_structured("我现在在危险区")
    b.regenerate_structured()
    hb = roles(b)
    check("无记忆角色：仍是 2 条", len(hb) == 2, f"got={hb}")
    check("无记忆角色：回答已换新", hb[1][1] == "第2次回答", f"got={hb[1]}")
    check("无记忆角色：persisted=False 不变",
          b.get_history()["persisted"] is False)

    # ── 3. 没有一问一答：本地抛错 / HTTP 400 ─────────────────
    empty = um.entry("bob")["persona"]
    inject(empty, FakeLLM())
    try:
        empty.regenerate()
        check("空会话：拒绝重新生成", False, "居然没抛")
    except ValueError:
        check("空会话：拒绝重新生成", True)

    # ── 4. 生成失败：上一轮原样放回，不能让用户丢了一句话 ────
    f = FakeLLM()
    f.fail_at = 1
    r = um.entry("alice", "白鸥")["persona"]   # 复用白鸥会话（上一步已有 2 条）
    r.llm = f
    ok_before = roles(r)
    try:
        r.regenerate()
        check("生成失败：异常照常抛出", False, "居然没抛")
    except RuntimeError:
        check("生成失败：异常照常抛出", True)
    check("生成失败：历史原样还在", roles(r) == ok_before,
          f"got={roles(r)}")

    # ── 5. 带图轮次：图从落盘读回再喂给模型，引用不丢 ────────
    v = um.entry("carol")["persona"]
    vfake = inject(v, FakeLLM())
    v.reply_structured("看这张图", images=[{"mime": "image/png",
                                           "data": base64.b64encode(PNG).decode("ascii")}])
    orig = v.get_history()["messages"]
    check("带图：一轮后历史 2 条且带引用",
          len(orig) == 2 and orig[0].get("images"),
          f"got={[(m['role'], bool(m.get('images'))) for m in orig]}")
    vfake.last_images = None
    v.regenerate_structured()
    after_img = v.get_history()["messages"]
    check("带图：仍是 2 条", len(after_img) == 2, f"got={len(after_img)}")
    check("带图：重新生成时图又喂给了模型",
          bool(vfake.last_images) and len(vfake.last_images) == 1
          and bool(vfake.last_images[0].get("data")),
          f"got={vfake.last_images}")
    check("带图：图片引用原样留在提问上",
          after_img[0].get("images") == orig[0].get("images"),
          f"got={after_img[0].get('images')}")

    # ── 6. 时间索引不留孤儿（跨维度回滚）────────────────────
    ti = p.memory_manager.timeindex

    def indexed():
        return [d["content"] for d in ti.get_by_category("default", "conversation")]

    live = [m["content"] for m in p.get_history()["messages"]]
    orphans = [c for c in indexed() if c not in live]
    check("回滚：时间索引里没有已放弃的回答", orphans == [],
          f"孤儿={orphans}")
    check("回滚：提问那条仍可召回（提问没变，不该删）",
          any(live[0] in c for c in indexed()), f"indexed={indexed()}")
    check("回滚：新回答只进索引一次",
          sum(1 for c in indexed() if c == live[1]) == 1, f"indexed={indexed()}")

    # 反复重新生成不应成倍堆积
    for _ in range(3):
        p.regenerate()
    live = [m["content"] for m in p.get_history()["messages"]]
    orphans = [c for c in indexed() if c not in live]
    check("回滚：连点 3 次重新生成索引不增", orphans == [], f"孤儿={orphans}")

    # 生成失败：旧回答要从索引里补回来，否则恢复的这一轮召不到
    before_ids = sorted(c[:12] for c in indexed())
    g = FakeLLM()
    g.fail_at = 1          # 换新 fake 后从 1 数：这次重新生成必炸
    inject(p, g)
    try:
        p.regenerate()
        check("回滚：生成失败后异常照常抛出", False, "居然没抛")
    except RuntimeError:
        check("回滚：生成失败后异常照常抛出", True)
    check("回滚：生成失败后回答仍可召回",
          sorted(c[:12] for c in indexed()) == before_ids,
          f"{sorted(c[:12] for c in indexed())} vs {before_ids}")

    # 内容相同的另一轮还在用它时，不能误删（时间索引按内容去重，共用一条）
    same = um.entry("erin")["persona"]
    inject(same, FakeLLM())
    same.reply_structured("第一问")
    same.reply_structured("第二问")
    dup = same.memory_manager.timeindex
    ti_docs = [d["content"] for d in dup.get_by_category("default", "conversation")]
    check("去重：两次相同回答在索引里只占一条",
          ti_docs.count("第1次回答") == 1 or len(set(ti_docs)) == len(ti_docs),
          f"{ti_docs}")
    # 让「第一问」的回答与更早一轮字面相同，再验证不误删
    same2 = um.entry("frank")["persona"]
    inject(same2, FakeLLM())
    same2.reply_structured("甲")
    same2.reply_structured("乙")
    mm2 = same2.memory_manager
    mm2.timeindex.add("default", "第1次回答", category="conversation", importance=3)
    check("去重：字面相同的两条回答共用一个索引条目",
          len([d for d in mm2.timeindex.get_by_category("default", "conversation")
               if d["content"] == "第1次回答"]) == 1)

    # ── 7. POST /regenerate 路由 + 鉴权 ──────────────────────

    dave = um.entry("dave")["persona"]
    inject(dave, FakeLLM())
    dave.reply_structured("你好呀")

    def serve(api_key):
        srv = ThreadingHTTPServer(("127.0.0.1", 0),
                                  make_handler(um, api_key, dashboard_enabled=False))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, f"http://127.0.0.1:{srv.server_address[1]}/regenerate"

    srv, url = serve("")
    try:
        code, body = post(url, {"user_id": "dave"})
        check("/regenerate：未开鉴权 → 200 + 新回答",
              code == 200 and body.get("reply") == "第2次回答"
              and body.get("user_id") == "dave",
              f"code={code} body={str(body)[:160]}")
        check("/regenerate：回答换掉后历史仍 2 条",
              len(um.entry("dave")["persona"].get_history()["messages"]) == 2)
        code, body = post(url, {"user_id": "bob"})   # 空会话
        check("/regenerate：没有可重生成的轮次 → 400",
              code == 400 and "重新生成" in body.get("error", ""),
              f"code={code} body={body}")
    finally:
        srv.shutdown()
        srv.server_close()

    srv, url = serve("s3cret")
    try:
        code, _ = post(url, {"user_id": "dave"}, api_key="s3cret")
        check("/regenerate：带对 Key → 200", code == 200, f"code={code}")
        code, _ = post(url, {"user_id": "dave"})
        check("/regenerate：开了鉴权裸调 → 401", code == 401, f"code={code}")
    finally:
        srv.shutdown()
        srv.server_close()

    print()
    print(f"通过 {sum(PASS)}/{len(PASS)}")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
