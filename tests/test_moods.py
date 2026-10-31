#!/usr/bin/env python3
"""mood 层：稳定卡 + 按轮可变状态段（A 路线最小验证）。

覆盖：moods 注册表清洗 / render_mood 结构 / build_messages 注入位置 /
      reply_structured 显式触发·未知降级·状态随轮流动 / 工作记忆里 mood
      只挂提问不进正文与模型上下文 / regenerate 沿用上一轮状态 /
      POST /chat 显式 mood 透传。
运行方式：cd zhizi && .venv/bin/python tests/test_moods.py
"""

import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import presets  # noqa: E402
from core.characters import CharacterSpec, load_prompt_text  # noqa: E402
from core.persona import Persona  # noqa: E402
from core.users import UserManager  # noqa: E402
from api import make_handler  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZHIZI_MD = os.path.join(BASE, "prompts", "zhizi_v1.md")
IDENTITY = os.path.join(BASE, "visual", "identity.json")

MOODS = {
    "病弱": {
        "状态": "今早起来就发烧，额头烫、嗓子发干，但智子嘴上不认。",
        "语气": "句子更短，句与句之间爱用省略号停顿。",
        "行为倾向": "往人身边靠、要热水和毯子。",
        "退出线": "对方真要叫医生就压下病气打起精神——不是永久人设。",
        "多余键": "会被清洗掉",
    },
    "困倦": {
        "状态": "凌晨还在硬撑，眼皮打架，思维黏糊糊的。",
        "语气": "句子拖尾音、打哈欠漏气。",
        "行为倾向": "往人肩上歪。",
        "退出线": "对方认真劝睡会强撑精神——困只是此刻的状态。",
    },
}

PASS = []


def check(name, cond, detail=""):
    status = "✅" if cond else "❌"
    PASS.append(bool(cond))
    print(f"{status} {name}" + (f"  ({detail})" if not cond else ""))


class ScriptLLM:
    """按序回脚本句子；单条消息（事实提取这类工具调用）恒回 []。"""

    model = "fake"
    multimodal = True

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []   # 对话调用收到的 messages（[0] 是 system）

    def chat(self, messages, temperature=None, images=None, enable_thinking=None):
        system = messages[0].get("content", "") if messages else ""
        if len(messages) <= 1:
            return "[]"
        if "图像描述助手" in system and "只描述客观可见内容" in system:
            return "一个女孩穿着白裙"
        self.calls.append(messages)
        if not self.replies:
            return "（停顿）嗯。"
        return self.replies.pop(0)


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
            "千夜智子": {"prompt": ZHIZI_MD, "visual_identity": IDENTITY,
                         "memory": True, "search": True, "default": True},
        },
        "moods": MOODS,
        "memory": {"note_file": os.path.join(tmp, "notes"), "top_k": 5},
        "attachments": {"caption": True},
        "provider": {"type": "local", "base_url": "http://x", "model": "fake",
                     "api_key": "x", "temperature": 0.85},
        "web_search": {"enabled": True, "top_k": 3,
                       "max_calls_per_turn": 1, "timeout": 5.0},
    }


def main():
    tmp = tempfile.mkdtemp(prefix="zhizi_moods_")

    # ── 1. 注册表清洗 ─────────────────────────────────────
    norm = presets.normalize_mood(MOODS["病弱"])
    check("normalize_mood：按 MOOD_FIELDS 顺序保留已知键",
          list(norm) == ["状态", "语气", "行为倾向", "退出线"],
          f"got={list(norm)}")
    check("normalize_mood：清掉非注册键", "多余键" not in norm)
    check("normalize_mood：非 dict / 空 → {}", presets.normalize_mood(None) == {}
          and presets.normalize_mood("病弱") == {})

    loaded = presets.load_moods({"moods": MOODS, "junk": 1})
    check("load_moods：取到两种状态且清洗过",
          set(loaded) == {"病弱", "困倦"}
          and loaded["困倦"] == presets.normalize_mood(MOODS["困倦"]))
    check("load_moods：无 moods 段 → {}", presets.load_moods({}) == {})

    # ── 2. render_mood 结构 ───────────────────────────────
    block = presets.render_mood("千夜智子", "病弱", loaded["病弱"])
    check("render_mood：含标题·角色·退出线提醒",
          "当前状态：病弱" in block and "千夜智子此刻" in block
          and "退出线" in block and "只覆盖表达层" in block)
    check("render_mood：空状态 → 空串",
          presets.render_mood("千夜智子", "病弱", None) == "")

    # ── 3. persona 装载与 _mood_block ─────────────────────
    s = CharacterSpec(name="千夜智子", prompt_path=ZHIZI_MD,
                      prompt_text=load_prompt_text(ZHIZI_MD),
                      memory=True, search=True)
    p = Persona(make_cfg(tmp), llm=ScriptLLM([]), character=s,
                note_dir=os.path.join(tmp, "notes", "probe"))
    check("persona：moods 注册表已装载", set(p.moods) == {"病弱", "困倦"})
    blk, name = p._mood_block("病弱")
    check("_mood_block：已知状态 → 渲染段 + 规范化名", "病弱" in blk and name == "病弱")
    check("_mood_block：未知/空 → 无状态", p._mood_block("中二病") == ("", "")
          and p._mood_block("") == ("", ""))

    # ── 4. reply_structured 显式触发 + 注入位置 ────────────
    um = UserManager(make_cfg(tmp))
    z = um.entry("alice")["persona"]
    fake = inject(z, ScriptLLM([]))

    fake.replies = ["……没事，睡一觉就好。"]
    r = z.reply_structured("哥哥我回来了", mood="病弱")
    check("显式触发：mood 生效并回带", r["mood"] == "病弱", f"got={r['mood']}")
    sys0 = fake.calls[0][0]["content"]
    check("注入位置：卡面完整 + 状态段追加在 system 末尾",
          z.system_prompt in sys0 and "当前状态：病弱" in sys0
          and sys0.endswith("退出该状态。"))
    check("显式触发：mood 不进回复正文", "当前状态" not in r["raw"])

    recent = z.memory_manager.recent.get_messages("default")
    users = [m for m in recent if m["role"] == "user"]
    check("工作记忆：状态挂在提问上、题目原文干净",
          users[-1]["mood"] == "病弱" and users[-1]["content"] == "哥哥我回来了"
          and "当前状态" not in users[-1]["content"])
    check("工作记忆：assistant 不带状态",
          not any("mood" in m for m in recent if m["role"] == "assistant"))
    ctx = z.memory_manager.recent.get_context("default", max_tokens=2000)
    check("模型上下文：状态段不回灌（只活在声明的那轮 system）",
          "当前状态" not in ctx)
    h = z.get_history()
    check("get_history：状态单列字段、不进内容",
          h["persisted"] is True and h["messages"][-2]["mood"] == "病弱"
          and h["messages"][-2]["content"] == "哥哥我回来了")

    # ── 5. 状态随轮流动 + 未知降级 ────────────────────────
    fake.replies = ["凑合吧。"]
    r2 = z.reply_structured("今天心情如何")
    check("下一轮不声明：回基线、不残留状态",
          r2["mood"] == "" and "当前状态" not in fake.calls[-1][0]["content"])
    recent = z.memory_manager.recent.get_messages("default")
    users = [m for m in recent if m["role"] == "user"]
    check("基线轮：提问上不留 mood 键", "mood" not in users[-1])

    fake.replies = ["（低头）明天有雨。"]
    r3 = z.reply_structured("明天天气呢", mood="中二病")
    check("未知状态名：当无状态处理",
          r3["mood"] == "" and "当前状态" not in fake.calls[-1][0]["content"])

    # ── 6. regenerate 沿用上一轮状态 ───────────────────────
    fake.replies = ["（咳了两声）……帮你泡好了，热得刚好。"]
    r4 = z.reply_structured("帮我泡杯蜂蜜水", mood="病弱")
    check("新一轮显式触发：状态生效", r4["mood"] == "病弱")

    fake.replies = ["（努力打起精神）……蜂蜜水，喝吧。"]
    rr = z.regenerate_structured()
    check("regenerate：沿用上一轮状态",
          rr["mood"] == "病弱" and "当前状态：病弱" in fake.calls[-1][0]["content"])
    recent = z.memory_manager.recent.get_messages("default")
    users = [m for m in recent if m["role"] == "user"]
    check("regenerate：提问上的状态仍挂着",
          users[-1]["mood"] == "病弱" and users[-1]["content"] == "帮我泡杯蜂蜜水")
    check("regenerate：末条回复是新的干净正文",
          [m["content"] for m in recent if m["role"] == "assistant"][-1] == rr["raw"])

    # ── 7. HTTP：POST /chat 显式 mood 透传 ─────────────────
    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(um, "", dashboard_enabled=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        dave = um.entry("dave")["persona"]
        dave_fake = ScriptLLM(["（打了个哈欠）……你回来了。"])
        inject(dave, dave_fake)
        code, body = post(f"{base}/chat", {"message": "我回来了",
                                           "user_id": "dave", "mood": "困倦"})
        check("/chat：显式 mood 生效并回带",
              code == 200 and body.get("mood") == "困倦"
              and "当前状态" not in body.get("raw", ""),
              f"code={code} body={str(body)[:200]}")
    finally:
        srv.shutdown()
        srv.server_close()

    print()
    print(f"通过 {sum(PASS)}/{len(PASS)}")
    sys.exit(0 if all(PASS) else 1)


def post(url, payload, api_key=""):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


if __name__ == "__main__":
    main()