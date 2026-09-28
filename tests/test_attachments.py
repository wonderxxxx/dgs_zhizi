#!/usr/bin/env python3
"""图片附件验证：看得见（原图落盘回读）+ 记得住（内容说明进记忆）。

覆盖：base64 → 落盘 → 引用回读 / 目录穿越与越权取图一律拒绝 /
      图片内容说明进五维记忆（工作记忆单列字段、近期记忆可检索、上下文带说明）/
      有身份卡的角色复用自识别那次描述（零额外视觉调用）/ 非多模态降级 /
      /history 带 images+caption、/image 出图、清记忆连带清图。
运行方式：cd zhizi && .venv/bin/python tests/test_attachments.py
"""

import base64
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

if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台默认 GBK，✅ 会崩
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core import attachments  # noqa: E402
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
    cond = bool(cond)
    status = "✅" if cond else "❌"
    PASS.append(cond)
    print(f"{status} {name}" + (f"  ({detail})" if detail and not cond else ""))


class FakeVLM:
    """fake VLM：按 system 提示分四种回法——自识别描述 / 外观比对 / 图片说明 / 对话。"""

    model = "fake"
    multimodal = True

    def __init__(self):
        self.calls = 0
        self.describe_calls = 0
        self.compare_calls = 0
        self.caption_calls = 0
        self.chat_calls = 0

    def chat(self, messages, temperature=None, images=None, enable_thinking=None):
        self.calls += 1
        system = messages[0].get("content", "") if messages else ""
        if "图像描述助手" in system and "只描述客观可见内容" in system:
            self.caption_calls += 1
            return "一个女孩穿着白裙站在樱花树下"
        if images and "外观比对助手" in system:
            self.compare_calls += 1
            return json.dumps({"same_character": 0.5, "reason": "都是黑长发"},
                              ensure_ascii=False)
        if images and "图像描述助手" in system:
            self.describe_calls += 1
            return json.dumps({
                "hair": "黑色长发", "eyes": "琥珀色", "style": "anime",
                "appearance_age": "young adult", "clothing": "白色连衣裙",
                "summary": "一个女孩穿着白裙站在樱花树下",
            }, ensure_ascii=False)
        self.chat_calls += 1
        return "（歪头）好漂亮的樱花。"


class TextOnlyLLM:
    """只文本模型：图不该被送进模型，也不该崩。"""

    model = "fake"
    multimodal = False

    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature=None, images=None, enable_thinking=None):
        self.calls += 1
        if images:
            raise AssertionError("只文本模型不该收到图片")
        return "（点头）嗯。"


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


def img():
    return [{"mime": "image/png", "data": base64.b64encode(PNG).decode("ascii")}]


def main():
    tmp = tempfile.mkdtemp(prefix="zhizi_att_")
    note_root = os.path.join(tmp, "notes", "alice")

    # ── 1. 落盘 + 回读 ──────────────────────────────────────
    refs = attachments.save_images(note_root, img())
    check("落盘：拿到 1 个引用", len(refs) == 1, f"got={refs}")
    name = refs[0]["name"]
    check("落盘：文件在 images/ 下",
          os.path.isfile(os.path.join(note_root, "images", name)))
    check("落盘：mime 认得出", refs[0]["mime"] == "image/png")
    blob = attachments.read_image(note_root, name)
    check("回读：字节一致", blob is not None and blob[0] == PNG and blob[1] == "image/png")
    check("落盘：坏 base64 跳过不崩", attachments.save_images(note_root, [{"mime": "image/png", "data": "!!!"}]) == [])
    big = base64.b64encode(b"\x00" * (13 * 1024 * 1024)).decode("ascii")
    check("落盘：超 12MB 的图跳过（别写爆磁盘）",
          attachments.save_images(note_root, [{"mime": "image/png", "data": big}]) == [])

    # ── 2. 目录穿越 / 越权 ───────────────────────────────────
    check("安全：路径穿越被拒", attachments.resolve(note_root, "../../default/memory/facts/facts.db") is None)
    check("安全：分隔符被拒", attachments.resolve(note_root, "images/x.png") is None)
    check("安全：不存在 → None", attachments.resolve(note_root, "nope.png") is None)
    check("安全：空目录名 → None", attachments.resolve("", name) is None)

    # ── 3. 内容说明：复用自识别那次描述（不额外调模型） ────────
    fake = FakeVLM()
    cap = attachments.caption(fake, img(), described={
        "hair": "黑色长发", "summary": "一个女孩穿着白裙站在樱花树下",
    })
    check("说明：拼自识别描述（零额外调用）",
          "樱花树下" in cap and fake.calls == 0, f"cap={cap}")
    cap2 = attachments.caption(fake, img(), described=None)
    check("说明：无描述时单独调一次 VLM",
          "樱花树下" in cap2 and fake.caption_calls == 1, f"cap={cap2}")
    check("说明：非多模态模型降级为空", attachments.caption(TextOnlyLLM(), img()) == "")
    check("说明：说明失败不抛异常", attachments.caption(_BoomLLM(), img()) == "")
    check("说明：模型回 JSON 时只取 summary",
          attachments._plain_sentence('{"summary": "樱花树下的女孩"}') == "樱花树下的女孩")
    check("说明：代码块围栏 / 多行收敛成一句",
          attachments._plain_sentence("```\n一个女孩\n第二行\n```") == "一个女孩")

    # ── 4. 全链路：带图一轮 → 落盘 + 说明进记忆 ───────────────
    cfg = make_cfg(tmp)
    um = UserManager(cfg)
    fresh = FakeVLM()          # 干净计数器，用来验「复用自识别、不额外调 VLM」
    z = um.entry("alice")["persona"]
    z.llm = fresh
    z.visual_identity.llm = fresh   # 自识别层持的是构造时那个 client，一并换掉
    res = z.reply_structured("你看这张", images=img())
    att = res.get("attachments") or {}
    check("全链路：/chat 回传 attachments 引用", len(att.get("images") or []) == 1, f"got={att}")
    check("全链路：/chat 回传 caption", "樱花树下" in (att.get("caption") or ""), f"got={att}")
    check("全链路：复用自识别那次描述，未额外调 VLM",
          fresh.describe_calls == 1 and fresh.caption_calls == 0,
          f"describe={fresh.describe_calls} caption={fresh.caption_calls}")

    hist = z.get_history()["messages"]
    user_msg = hist[0]
    check("历史：带 images 引用", user_msg.get("images", [{}])[0].get("name", "").endswith(".png"))
    check("历史：原文保持干净（没混进说明）", user_msg["content"] == "你看这张")
    check("历史：caption 单列一个字段", "樱花树下" in user_msg.get("caption", ""))

    mm = z.memory_manager
    recent = mm.recent.get_messages("default", limit=5)
    check("记忆·工作记忆：原文字段干净", recent[0]["content"] == "你看这张")
    check("记忆·工作记忆：说明单列", "樱花树下" in recent[0].get("caption", ""))
    check("记忆·近期记忆：带说明可检索",
          any("樱花树下" in i["content"] for i in mm.timeindex.all("default", limit=20)))
    ctx, _ = mm.get_context_traced(user_id="default", query="樱花")
    check("记忆·注入上下文：带图片说明", "樱花树下" in ctx, f"ctx={ctx[:120]}")
    trace_items = mm.recall_trace(user_id="default", query="樱花")["dimensions"][0]["items"]
    check("记忆·召回 trace：看得见图与说明",
          bool(trace_items[0].get("caption")) and bool(trace_items[0].get("images")),
          f"got={trace_items[0]}")

    # ── 5. 跨进程重启：图与说明都还在 ─────────────────────────
    um2 = UserManager(make_cfg(tmp))
    h2 = um2.get("alice").get_history()["messages"]
    check("重启：历史仍带 images + caption",
          h2[0].get("images") and "樱花树下" in h2[0].get("caption", ""))
    check("重启：原图仍能读回", attachments.read_image(um2.get("alice").note_dir,
                                            h2[0]["images"][0]["name"]) is not None)

    # ── 6. 无记忆角色（白鸥）：图落盘、说明进上下文窗口 ────────
    b = um2.entry("alice", "白鸥")["persona"]
    b.llm = FakeVLM()
    b.reply_structured("现在呢", images=img())
    hb = b.get_history()
    check("无记忆角色：进程内历史带图", bool(hb["messages"][0].get("images")))
    built = b.build_messages("下一句", self_context="")
    check("无记忆角色：上下文带图片说明",
          any("樱花树下" in m["content"] for m in built),
          f"built={[m['content'][:40] for m in built]}")
    check("无记忆角色：仍标 persisted=False", hb["persisted"] is False)

    # ── 7. HTTP：/history 带图、/image 出图、越权取不到 ───────
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(um2, "", False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/history?user_id=alice", timeout=5) as r:
            body = json.loads(r.read().decode("utf-8"))
        m0 = body["messages"][0]
        check("HTTP /history：带 images + caption",
              m0.get("images") and "樱花树下" in m0.get("caption", ""), f"got={m0}")
        name = m0["images"][0]["name"]
        with urllib.request.urlopen(
                f"{base}/image?user_id=alice&name={urllib.parse.quote(name)}", timeout=5) as r:
            raw = r.read()
        check("HTTP /image：回原图字节与 mime",
              raw == PNG and r.headers["Content-Type"] == "image/png",
              f"len={len(raw)} ct={r.headers.get('Content-Type')}")
        for evil in ("../../../etc/passwd", "x/y.png"):
            try:
                urllib.request.urlopen(
                    f"{base}/image?user_id=alice&name={urllib.parse.quote(evil)}", timeout=5)
                check(f"HTTP /image：越权 {evil} 被拒", False, "居然放行了")
            except urllib.error.HTTPError as e:
                check(f"HTTP /image：越权 {evil} → 404", e.code == 404, f"code={e.code}")
    finally:
        srv.shutdown()
        srv.server_close()

    # ── 8. 清记忆连带清图（不留孤儿文件） ─────────────────────
    z2 = um2.get("alice")
    cleared = z2.clear_memory()
    check("清记忆：图片一并清掉",
          cleared.get("images", 0) >= 1
          and not os.listdir(attachments.image_dir(z2.note_dir)),
          f"cleared={cleared} left={os.listdir(attachments.image_dir(z2.note_dir))}")

    print()
    print(f"通过 {sum(PASS)}/{len(PASS)}")
    sys.exit(0 if all(PASS) else 1)


class _BoomLLM:
    model = "fake"
    multimodal = True

    def chat(self, *a, **kw):
        raise RuntimeError("视觉调用炸了")


if __name__ == "__main__":
    main()
