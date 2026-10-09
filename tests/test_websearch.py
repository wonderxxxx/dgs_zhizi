#!/usr/bin/env python3
"""按需联网搜索：文本协议触发 + 两段式生成 + Bing 执行层。

背景：默认 provider 是 mlx 本地推理，没有 function calling，所以走文本协议
——模型输出 <search>查询</search>，系统代查（Bing HTML 免 key）并把结果
喂回 system 重建第二遍。工具信封只活在两遍之间：最终回复、历史、观测笔记
永远看不到标签（人物卡红线：不承认自己是 AI，搜索过程对她/对用户都不可见）。

覆盖：信封解析/剥离 / 工具说明与结果块渲染 / Bing 页面解析与缓存降级 /
      两段式（有标签两遍、无标签一遍、额度、顽固模型第三遍兜底）/
      allow_search 开关 / 角色开关（白鸥无此能力）/ 工具文本不进历史与笔记 /
      regenerate 路径 / POST /chat·GET /health·GET /characters。
运行方式：cd zhizi && .venv/bin/python tests/test_websearch.py
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

from core import websearch as W  # noqa: E402
from core.characters import CharacterSpec, load_characters, load_prompt_text  # noqa: E402
from core.persona import Persona  # noqa: E402
from core.users import UserManager  # noqa: E402
from api import make_handler  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZHIZI_MD = os.path.join(BASE, "prompts", "zhizi_v1.md")
BAIOU_TXT = os.path.join(BASE, "prompts", "白鸥人物卡-万物皆数定制版.txt")
IDENTITY = os.path.join(BASE, "visual", "identity.json")

PASS = []

# 合成 Bing 结果页（b_algo 块结构与实测 www.bing.com/search 一致）
FIXTURE = """
<html><body><div id="b_results">
<div class="b_algo"><div data-id="a"><h2><a href="https://example.com/a" h="ID=SERP">示例<u>标题</u>A</a></h2>
<div class="b_caption"><div>这是<u>摘要</u> A，含 &amp; 实体。</div></div></div></div>
<div class="b_algo"><div><h2><a href="https://example.com/b">标题B</a></h2>
<div class="b_caption"><div>摘要 B。</div></div></div></div>
<div class="b_algo"><div><h2><a href="https://example.com/a">重复链接</a></h2>
<div class="b_caption"><div>不该出现。</div></div></div></div>
</div></body></html>
"""


def check(name, cond, detail=""):
    status = "✅" if cond else "❌"
    PASS.append(bool(cond))
    print(f"{status} {name}" + (f"  ({detail})" if not cond else ""))


class ScriptLLM:
    """按序回脚本句子；单条消息（事实提取/反思这类工具调用）恒回 []。"""

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


def make_cfg(tmp, **web_search):
    cfg = {
        "characters": {
            "千夜智子": {"prompt": ZHIZI_MD, "visual_identity": IDENTITY,
                         "memory": True, "default": True},
            "白鸥": {"prompt": BAIOU_TXT, "visual_identity": "",
                     "memory": False, "search": False},
        },
        "memory": {"note_file": os.path.join(tmp, "notes"), "top_k": 5},
        "attachments": {"caption": True},
        "provider": {"type": "local", "base_url": "http://x", "model": "fake",
                     "api_key": "x", "temperature": 0.85},
        "web_search": {"enabled": True, "top_k": 3,
                       "max_calls_per_turn": 1, "timeout": 5.0,
                       **web_search},
    }
    return cfg


def bare_persona(tmp, replies, **web_search):
    """无记忆角色 + 脚本 LLM + 桩搜索（不碰网络）的最小两段式试验台。"""
    cfg = make_cfg(tmp, **web_search)
    s = CharacterSpec(name="小网", prompt_path=BAIOU_TXT,
                      prompt_text=load_prompt_text(BAIOU_TXT), memory=False,
                      search=True)
    p = Persona(cfg, llm=ScriptLLM(replies), character=s,
                note_dir=os.path.join(tmp, "notes", "bare"))
    p.websearch_hits = []
    p.websearch.search = lambda q: p.websearch_hits.append(q) or [
        {"title": "头条A", "url": "https://news.a", "snippet": "摘要A"},
        {"title": "头条B", "url": "https://news.b", "snippet": "摘要B"}]
    return p


def main():
    # ── 1. 信封解析 / 剥离 ─────────────────────────────────────
    calls = W.parse_tool_calls("（歪头）<search>今日新闻</search>好")
    check("解析：夹在正文里的信封", calls == [{"name": "search",
          "args": "今日新闻"}], f"got={calls}")
    check("解析：无标签 → []", W.parse_tool_calls("今天天气不错") == [])
    check("解析：大小写不敏感", W.parse_tool_calls("<SEARCH>天气</SEARCH>")
          and W.parse_tool_calls("<SEARCH>天气</SEARCH>")[0]["args"] == "天气")
    check("解析：空参数丢弃", W.parse_tool_calls("<search> </search>") == [])
    check("解析：注册表外的工具名不认", W.parse_tool_calls("<foo>x</foo>") == [])
    multi = W.parse_tool_calls("<search>甲</search>垫话<search>乙</search>")
    check("解析：多信封按出现顺序", [c["args"] for c in multi] == ["甲", "乙"],
          f"got={multi}")
    check("剥离：标签去掉、正文保留",
          W.strip_tool_calls("（歪头）<search>今日新闻</search>好")
          == "（歪头）好")
    check("剥离：无标签原样返回",
          W.strip_tool_calls("今天天气不错") == "今天天气不错")
    check("剥离：截断产生的未闭合尾巴也清掉",
          W.strip_tool_calls("前面<search>今日新") == "前面")

    # ── 2. 工具说明 / 结果块渲染 ──────────────────────────────
    instr = W.tool_instructions()
    check("说明：含信封用法", "<search>查询词</search>" in instr)
    check("说明：含纪律（最多查一次/闲聊不查）",
          "最多查一次" in instr and "闲聊" in instr)
    empty_block = W.render_results([])
    check("空结果：按没查到处理且要求别提「查过」",
          "没有查到" in empty_block and "不要提" in empty_block)
    block = W.render_results([{"title": "T1", "url": "https://u1", "snippet": "S1"},
                              {"title": "T2", "url": "https://u2", "snippet": "S2"}])
    check("结果块：编号列出", "1. T1" in block and "2. T2" in block)
    check("结果块：要求别提查过、别报出处",
          "禁止说「我查了" in block and "禁止报出处" in block)

    # ── 3. 配置读取 ───────────────────────────────────────────
    ws = W.WebSearch({"web_search": {"top_k": 99, "max_calls_per_turn": 0,
                                     "timeout": "坏值"}})
    check("配置：top_k 夹到上限 8", ws.top_k == 8, f"got={ws.top_k}")
    check("配置：每轮次数夹到下限 1", ws.max_calls_per_turn == 1)
    check("配置：timeout 坏值回退 5.0", ws.timeout == 5.0, f"got={ws.timeout}")
    d = W.WebSearch(None)
    check("配置：整段缺失用默认", (d.top_k, d.max_calls_per_turn, d.timeout)
          == (3, 1, 5.0))

    # ── 4. Bing 页面解析（fixture，不碰网络）─────────────────
    got = W._bing_parse(FIXTURE, top_k=5)
    check("解析页：标题去标签去实体",
          got and got[0]["title"] == "示例标题A", f"got={got[:1]}")
    check("解析页：摘要去标签去实体",
          got and got[0]["snippet"] == "这是摘要 A，含 & 实体。",
          f"got={got[:1]}")
    check("解析页：链接抽出", got and got[0]["url"] == "https://example.com/a")
    check("解析页：重复链接去重", len(got) == 2, f"got={len(got)}")
    check("解析页：top_k 生效", len(W._bing_parse(FIXTURE, top_k=1)) == 1)
    check("解析页：验证码页/空页面 → []",
          W._bing_parse("<html>captcha</html>", 3) == [])

    # ── 5. search()：缓存 / 异常降级（桩掉网络）───────────────
    W._cache_clear()
    hits = {"n": 0}
    orig_fetch = W._bing_fetch
    W._bing_fetch = lambda q, t: hits.__setitem__("n", hits["n"] + 1) or FIXTURE
    try:
        ws2 = W.WebSearch({})
        r1 = ws2.search("测试查询词")
        r2 = ws2.search("测试查询词")
        check("search：桩后端返回解析结果", len(r1) == 2, f"got={r1}")
        check("search：同问走缓存只打一次后端",
              hits["n"] == 1 and r2 == r1, f"fetch={hits['n']}")

        def boom(q, t):
            raise OSError("网络炸了")
        W._bing_fetch = boom
        W._cache_clear()
        ws3 = W.WebSearch({})
        check("search：后端异常 → [] 不抛", ws3.search("任意查询") == [])
        check("search：空查询不碰后端", ws3.search("   ") == [])
    finally:
        W._bing_fetch = orig_fetch
        W._cache_clear()

    # ── 6. 两段式：有标签才第二遍，工具文本不外泄 ──────────────
    tmp = tempfile.mkdtemp(prefix="zhizi_wsearch_")

    p = bare_persona(tmp, ["<search>今日新闻</search>",
                           "（眨眼）今天的新闻是……"])
    res = p.reply_structured("今天有什么新闻？")
    check("两段式：回复是第二遍的正文",
          res["reply"] == "今天的新闻是……", f"got={res['reply']}")
    check("两段式：searched=True", res["searched"] is True)
    check("两段式：恰好两次模型调用", len(p.llm.calls) == 2,
          f"got={len(p.llm.calls)}")
    sys1, sys2 = p.llm.calls[0][0]["content"], p.llm.calls[1][0]["content"]
    check("第一遍：system 带工具说明", "联网搜索（按需启用的能力）" in sys1)
    check("第一遍：还没有结果块", "【搜索结果" not in sys1)
    check("第二遍：结果块进了 system", "头条A" in sys2 and "摘要B" in sys2)
    check("第二遍：仍带工具说明", "联网搜索（按需启用的能力）" in sys2)
    check("执行器：收到模型给的查询词", p.websearch_hits == ["今日新闻"],
          f"got={p.websearch_hits}")
    hist = p.get_history()["messages"]
    check("历史：只有干净正文（无信封、无结果块）",
          len(hist) == 2 and "<search>" not in str(hist)
          and "【搜索结果" not in str(hist), f"got={hist}")

    # ── 7. 无标签 → 一遍，零额外开销 ──────────────────────────
    p2 = bare_persona(tmp, ["（挑眉）今晚吃面。"])
    res2 = p2.reply_structured("晚上吃什么")
    check("无标签：一次调用直接回答", len(p2.llm.calls) == 1)
    check("无标签：searched=False", res2["searched"] is False)
    check("无标签：没碰执行器", p2.websearch_hits == [])

    # ── 8. allow_search=false：关掉触发（壳子/请求级开关）──────
    p3 = bare_persona(tmp, ["查一下<search>今日新闻</search>",
                            "不该被调用"])
    res3 = p3.reply_structured("今天有新闻吗", allow_search=False)
    check("关搜索：只调一次模型", len(p3.llm.calls) == 1)
    check("关搜索：没碰执行器", p3.websearch_hits == [])
    check("关搜索：searched=False", res3["searched"] is False)
    check("关搜索：标签仍被剥掉不进回复",
          "<search>" not in res3["reply"], f"got={res3['reply']}")

    # ── 9. 额度：一轮最多 1 次（模型吐两个信封也只认第一个）────
    p4 = bare_persona(tmp, ["<search>甲查询</search><search>乙查询</search>",
                            "只查了一次的答复"])
    p4.reply_structured("查两下")
    check("额度：每轮只执行 1 次搜索", p4.websearch_hits == ["甲查询"],
          f"got={p4.websearch_hits}")

    # ── 10. 顽固模型：第二遍还吐信封 → 撤工具说明要第三遍 ──────
    p5 = bare_persona(tmp, ["<search>甲</search>", "<search>甲</search>",
                            "最终还是答出来了"])
    res5 = p5.reply_structured("顽固测试")
    check("兜底：第三遍拿到正文", res5["reply"] == "最终还是答出来了",
          f"got={res5['reply']} calls={len(p5.llm.calls)}")
    check("兜底：封顶 3 次模型调用", len(p5.llm.calls) == 3,
          f"got={len(p5.llm.calls)}")
    sys3 = p5.llm.calls[2][0]["content"]
    check("兜底：第三遍撤掉工具说明", "联网搜索（按需启用的能力）" not in sys3)
    check("兜底：第三遍仍有结果块", "头条A" in sys3)
    check("兜底：仍然只搜了一次", p5.websearch_hits == ["甲"],
          f"got={p5.websearch_hits}")

    # 第二遍「标签+正文」混出：取正文、不再搜
    p5b = bare_persona(tmp, ["<search>甲</search>",
                             "查过了<search>甲</search>所以结论是……"])
    res5b = p5b.reply_structured("混合测试")
    check("混合：取剥干净的正文、不再补搜",
          res5b["reply"] == "查过了所以结论是……" and p5b.websearch_hits == ["甲"]
          and len(p5b.llm.calls) == 2, f"got={res5b['reply']}")

    # ── 11. 搜索没查到 → 按没查到降级，不报错 ──────────────────
    p6 = bare_persona(tmp, ["<search>查无此词</search>", "凭记忆答。"])
    p6.websearch.search = lambda q: p6.websearch_hits.append(q) or []
    res6 = p6.reply_structured("查不到怎么办")
    check("空结果：仍正常生成第二遍", res6["reply"] == "凭记忆答。"
          and res6["searched"] is True)
    check("空结果：第二遍 system 是「没查到」块",
          "没有查到有用的信息" in p6.llm.calls[1][0]["content"])

    # ── 12. 角色开关：白鸥没有这个能力 ────────────────────────
    cfgw = make_cfg(tmp)
    chars = load_characters(cfgw)
    check("角色开关：config 里白鸥 search=false",
          chars["白鸥"].search is False and chars["千夜智子"].search is True)
    b = Persona(cfgw, llm=ScriptLLM(["嗯。"]), character=chars["白鸥"],
                note_dir=os.path.join(tmp, "notes", "b"))
    check("角色开关：白鸥不装配执行器", b.websearch is None)
    check("角色开关：白鸥 system 无工具说明",
          "联网搜索（按需启用的能力）" not in b.build_messages("hi")[0]["content"])
    resb = b.reply_structured("你好")
    check("角色开关：白鸥单遍直答", len(b.llm.calls) == 1
          and resb["searched"] is False)

    # 全局总开关
    cfg_off = make_cfg(tmp, enabled=False)
    chars_off = load_characters(cfg_off)
    p_off = Persona(cfg_off, llm=ScriptLLM(["嗯。"]),
                    character=chars_off["千夜智子"],
                    note_dir=os.path.join(tmp, "notes", "off"))
    check("总开关：web_search.enabled=false 不装配", p_off.websearch is None)

    # ── 13. 有记忆角色：工具文本不进工作记忆/笔记 ──────────────
    um = UserManager(make_cfg(tmp))
    z = um.entry("alice")["persona"]
    inject(z, ScriptLLM(["<search>量子计算进展</search>",
                         "（推眼镜）进展如下。"]))
    z.websearch_hits = []
    z.websearch.search = lambda q: z.websearch_hits.append(q) or [
        {"title": "进展X", "url": "https://x", "snippet": "论文X"}]
    r = z.reply_structured("量子计算有啥新进展")
    check("笔记角色：两段式正常",
          r["searched"] is True and r["reply"] == "进展如下。"
          and r["raw"] == "（推眼镜）进展如下。",
          f"got={r['reply']} raw={r['raw']}")
    recent = z.memory_manager.recent.get_messages("default")
    joined = json.dumps(recent, ensure_ascii=False)
    check("笔记角色：工作记忆无信封", "<search>" not in joined)
    check("笔记角色：工作记忆无结果块", "【搜索结果" not in joined)
    check("笔记角色：assistant 落的是最终正文",
          [m["content"] for m in recent if m["role"] == "assistant"]
          == ["（推眼镜）进展如下。"], f"got={recent}")
    idx = z.memory_manager.timeindex.get_by_category("default", "conversation")
    check("笔记角色：时间索引也干净",
          all("<search>" not in d["content"] and "【搜索结果" not in d["content"]
              for d in idx), f"got={[d['content'] for d in idx]}")

    # ── 14. regenerate 同样走两段式 ───────────────────────────
    z.reply_structured("普通的一问")                       # 先攒一轮普通问答（无标签）
    before = len(z.llm.calls)
    z.websearch_hits.clear()
    z.llm.replies = ["<search>再查一次</search>", "重新生成的答复"]
    rr = z.regenerate_structured()
    check("重新生成：也触发了两段式",
          rr["searched"] is True and len(z.llm.calls) == before + 2,
          f"got={rr['searched']} calls={len(z.llm.calls) - before}")
    check("重新生成：查询词是新一轮的", z.websearch_hits == ["再查一次"],
          f"got={z.websearch_hits}")
    check("重新生成：历史仍是一问一答、末条干净",
          "<search>" not in str(z.get_history()["messages"][-1])
          and "【搜索结果" not in str(z.get_history()["messages"][-1]))
    zr = z.memory_manager.recent.get_messages("default")
    check("重新生成：笔记里的回复是新的干净正文",
          [m["content"] for m in zr if m["role"] == "assistant"][-1]
          == rr["raw"], f"got={[m['content'] for m in zr if m['role'] == 'assistant']}")

    # ── 15. HTTP：allow_search 透传 + /health + /characters ───
    dave = um.entry("dave")["persona"]
    inject(dave, ScriptLLM(["<search>英伟达股价</search>", "（眯眼）涨了。"]))
    dave.websearch_hits = []
    dave.websearch.search = lambda q: dave.websearch_hits.append(q) or [
        {"title": "股价", "url": "https://s", "snippet": "详情"}]

    srv = ThreadingHTTPServer(("127.0.0.1", 0),
                              make_handler(um, "", dashboard_enabled=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        code, body = post(f"{base}/chat",
                          {"message": "英伟达股价怎么样", "user_id": "dave"})
        check("/chat：默认允许搜索 → searched=true + 干净回复",
              code == 200 and body.get("searched") is True
              and body.get("reply") == "涨了。" and body.get("actions") == ["眯眼"]
              and "<search>" not in body.get("raw", "")
              and "【搜索结果" not in body.get("raw", ""),
              f"code={code} body={str(body)[:200]}")

        code, body = post(f"{base}/chat",
                          {"message": "随便聊聊", "user_id": "dave",
                           "allow_search": False})
        check("/chat：allow_search=false → searched=false",
              code == 200 and body.get("searched") is False,
              f"code={code} body={str(body)[:160]}")
        check("/chat：关掉后没再碰执行器",
              dave.websearch_hits == ["英伟达股价"], f"got={dave.websearch_hits}")

        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            health = json.loads(r.read().decode("utf-8"))
        check("/health：暴露 search 能力位", health.get("search") is True,
              f"got={health}")

        with urllib.request.urlopen(f"{base}/characters", timeout=5) as r:
            chars_body = json.loads(r.read().decode("utf-8"))
        by_name = {c["name"]: c for c in chars_body["characters"]}
        check("/characters：角色带 search 开关",
              by_name.get("千夜智子", {}).get("search") is True
              and by_name.get("白鸥", {}).get("search") is False,
              f"got={list(by_name)}")
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
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


if __name__ == "__main__":
    main()
