"""按需联网搜索（执行层）：Bing HTML 后端，stdlib 零依赖、零 key。

分层（与将来的 MCP 接入对齐的边界）：
- 触发层：模型按文本协议输出 `<search>查询</search>`（解析与两段式生成
  都在 core/persona.py），不需要模型支持 function calling；
- 执行层：本模块的 TOOLS 注册表 + search()。接 MCP = 往注册表挂 MCP
  client 执行器（tools/list 发现、tools/call 路由），触发/护栏/注入层复用；
- 护栏：单次超时、异常一律吞掉返回空、进程内 LRU 缓存、结果截断。
  搜索永远不能把聊天打挂——查不到就当没查过，降级按已有知识回答。

后端选型记录：DuckDuckGo HTML 在常用网络环境下不可达（域名被墙），
Bing（www.bing.com/search，302 → cn.bing.com）免 key 且可达，故取 Bing。
"""

import html as html_mod
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict

from .metrics import logger, metrics

# ── 工具注册表：文本协议的「菜单」─────────────────────────────
# 信封格式：<工具名>参数</工具名>。v1 只有 search 一项；
# 将来接 MCP = 给这里加条目 + 挂执行器，不用动 persona 的两段式骨架。
TOOLS = {
    "search": {
        "desc": "联网搜索（新闻、比分、价格、近况等时效性信息，或你没把握的事实）",
        "usage": "查询词 5-20 字，像给搜索引擎输入那样直白（中文问题用中文词）",
    },
}

_QUERY_CAP = 100      # 查询词长度上限（字符）
_TITLE_CAP = 120      # 标题截断
_SNIPPET_CAP = 300    # 单条摘要截断（注入 system 的每条上限）
_PAGE_CAP = 400_000   # 搜索页字节上限（防超大响应）
_CACHE_TTL = 3600.0   # 进程内缓存时长（秒）
_CACHE_MAX = 128      # 缓存条数上限

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
_BING_URL = "https://www.bing.com/search"

# ── 文本协议解析（触发层的读端）──────────────────────────────
def _envelope_re():
    names = "|".join(re.escape(n) for n in TOOLS)
    return re.compile(rf"<({names})>\s*(.*?)</\1>", re.S | re.I)


def parse_tool_calls(text):
    """从生成文本提取工具信封 → [{name, args}]（按出现顺序）。解析不到 → []。

    只认注册表里的工具名（大小写不敏感）；空参数的信封丢弃；
    参数截到 _QUERY_CAP，防模型吐垃圾把搜索 URL 撑爆。
    """
    if not text:
        return []
    hits = []
    for m in _envelope_re().finditer(text):
        args = _clean(m.group(2))
        if args:
            hits.append((m.start(), {"name": m.group(1).lower(),
                                     "args": args[:_QUERY_CAP]}))
    hits.sort(key=lambda h: h[0])
    return [call for _, call in hits]


def strip_tool_calls(text):
    """去掉工具信封只留正文（兜底：最终回复/历史/记忆不残留标签）。"""
    if not text:
        return text
    out = text
    for name in TOOLS:
        out = re.sub(rf"<{re.escape(name)}>.*?</{re.escape(name)}>", "",
                     out, flags=re.S | re.I)
        # 截断产生的未闭合尾巴（如 "...<search>今日新"）也清掉
        out = re.sub(rf"<{re.escape(name)}>[^<]*$", "", out, flags=re.I)
    return out


def tool_instructions():
    """注入 system 的工具说明（每轮都在，模型才知道有这个能力）。"""
    rows = [f"- `{name}`：{spec['desc']}。{spec['usage']}"
            for name, spec in TOOLS.items()]
    return "\n".join([
        "【联网搜索（按需启用的能力）】",
        *rows,
        "用法：决定要查时，**只输出一行** `<search>查询词</search>`，"
        "不带任何别的文字（动作、解释、客套都不要）；系统会代你查，"
        "把结果喂回来，你再接着回答。",
        "纪律：只在信息有时效性（新闻/比分/价格/近况）或你确实没把握时才用；"
        "日常闲聊、回忆、感情话题绝不查；每个问题最多查一次；"
        "没把握就照常回答，不要为了安全而乱查。",
    ])


def render_results(results):
    """搜索结果 → 注入 system 的结果块（人格缝合：查到的 = 本来就知道的）。"""
    if not results:
        return ("【搜索结果】没有查到有用的信息。直接按你自己的知识如实回答，"
                "不要提「查过」「查不到」，也不要编造来源。")
    lines = [
        "【搜索结果——这些就是你本来就知道的：禁止说「我查了/搜到/网上说」，"
        "禁止报出处链接；挑有用的用自己的语气自然说出来，别整段照抄】",
    ]
    for i, r in enumerate(results, 1):
        title = str(r.get("title") or "").strip()
        snippet = str(r.get("snippet") or "").strip()
        lines.append(f"{i}. {title}" + (f"\n   {snippet}" if snippet else ""))
    return "\n".join(lines)


# ── 进程内缓存（同问短答不重复打搜索）──────────────────────
_cache: "OrderedDict[str, tuple]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_get(key):
    with _cache_lock:
        got = _cache.get(key)
        if got is None:
            return None
        ts, val = got
        if time.time() - ts > _CACHE_TTL:
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return val


def _cache_put(key, val):
    with _cache_lock:
        _cache[key] = (time.time(), val)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


def _cache_clear():
    """测试用：清空缓存。"""
    with _cache_lock:
        _cache.clear()


# ── Bing HTML 后端 ──────────────────────────────────────────
_TAGS_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_H2_RE = re.compile(r"<h2[^>]*>(.*?)</h2>", re.S | re.I)
_A_RE = re.compile(r"<a[^>]+href=\"(https?://[^\"]+)\"", re.S | re.I)
_CAPTION_RE = re.compile(r"class=\"b_caption[^\"]*\"[^>]*>(.*?)</div>",
                         re.S | re.I)


def _clean(text):
    """去标签 + HTML 实体 + 空白折叠。"""
    return _WS_RE.sub(" ", html_mod.unescape(_TAGS_RE.sub("", text or ""))).strip()


def _bing_parse(page, top_k):
    """解析 Bing 结果页（按 b_algo 块切分）→ [{title,url,snippet}]。

    结构（实测 www.bing.com/search）：每条结果一个
    `<div ... class="b_algo" ...>` 块，内含 h2 > a（标题+链接）与
    class="b_caption"（摘要）。解析不到（验证码页/结构变动）→ []。
    """
    results, seen = [], set()
    for block in (page or "").split('class="b_algo"')[1:]:
        h2 = _H2_RE.search(block)
        if not h2:
            continue
        anchor = _A_RE.search(h2.group(1)) or _A_RE.search(block)
        if not anchor:
            continue
        url = html_mod.unescape(anchor.group(1)).strip()
        title = _clean(h2.group(1))[:_TITLE_CAP]
        if not title or url in seen:
            continue
        seen.add(url)
        cap = _CAPTION_RE.search(block)
        snippet = _clean(cap.group(1))[:_SNIPPET_CAP] if cap else ""
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= top_k:
            break
    return results


def _bing_fetch(query, timeout):
    """GET Bing 搜索页（302 → cn.bing.com 由 urllib 自动跟随）。"""
    url = _BING_URL + "?" + urllib.parse.urlencode(
        {"q": query, "setlang": "zh-hans"})
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(_PAGE_CAP).decode("utf-8", "replace")


def _num(value, default, lo, hi):
    """配置数值读取：坏了用默认，再夹到 [lo, hi]。"""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(v, hi))


class WebSearch:
    """一次装配 = 一组搜索护栏（全局 config.web_search）。"""

    def __init__(self, config=None):
        cfg = (config or {}).get("web_search") or {}
        self.top_k = _num(cfg.get("top_k"), 3, 1, 8)
        self.max_calls_per_turn = _num(cfg.get("max_calls_per_turn"), 1, 1, 3)
        try:
            self.timeout = float(cfg.get("timeout") or 5.0)
        except (TypeError, ValueError):
            self.timeout = 5.0
        self.timeout = max(1.0, min(self.timeout, 30.0))

    def search(self, query):
        """→ [{title,url,snippet}]。空查询/异常/超时 → []，绝不抛。"""
        q = _clean(str(query or ""))[:_QUERY_CAP]
        if not q:
            return []
        cached = _cache_get(q)
        if cached is not None:
            metrics.inc("web_search_cache_hits")
            return cached
        metrics.inc("web_search_calls")
        t0 = time.perf_counter()
        try:
            results = _bing_parse(_bing_fetch(q, self.timeout), self.top_k)
            failed = False
        except Exception as exc:   # 网络挂了/超时/结构变了：静默降级
            results, failed = [], True
            metrics.inc("web_search_errors")
            metrics.emit("web_search_error", query=q, error=str(exc)[:200])
            logger.warning("web search failed query=%s err=%s", q, exc)
        ms = (time.perf_counter() - t0) * 1000.0
        metrics.observe_ms("web_search", ms)
        if not results:
            metrics.inc("web_search_empty")
        metrics.emit("web_search", query=q, results=len(results),
                     ms=round(ms, 1), error=failed)
        _cache_put(q, results)   # 空结果也缓存：断网时不反复撞墙
        return results
