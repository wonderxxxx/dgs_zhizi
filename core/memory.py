"""记忆层：「观测笔记」——智子用科研口吻记录值得记住的事。

v2（P0 升级）：SQLite 持久化 + 自研 BM25 倒排检索 + importance + hash 去重。
- 数据表 notes(id, time, text, context, importance, hash, updated_at)，time 建索引；
- 倒排索引启动时全量加载，增删改同步维护；
- 检索：中文 bigram（单字回退）+ 英文/数字整词 → BM25 打分（词频/IDF/长度归一）
  → importance 微调 → top_k；
- 去重：文本规范化后 SHA-256；相同笔记只保留一条（更新最近时间，重要度取较大值）；
- 兼容：NoteStore(note_dir) 接口不变；旧 JSONL 首次启动自动迁移（原文件改名保留）。

零新依赖（标准库 sqlite3）。
"""

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from datetime import datetime

# BM25 参数（标准取值）
_K1 = 1.2
_B = 0.75
# importance 对排名的微调系数：importance 10 相对 5 约 +1.5 分（BM25 分通常 0~5）
_IMPORTANCE_WEIGHT = 0.3
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE = re.compile(r"[A-Za-z0-9_]+")
_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[。！？!?．.、，,；;：:]+$")


def _tokenize(text, include_unigram=True):
    """分词：英文/数字整词（小写）+ 中文 bigram（可选 unigram）。

    - 索引侧 include_unigram=True：保留单字 token，使「哥」这类单字查询可命中，
      常见虚字的噪声由 BM25 的 IDF 与查询侧 bigram 优先策略共同压制；
    - 查询侧由 search() 决定：查询含中文 bigram 时只用 bigram（精确），
      查询整体为单字时才回退 unigram（召回）。
    """
    out = set()
    for part in _ASCII_RE.findall(text.lower()):
        out.add(part)
    for part in _CJK_RE.findall(text):
        if len(part) == 1:
            if include_unigram:
                out.add(part)
        else:
            for i in range(len(part) - 1):
                out.add(part[i : i + 2])  # bigram
            if include_unigram:
                for i in range(len(part)):
                    out.add(part[i])  # unigram
    return out


def _normalize(text):
    """规范化文本用于去重 hash：去首尾空白、压缩空白、去尾部标点。"""
    return _TRAILING_PUNCT_RE.sub("", _WS_RE.sub(" ", text.strip()))


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _query_tokens(query):
    """查询侧分词：含中文 bigram 时只用 bigram（精确匹配）；整体为单字时才回退 unigram。"""
    parts = _tokenize(query, include_unigram=True)
    has_cjk_bigram = any(len(t) == 2 and not t.isascii() for t in parts)
    if has_cjk_bigram:
        return _tokenize(query, include_unigram=False)
    return parts


class NoteStore:
    """观测笔记存储：SQLite 落盘 + 内存倒排索引（BM25 检索）。"""

    def __init__(self, note_dir):
        self.note_dir = note_dir
        os.makedirs(note_dir, exist_ok=True)
        self._db_path = os.path.join(note_dir, "notes.db")
        self._lock = threading.RLock()  # 防御性：api 层已按用户串行，这里再兜一层
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._migrate_jsonl()
        # 内存检索结构
        self._docs = {}      # id -> dict(text, context, importance, time)
        self._postings = {}  # term -> {id: tf}
        self._doc_len = {}   # id -> token 数
        self._load_index()

    # ── 存储层 ──────────────────────────────────────────────────────
    def _init_schema(self):
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    time        TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    context     TEXT NOT NULL DEFAULT '',
                    importance  INTEGER NOT NULL DEFAULT 5,
                    hash        TEXT NOT NULL UNIQUE,
                    updated_at  TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_time ON notes(time)"
            )

    def _migrate_jsonl(self):
        """首次启动：把旧版 notes.jsonl 导入 SQLite（幂等，仅当库为空且文件存在）。"""
        jsonl_path = os.path.join(self.note_dir, "notes.jsonl")
        if not os.path.exists(jsonl_path):
            return
        count = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        if count:
            return  # 库已有数据，跳过（避免重复导入）
        imported = 0
        with self._conn:  # 迁移事务：成功提交，异常回滚
            with open(jsonl_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(item.get("text", "")).strip()
                    if not text:
                        continue
                    self._insert_row(
                        time=str(item.get("time", "") or ""),
                        text=text,
                        context=str(item.get("context", "") or "")[:200],
                        importance=int(item.get("importance", 5) or 5),
                    )
                    imported += 1
        if imported:
            os.replace(jsonl_path, jsonl_path + ".migrated")  # 保留原数据，不再重复读

    def _insert_row(self, time, text, context, importance):
        now = datetime.now().isoformat(timespec="seconds")
        digest = _sha256(_normalize(text))
        cur = self._conn.execute(
            "INSERT INTO notes(time, text, context, importance, hash, updated_at)"
            " VALUES(?, ?, ?, ?, ?, ?)",
            (time or now, text, context, importance, digest, now),
        )
        return cur.lastrowid

    def _load_index(self):
        rows = self._conn.execute(
            "SELECT id, time, text, context, importance FROM notes"
        ).fetchall()
        for row in rows:
            self._index_doc(row[0], row[2], row[3], row[4], row[1])

    def _index_doc(self, doc_id, text, context, importance, time=""):
        tokens = _tokenize(text + " " + context)
        self._docs[doc_id] = {
            "id": doc_id,
            "text": text,
            "context": context,
            "importance": int(importance),
            "time": time,
        }
        self._doc_len[doc_id] = len(tokens)
        for term in tokens:
            postings = self._postings.setdefault(term, {})
            postings[doc_id] = postings.get(doc_id, 0) + 1

    def _drop_index(self, doc_id):
        self._docs.pop(doc_id, None)
        self._doc_len.pop(doc_id, None)
        for postings in self._postings.values():
            postings.pop(doc_id, None)

    # ── 公开接口（与 v1 兼容）───────────────────────────────────────
    def add(self, text, context="", importance=None):
        """写入一条观测笔记。相同内容（规范化后 hash 相同）只更新不新增。

        importance：1-10，越界自动截断；None 时默认 5。
        返回写入/更新后的笔记 dict。
        """
        text = (text or "").strip()
        if not text:
            return None
        importance = int(importance) if importance is not None else 5
        importance = max(1, min(10, importance))
        context = (context or "")[:200]

        with self._lock:
            with self._conn:  # 事务：成功提交，异常回滚
                digest = _sha256(_normalize(text))
                row = self._conn.execute(
                    "SELECT id, importance FROM notes WHERE hash = ?", (digest,)
                ).fetchone()
                now = datetime.now().isoformat(timespec="seconds")
                if row:
                    doc_id, old_imp = row
                    # 重复确认：重要度只增不减，时间刷新，context 保留较新者
                    new_imp = max(old_imp, importance)
                    self._conn.execute(
                        "UPDATE notes SET time = ?, context = ?, importance = ?,"
                        " updated_at = ? WHERE id = ?",
                        (now, context, new_imp, now, doc_id),
                    )
                    self._drop_index(doc_id)
                    self._index_doc(doc_id, text, context, new_imp, now)
                    note = self._docs[doc_id]
                else:
                    doc_id = self._insert_row(now, text, context, importance)
                    self._index_doc(doc_id, text, context, importance, now)
                    note = self._docs[doc_id]
            return dict(note, time=now)

    def search(self, query, top_k=5, since=None, until=None):
        """BM25 打分检索相关笔记；无命中返回空列表（不注入）。

        since/until：ISO 时间字符串，按 time 字段过滤（供后续情绪片段/时间线使用）。
        """
        terms = _query_tokens(query or "")
        if not terms:
            return []
        with self._lock:
            n = len(self._docs)
            if n == 0:
                return []
            avgdl = sum(self._doc_len.values()) / n

            scores = {}
            for term in terms:
                postings = self._postings.get(term)
                if not postings:
                    continue
                df = len(postings)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                for doc_id, tf in postings.items():
                    dl = self._doc_len.get(doc_id, 0)
                    denom = tf + _K1 * (1 - _B + _B * dl / avgdl)
                    scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (_K1 + 1) / denom

            results = []
            for doc_id, bm25 in scores.items():
                note = self._docs[doc_id]
                if since and note.get("time", "") < since:
                    continue
                if until and note.get("time", "") > until:
                    continue
                results.append(
                    (bm25 + (note["importance"] - 5) * _IMPORTANCE_WEIGHT, note)
                )
            results.sort(key=lambda x: -x[0])
            return [note for _, note in results[:top_k]]

    def get(self, note_id):
        with self._lock:
            return dict(self._docs[note_id]) if note_id in self._docs else None

    def all(self):
        """全部笔记（按时间倒序），供调试/管理用。"""
        with self._lock:
            rows = sorted(self._docs.values(), key=lambda x: x["time"], reverse=True)
            return [dict(r) for r in rows]

    def close(self):
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __len__(self):
        with self._lock:
            return len(self._docs)
