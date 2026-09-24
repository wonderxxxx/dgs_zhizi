"""近期记忆：基于时间索引的记忆存储。

简化版实现：
- 使用 SQLite 存储记忆条目
- 支持时间范围查询
- 支持 BM25 文本检索
- 按用户隔离
"""

import json
import math
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple


# BM25 参数
_K1 = 1.2
_B = 0.75
_IMPORTANCE_WEIGHT = 0.3

# 分词正则
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE = re.compile(r"[A-Za-z0-9_]+")
_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[。！？!?．.、，,；;：:]+$")


def _tokenize(text: str, include_unigram: bool = True) -> set:
    """分词：英文/数字整词 + 中文 bigram。"""
    out = set()
    for part in _ASCII_RE.findall(text.lower()):
        out.add(part)
    for part in _CJK_RE.findall(text):
        if len(part) == 1:
            if include_unigram:
                out.add(part)
        else:
            for i in range(len(part) - 1):
                out.add(part[i:i + 2])  # bigram
            if include_unigram:
                for i in range(len(part)):
                    out.add(part[i])  # unigram
    return out


def _normalize(text: str) -> str:
    """规范化文本用于去重。"""
    return _TRAILING_PUNCT_RE.sub("", _WS_RE.sub(" ", text.strip()))


def _sha256(text: str) -> str:
    """计算 SHA-256 哈希。"""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _query_tokens(query: str) -> set:
    """查询侧分词。"""
    parts = _tokenize(query, include_unigram=True)
    has_cjk_bigram = any(len(t) == 2 and not t.isascii() for t in parts)
    if has_cjk_bigram:
        return _tokenize(query, include_unigram=False)
    return parts


class TimeIndexedMemory:
    """基于时间索引的记忆存储。"""
    
    def __init__(self, memory_dir: str):
        """
        Args:
            memory_dir: 记忆存储目录
        """
        self.memory_dir = memory_dir
        self._lock = threading.RLock()
        
        # 确保目录存在
        os.makedirs(memory_dir, exist_ok=True)
        
        # 数据库路径
        self._db_path = os.path.join(memory_dir, "timeindex.db")
        
        # 初始化数据库
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        
        # 内存索引
        self._docs: Dict[int, Dict] = {}
        self._postings: Dict[str, Dict[int, int]] = {}
        self._doc_len: Dict[int, int] = {}
        self._load_index()
    
    def _init_schema(self):
        """初始化数据库表结构。"""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    time        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    category    TEXT NOT NULL DEFAULT '',
                    importance  INTEGER NOT NULL DEFAULT 5,
                    hash        TEXT NOT NULL,
                    metadata    TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(user_id, hash)
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_time ON memories(user_id, time)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(user_id, category)"
            )
    
    def _load_index(self):
        """加载内存索引。"""
        rows = self._conn.execute(
            "SELECT id, user_id, time, content, category, importance FROM memories"
        ).fetchall()
        
        for row in rows:
            doc_id, user_id, time, content, category, importance = row
            self._index_doc(doc_id, user_id, content, category, importance, time)
    
    def _index_doc(self, doc_id: int, user_id: str, content: str, category: str, 
                   importance: int, time: str):
        """索引一个文档。"""
        tokens = _tokenize(content)
        self._docs[doc_id] = {
            "id": doc_id,
            "user_id": user_id,
            "content": content,
            "category": category,
            "importance": importance,
            "time": time,
        }
        self._doc_len[doc_id] = len(tokens)
        for term in tokens:
            if term not in self._postings:
                self._postings[term] = {}
            self._postings[term][doc_id] = self._postings[term].get(doc_id, 0) + 1
    
    def _drop_index(self, doc_id: int):
        """从索引中移除文档。"""
        self._docs.pop(doc_id, None)
        self._doc_len.pop(doc_id, None)
        for postings in self._postings.values():
            postings.pop(doc_id, None)
    
    def add(self, user_id: str, content: str, category: str = "", 
            importance: int = 5, metadata: Dict = None) -> Dict:
        """添加一条记忆。
        
        Args:
            user_id: 用户ID
            content: 记忆内容
            category: 分类标签
            importance: 重要度 (1-10)
            metadata: 额外元数据
            
        Returns:
            添加的记忆条目
        """
        content = (content or "").strip()
        if not content:
            return None
        
        importance = max(1, min(10, importance))
        now = datetime.now().isoformat(timespec="seconds")
        digest = _sha256(_normalize(content))
        metadata = metadata or {}
        
        with self._lock:
            with self._conn:
                # 检查是否已存在
                existing = self._conn.execute(
                    "SELECT id, importance FROM memories WHERE user_id = ? AND hash = ?",
                    (user_id, digest)
                ).fetchone()
                
                if existing:
                    # 更新已存在的记忆
                    doc_id, old_imp = existing
                    new_imp = max(old_imp, importance)
                    self._conn.execute(
                        "UPDATE memories SET time = ?, importance = ?, metadata = ? WHERE id = ?",
                        (now, new_imp, json.dumps(metadata), doc_id)
                    )
                    self._drop_index(doc_id)
                    self._index_doc(doc_id, user_id, content, category, new_imp, now)
                    return self._docs[doc_id]
                else:
                    # 插入新记忆
                    cur = self._conn.execute(
                        "INSERT INTO memories(user_id, time, content, category, importance, hash, metadata)"
                        " VALUES(?, ?, ?, ?, ?, ?, ?)",
                        (user_id, now, content, category, importance, digest, json.dumps(metadata))
                    )
                    doc_id = cur.lastrowid
                    self._index_doc(doc_id, user_id, content, category, importance, now)
                    return self._docs[doc_id]
    
    def _score_docs(self, user_id: str, terms: set, since: str = None,
                    until: str = None, category: str = None) -> List[Tuple[float, Dict, set]]:
        """BM25 打分（含匹配 term 明细），供 search / search_traced 共用。

        Returns:
            [(final_score, doc, matched_terms), ...] 按分数降序
        """
        # 过滤用户记忆
        user_docs = {
            doc_id: doc for doc_id, doc in self._docs.items()
            if doc["user_id"] == user_id
        }
        if not user_docs:
            return []

        n = len(user_docs)
        avgdl = sum(self._doc_len.get(doc_id, 0) for doc_id in user_docs) / n if n > 0 else 0

        scores = {}
        matched: Dict[int, set] = {}
        for term in terms:
            postings = self._postings.get(term, {})
            for doc_id, tf in postings.items():
                if doc_id not in user_docs:
                    continue
                df = len(postings)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                dl = self._doc_len.get(doc_id, 0)
                denom = tf + _K1 * (1 - _B + _B * dl / avgdl) if avgdl > 0 else tf
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (_K1 + 1) / denom
                matched.setdefault(doc_id, set()).add(term)

        # 应用过滤和排序
        results = []
        for doc_id, bm25 in scores.items():
            doc = user_docs[doc_id]
            if since and doc.get("time", "") < since:
                continue
            if until and doc.get("time", "") > until:
                continue
            if category and doc.get("category", "") != category:
                continue
            final_score = bm25 + (doc["importance"] - 5) * _IMPORTANCE_WEIGHT
            results.append((final_score, doc, matched.get(doc_id, set())))

        results.sort(key=lambda x: -x[0])
        return results

    def search(self, user_id: str, query: str, top_k: int = 5, 
               since: str = None, until: str = None,
               category: str = None) -> List[Dict]:
        """搜索记忆。
        
        Args:
            user_id: 用户ID
            query: 搜索查询
            top_k: 返回的最大结果数
            since: 起始时间 (ISO格式)
            until: 结束时间 (ISO格式)
            category: 分类过滤
            
        Returns:
            匹配的记忆列表
        """
        terms = _query_tokens(query or "")
        if not terms:
            return []
        
        with self._lock:
            results = self._score_docs(user_id, terms, since, until, category)
            return [doc for _, doc, _ in results[:top_k]]

    def search_traced(self, user_id: str, query: str, top_k: int = 5,
                      since: str = None, until: str = None,
                      category: str = None) -> List[Dict]:
        """搜索记忆并返回召回明细（评分 + 匹配 term），供召回过程观测。"""
        terms = _query_tokens(query or "")
        if not terms:
            return []

        with self._lock:
            results = self._score_docs(user_id, terms, since, until, category)
            out = []
            for final_score, doc, matched_terms in results[:top_k]:
                out.append({
                    **doc,
                    "score": round(final_score, 3),
                    "matched_terms": sorted(matched_terms),
                })
            return out

    def all(self, user_id: str, limit: int = 500) -> List[Dict]:
        """该用户的全部记忆（时间倒序），供查看/调试用。"""
        with self._lock:
            docs = [
                doc.copy() for doc in self._docs.values()
                if doc["user_id"] == user_id
            ]
            docs.sort(key=lambda x: x["time"], reverse=True)
            return docs[:limit]
    
    def get(self, user_id: str, memory_id: int) -> Optional[Dict]:
        """获取指定记忆。"""
        with self._lock:
            doc = self._docs.get(memory_id)
            if doc and doc["user_id"] == user_id:
                return doc.copy()
            return None
    
    def get_by_time_range(self, user_id: str, start_time: str, end_time: str) -> List[Dict]:
        """按时间范围获取记忆。"""
        with self._lock:
            return [
                doc.copy() for doc in self._docs.values()
                if doc["user_id"] == user_id 
                and start_time <= doc.get("time", "") <= end_time
            ]
    
    def get_by_category(self, user_id: str, category: str) -> List[Dict]:
        """按分类获取记忆。"""
        with self._lock:
            return [
                doc.copy() for doc in self._docs.values()
                if doc["user_id"] == user_id and doc.get("category", "") == category
            ]
    
    def delete(self, user_id: str, memory_id: int) -> bool:
        """删除指定记忆。"""
        with self._lock:
            doc = self._docs.get(memory_id)
            if not doc or doc["user_id"] != user_id:
                return False
            
            with self._conn:
                self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._drop_index(memory_id)
            return True
    
    def clear(self, user_id: str) -> int:
        """清空该用户的全部近期记忆。返回删除条数。"""
        with self._lock:
            ids = [
                doc_id for doc_id, doc in self._docs.items()
                if doc["user_id"] == user_id
            ]
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM memories WHERE user_id = ?", (user_id,)
                )
                deleted = cur.rowcount
            for doc_id in ids:
                self._drop_index(doc_id)
            return deleted
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户记忆统计信息。"""
        with self._lock:
            user_docs = [
                doc for doc in self._docs.values() 
                if doc["user_id"] == user_id
            ]
            
            categories = {}
            for doc in user_docs:
                cat = doc.get("category", "")
                categories[cat] = categories.get(cat, 0) + 1
            
            return {
                "user_id": user_id,
                "total_memories": len(user_docs),
                "categories": categories,
                "latest_memory": max(
                    (doc.get("time", "") for doc in user_docs), 
                    default=None
                )
            }
    
    def close(self):
        """关闭数据库连接。"""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass