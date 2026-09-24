"""事实记忆：从对话中提取原子事实并存储。

简化版实现：
- 使用 LLM 从对话中提取事实
- 使用 SHA-256 哈希去重
- 支持事实检索和管理
"""

import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional


def _normalize(text: str) -> str:
    """规范化文本用于去重。"""
    return re.sub(r"\s+", " ", text.strip()).strip("。！？!?．.、，,；;：:")


def _sha256(text: str) -> str:
    """计算 SHA-256 哈希。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FactMemory:
    """事实记忆管理器。"""
    
    def __init__(self, memory_dir: str, llm_client=None):
        """
        Args:
            memory_dir: 记忆存储目录
            llm_client: LLM 客户端实例（用于提取事实）
        """
        self.memory_dir = memory_dir
        self.llm_client = llm_client
        self._lock = threading.RLock()
        
        # 确保目录存在
        os.makedirs(memory_dir, exist_ok=True)
        
        # 数据库路径
        self._db_path = os.path.join(memory_dir, "facts.db")
        
        # 初始化数据库
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
    
    def _init_schema(self):
        """初始化数据库表结构。"""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    fact        TEXT NOT NULL,
                    source      TEXT NOT NULL DEFAULT '',
                    confidence  REAL NOT NULL DEFAULT 1.0,
                    hash        TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    UNIQUE(user_id, hash)
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id)"
            )
    
    def extract_facts(self, user_id: str, conversation: str) -> List[Dict]:
        """从对话中提取事实。
        
        Args:
            user_id: 用户ID
            conversation: 对话内容
            
        Returns:
            提取的事实列表
        """
        if not self.llm_client:
            return []
        
        prompt = f"""请从以下对话中提取关于用户的关键事实信息。

要求：
1. 每个事实应该是独立的、原子性的
2. 事实应该是陈述句，如"用户喜欢猫"
3. 只提取明确提到的信息，不要推测
4. 事实应该具体、有价值

对话：
{conversation}

请以 JSON 数组格式返回提取的事实，每个事实包含：
- fact: 事实内容
- confidence: 置信度 (0-1)

示例：
[
    {{"fact": "用户喜欢猫", "confidence": 0.9}},
    {{"fact": "用户是一名程序员", "confidence": 0.8}}
]

提取的事实："""
        
        try:
            response = self.llm_client.chat([
                {"role": "user", "content": prompt}
            ], temperature=0.2)
            
            # 解析 JSON 响应
            # 尝试提取 JSON 部分
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                facts_data = json.loads(json_match.group())
            else:
                facts_data = json.loads(response)
            
            # 存储事实
            results = []
            for item in facts_data:
                if isinstance(item, dict) and "fact" in item:
                    fact = self.add_fact(
                        user_id=user_id,
                        fact=item["fact"],
                        source=conversation[:200],  # 保存部分源对话
                        confidence=float(item.get("confidence", 1.0))
                    )
                    if fact:
                        results.append(fact)
            
            return results
            
        except Exception as e:
            print(f"事实提取失败: {e}")
            return []
    
    def add_fact(self, user_id: str, fact: str, source: str = "", 
                 confidence: float = 1.0) -> Optional[Dict]:
        """添加一个事实。
        
        Args:
            user_id: 用户ID
            fact: 事实内容
            source: 来源
            confidence: 置信度
            
        Returns:
            添加的事实条目
        """
        fact = (fact or "").strip()
        if not fact:
            return None
        
        confidence = max(0.0, min(1.0, confidence))
        now = datetime.now().isoformat(timespec="seconds")
        digest = _sha256(_normalize(fact))
        
        with self._lock:
            with self._conn:
                # 检查是否已存在
                existing = self._conn.execute(
                    "SELECT id, confidence FROM facts WHERE user_id = ? AND hash = ?",
                    (user_id, digest)
                ).fetchone()
                
                if existing:
                    # 更新已存在的事实
                    fact_id, old_conf = existing
                    new_conf = max(old_conf, confidence)
                    self._conn.execute(
                        "UPDATE facts SET confidence = ?, updated_at = ? WHERE id = ?",
                        (new_conf, now, fact_id)
                    )
                    return {
                        "id": fact_id,
                        "user_id": user_id,
                        "fact": fact,
                        "source": source,
                        "confidence": new_conf,
                        "created_at": now,
                        "updated_at": now
                    }
                else:
                    # 插入新事实
                    cur = self._conn.execute(
                        "INSERT INTO facts(user_id, fact, source, confidence, hash, created_at, updated_at)"
                        " VALUES(?, ?, ?, ?, ?, ?, ?)",
                        (user_id, fact, source, confidence, digest, now, now)
                    )
                    fact_id = cur.lastrowid
                    return {
                        "id": fact_id,
                        "user_id": user_id,
                        "fact": fact,
                        "source": source,
                        "confidence": confidence,
                        "created_at": now,
                        "updated_at": now
                    }
    
    def get_facts(self, user_id: str, limit: int = 100, 
                  min_confidence: float = 0.5) -> List[Dict]:
        """获取用户的事实列表。
        
        Args:
            user_id: 用户ID
            limit: 返回的最大数量
            min_confidence: 最小置信度
            
        Returns:
            事实列表
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, fact, source, confidence, created_at, updated_at"
                " FROM facts WHERE user_id = ? AND confidence >= ?"
                " ORDER BY confidence DESC, updated_at DESC LIMIT ?",
                (user_id, min_confidence, limit)
            ).fetchall()
            
            return [
                {
                    "id": row[0],
                    "user_id": user_id,
                    "fact": row[1],
                    "source": row[2],
                    "confidence": row[3],
                    "created_at": row[4],
                    "updated_at": row[5]
                }
                for row in rows
            ]
    
    def search_facts(self, user_id: str, query: str, top_k: int = 10) -> List[Dict]:
        """搜索事实。
        
        Args:
            user_id: 用户ID
            query: 搜索查询
            top_k: 返回的最大结果数
            
        Returns:
            匹配的事实列表
        """
        # 简单实现：使用 LIKE 搜索
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, fact, source, confidence, created_at, updated_at"
                " FROM facts WHERE user_id = ? AND fact LIKE ?"
                " ORDER BY confidence DESC LIMIT ?",
                (user_id, f"%{query}%", top_k)
            ).fetchall()
            
            return [
                {
                    "id": row[0],
                    "user_id": user_id,
                    "fact": row[1],
                    "source": row[2],
                    "confidence": row[3],
                    "created_at": row[4],
                    "updated_at": row[5]
                }
                for row in rows
            ]
    
    def get_fact(self, user_id: str, fact_id: int) -> Optional[Dict]:
        """获取指定事实。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, fact, source, confidence, created_at, updated_at"
                " FROM facts WHERE id = ? AND user_id = ?",
                (fact_id, user_id)
            ).fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "user_id": user_id,
                    "fact": row[1],
                    "source": row[2],
                    "confidence": row[3],
                    "created_at": row[4],
                    "updated_at": row[5]
                }
            return None
    
    def delete_fact(self, user_id: str, fact_id: int) -> bool:
        """删除指定事实。"""
        with self._lock:
            # 检查是否存在
            existing = self._conn.execute(
                "SELECT id FROM facts WHERE id = ? AND user_id = ?",
                (fact_id, user_id)
            ).fetchone()
            
            if not existing:
                return False
            
            with self._conn:
                self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            return True
    
    def clear(self, user_id: str) -> int:
        """清空该用户的全部事实。返回删除条数。"""
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM facts WHERE user_id = ?", (user_id,)
                )
                return cur.rowcount
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户事实统计信息。"""
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE user_id = ?",
                (user_id,)
            ).fetchone()[0]
            
            avg_confidence = self._conn.execute(
                "SELECT AVG(confidence) FROM facts WHERE user_id = ?",
                (user_id,)
            ).fetchone()[0] or 0.0
            
            return {
                "user_id": user_id,
                "total_facts": count,
                "average_confidence": round(avg_confidence, 3)
            }
    
    def close(self):
        """关闭数据库连接。"""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass