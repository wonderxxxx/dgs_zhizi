"""反思记忆：合成高层次的反思和洞察。

简化版实现：
- 从事实和对话中生成反思
- 支持反思的确认和管理
- 提供反思洞察
"""

import json
import os
import re
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional


class ReflectionMemory:
    """反思记忆管理器。"""
    
    def __init__(self, memory_dir: str, llm_client=None):
        """
        Args:
            memory_dir: 记忆存储目录
            llm_client: LLM 客户端实例（用于生成反思）
        """
        self.memory_dir = memory_dir
        self.llm_client = llm_client
        self._lock = threading.RLock()
        
        # 确保目录存在
        os.makedirs(memory_dir, exist_ok=True)
        
        # 数据库路径
        self._db_path = os.path.join(memory_dir, "reflections.db")
        
        # 初始化数据库
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
    
    def _init_schema(self):
        """初始化数据库表结构。"""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    reflection  TEXT NOT NULL,
                    insight     TEXT NOT NULL DEFAULT '',
                    source_facts TEXT NOT NULL DEFAULT '[]',
                    status      TEXT NOT NULL DEFAULT 'pending',
                    confidence  REAL NOT NULL DEFAULT 0.5,
                    created_at  TEXT NOT NULL,
                    confirmed_at TEXT,
                    UNIQUE(user_id, reflection)
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reflections_user ON reflections(user_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reflections_status ON reflections(user_id, status)"
            )
    
    def generate_reflection(self, user_id: str, facts: List[Dict], 
                           recent_conversation: str = "") -> List[Dict]:
        """从事实和对话中生成反思。
        
        Args:
            user_id: 用户ID
            facts: 事实列表
            recent_conversation: 最近的对话
            
        Returns:
            生成的反思列表
        """
        if not self.llm_client or not facts:
            return []
        
        # 构建事实摘要
        facts_text = "\n".join([f"- {f['fact']}" for f in facts[:10]])
        
        prompt = f"""基于以下关于用户的信息，请生成深层次的反思和洞察。

用户事实：
{facts_text}

{f"最近对话：{recent_conversation}" if recent_conversation else ""}

请生成 2-3 个有价值的反思，每个反思应该：
1. 基于多个事实进行推理
2. 揭示用户的深层需求、偏好或模式
3. 对后续互动有指导意义

请以 JSON 数组格式返回，每个反思包含：
- reflection: 反思内容
- insight: 洞察说明
- confidence: 置信度 (0-1)

示例：
[
    {{"reflection": "用户可能在寻求情感支持", "洞察": "从用户频繁讨论工作压力可以看出", "confidence": 0.7}},
    {{"reflection": "用户对技术有浓厚兴趣", "洞察": "用户多次询问编程相关问题", "confidence": 0.8}}
]

反思："""
        
        try:
            response = self.llm_client.chat([
                {"role": "user", "content": prompt}
            ], temperature=0.4)
            
            # 解析 JSON 响应
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                reflections_data = json.loads(json_match.group())
            else:
                reflections_data = json.loads(response)
            
            # 存储反思
            results = []
            for item in reflections_data:
                if isinstance(item, dict) and "reflection" in item:
                    reflection = self.add_reflection(
                        user_id=user_id,
                        reflection=item["reflection"],
                        insight=item.get("insight", ""),
                        source_facts=[f["fact"] for f in facts[:5]],
                        confidence=float(item.get("confidence", 0.5))
                    )
                    if reflection:
                        results.append(reflection)
            
            return results
            
        except Exception as e:
            print(f"反思生成失败: {e}")
            return []
    
    def add_reflection(self, user_id: str, reflection: str, insight: str = "",
                      source_facts: List[str] = None, 
                      confidence: float = 0.5) -> Optional[Dict]:
        """添加一个反思。
        
        Args:
            user_id: 用户ID
            reflection: 反思内容
            insight: 洞察说明
            source_facts: 来源事实
            confidence: 置信度
            
        Returns:
            添加的反思条目
        """
        reflection = (reflection or "").strip()
        if not reflection:
            return None
        
        confidence = max(0.0, min(1.0, confidence))
        now = datetime.now().isoformat(timespec="seconds")
        source_facts = source_facts or []
        
        with self._lock:
            with self._conn:
                # 检查是否已存在
                existing = self._conn.execute(
                    "SELECT id FROM reflections WHERE user_id = ? AND reflection = ?",
                    (user_id, reflection)
                ).fetchone()
                
                if existing:
                    return None  # 已存在，不重复添加
                
                # 插入新反思
                cur = self._conn.execute(
                    "INSERT INTO reflections(user_id, reflection, insight, source_facts, confidence, created_at)"
                    " VALUES(?, ?, ?, ?, ?, ?)",
                    (user_id, reflection, insight, json.dumps(source_facts), confidence, now)
                )
                reflection_id = cur.lastrowid
                
                return {
                    "id": reflection_id,
                    "user_id": user_id,
                    "reflection": reflection,
                    "insight": insight,
                    "source_facts": source_facts,
                    "status": "pending",
                    "confidence": confidence,
                    "created_at": now,
                    "confirmed_at": None
                }
    
    def get_reflections(self, user_id: str, status: str = None, 
                       limit: int = 50) -> List[Dict]:
        """获取用户的反思列表。
        
        Args:
            user_id: 用户ID
            status: 状态过滤 (pending/confirmed/rejected)
            limit: 返回的最大数量
            
        Returns:
            反思列表
        """
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT id, reflection, insight, source_facts, status, confidence, created_at, confirmed_at"
                    " FROM reflections WHERE user_id = ? AND status = ?"
                    " ORDER BY confidence DESC, created_at DESC LIMIT ?",
                    (user_id, status, limit)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, reflection, insight, source_facts, status, confidence, created_at, confirmed_at"
                    " FROM reflections WHERE user_id = ?"
                    " ORDER BY confidence DESC, created_at DESC LIMIT ?",
                    (user_id, limit)
                ).fetchall()
            
            return [
                {
                    "id": row[0],
                    "user_id": user_id,
                    "reflection": row[1],
                    "insight": row[2],
                    "source_facts": json.loads(row[3]) if row[3] else [],
                    "status": row[4],
                    "confidence": row[5],
                    "created_at": row[6],
                    "confirmed_at": row[7]
                }
                for row in rows
            ]
    
    def confirm_reflection(self, user_id: str, reflection_id: int) -> bool:
        """确认一个反思。"""
        return self._update_status(user_id, reflection_id, "confirmed")
    
    def reject_reflection(self, user_id: str, reflection_id: int) -> bool:
        """拒绝一个反思。"""
        return self._update_status(user_id, reflection_id, "rejected")
    
    def _update_status(self, user_id: str, reflection_id: int, status: str) -> bool:
        """更新反思状态。"""
        with self._lock:
            # 检查是否存在
            existing = self._conn.execute(
                "SELECT id FROM reflections WHERE id = ? AND user_id = ?",
                (reflection_id, user_id)
            ).fetchone()
            
            if not existing:
                return False
            
            now = datetime.now().isoformat(timespec="seconds")
            with self._conn:
                if status == "confirmed":
                    self._conn.execute(
                        "UPDATE reflections SET status = ?, confirmed_at = ? WHERE id = ?",
                        (status, now, reflection_id)
                    )
                else:
                    self._conn.execute(
                        "UPDATE reflections SET status = ? WHERE id = ?",
                        (status, reflection_id)
                    )
            return True
    
    def get_confirmed_reflections(self, user_id: str, limit: int = 20) -> List[Dict]:
        """获取已确认的反思。"""
        return self.get_reflections(user_id, status="confirmed", limit=limit)
    
    def get_pending_reflections(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取待确认的反思。"""
        return self.get_reflections(user_id, status="pending", limit=limit)
    
    def delete_reflection(self, user_id: str, reflection_id: int) -> bool:
        """删除指定反思。"""
        with self._lock:
            # 检查是否存在
            existing = self._conn.execute(
                "SELECT id FROM reflections WHERE id = ? AND user_id = ?",
                (reflection_id, user_id)
            ).fetchone()
            
            if not existing:
                return False
            
            with self._conn:
                self._conn.execute("DELETE FROM reflections WHERE id = ?", (reflection_id,))
            return True
    
    def clear(self, user_id: str) -> int:
        """清空该用户的全部反思。返回删除条数。"""
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM reflections WHERE user_id = ?", (user_id,)
                )
                return cur.rowcount
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户反思统计信息。"""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM reflections WHERE user_id = ?",
                (user_id,)
            ).fetchone()[0]
            
            pending = self._conn.execute(
                "SELECT COUNT(*) FROM reflections WHERE user_id = ? AND status = 'pending'",
                (user_id,)
            ).fetchone()[0]
            
            confirmed = self._conn.execute(
                "SELECT COUNT(*) FROM reflections WHERE user_id = ? AND status = 'confirmed'",
                (user_id,)
            ).fetchone()[0]
            
            avg_confidence = self._conn.execute(
                "SELECT AVG(confidence) FROM reflections WHERE user_id = ?",
                (user_id,)
            ).fetchone()[0] or 0.0
            
            return {
                "user_id": user_id,
                "total_reflections": total,
                "pending": pending,
                "confirmed": confirmed,
                "rejected": total - pending - confirmed,
                "average_confidence": round(avg_confidence, 3)
            }
    
    def close(self):
        """关闭数据库连接。"""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass