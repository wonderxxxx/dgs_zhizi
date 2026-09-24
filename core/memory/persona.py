"""人格记忆：管理角色的人格特征和一致性。

简化版实现：
- 存储和检索人格特征
- 支持人格特征的更新和融合
- 提供人格一致性检查
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional


class PersonaMemory:
    """人格记忆管理器。"""
    
    def __init__(self, memory_dir: str, llm_client=None):
        """
        Args:
            memory_dir: 记忆存储目录
            llm_client: LLM 客户端实例（用于人格分析）
        """
        self.memory_dir = memory_dir
        self.llm_client = llm_client
        self._lock = threading.RLock()
        
        # 确保目录存在
        os.makedirs(memory_dir, exist_ok=True)
        
        # 数据库路径
        self._db_path = os.path.join(memory_dir, "persona.db")
        
        # 初始化数据库
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
    
    def _init_schema(self):
        """初始化数据库表结构。"""
        with self._conn:
            # 人格特征表
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS traits (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    trait       TEXT NOT NULL,
                    category    TEXT NOT NULL DEFAULT 'general',
                    value       TEXT NOT NULL DEFAULT '',
                    confidence  REAL NOT NULL DEFAULT 0.5,
                    source      TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    UNIQUE(user_id, trait)
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traits_user ON traits(user_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traits_category ON traits(user_id, category)"
            )
            
            # 人格摘要表
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    summary     TEXT NOT NULL,
                    version     INTEGER NOT NULL DEFAULT 1,
                    created_at  TEXT NOT NULL,
                    UNIQUE(user_id, version)
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_summaries_user ON summaries(user_id)"
            )
    
    def add_trait(self, user_id: str, trait: str, value: str, 
                 category: str = "general", confidence: float = 0.5,
                 source: str = "") -> Optional[Dict]:
        """添加或更新一个人格特征。
        
        Args:
            user_id: 用户ID
            trait: 特征名称（如"性格"、"爱好"）
            value: 特征值
            category: 分类
            confidence: 置信度
            source: 来源
            
        Returns:
            添加/更新的特征条目
        """
        trait = (trait or "").strip()
        value = (value or "").strip()
        if not trait or not value:
            return None
        
        confidence = max(0.0, min(1.0, confidence))
        now = datetime.now().isoformat(timespec="seconds")
        
        with self._lock:
            with self._conn:
                # 检查是否已存在
                existing = self._conn.execute(
                    "SELECT id, confidence FROM traits WHERE user_id = ? AND trait = ?",
                    (user_id, trait)
                ).fetchone()
                
                if existing:
                    # 更新已存在的特征
                    trait_id, old_conf = existing
                    new_conf = max(old_conf, confidence)
                    self._conn.execute(
                        "UPDATE traits SET value = ?, category = ?, confidence = ?, source = ?, updated_at = ?"
                        " WHERE id = ?",
                        (value, category, new_conf, source, now, trait_id)
                    )
                    return {
                        "id": trait_id,
                        "user_id": user_id,
                        "trait": trait,
                        "category": category,
                        "value": value,
                        "confidence": new_conf,
                        "source": source,
                        "created_at": now,
                        "updated_at": now
                    }
                else:
                    # 插入新特征
                    cur = self._conn.execute(
                        "INSERT INTO traits(user_id, trait, category, value, confidence, source, created_at, updated_at)"
                        " VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                        (user_id, trait, category, value, confidence, source, now, now)
                    )
                    trait_id = cur.lastrowid
                    return {
                        "id": trait_id,
                        "user_id": user_id,
                        "trait": trait,
                        "category": category,
                        "value": value,
                        "confidence": confidence,
                        "source": source,
                        "created_at": now,
                        "updated_at": now
                    }
    
    def get_traits(self, user_id: str, category: str = None, 
                  min_confidence: float = 0.3) -> List[Dict]:
        """获取用户的人格特征。
        
        Args:
            user_id: 用户ID
            category: 分类过滤
            min_confidence: 最小置信度
            
        Returns:
            特征列表
        """
        with self._lock:
            if category:
                rows = self._conn.execute(
                    "SELECT id, trait, category, value, confidence, source, created_at, updated_at"
                    " FROM traits WHERE user_id = ? AND category = ? AND confidence >= ?"
                    " ORDER BY confidence DESC",
                    (user_id, category, min_confidence)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, trait, category, value, confidence, source, created_at, updated_at"
                    " FROM traits WHERE user_id = ? AND confidence >= ?"
                    " ORDER BY confidence DESC",
                    (user_id, min_confidence)
                ).fetchall()
            
            return [
                {
                    "id": row[0],
                    "user_id": user_id,
                    "trait": row[1],
                    "category": row[2],
                    "value": row[3],
                    "confidence": row[4],
                    "source": row[5],
                    "created_at": row[6],
                    "updated_at": row[7]
                }
                for row in rows
            ]
    
    def get_trait(self, user_id: str, trait: str) -> Optional[Dict]:
        """获取指定特征。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, category, value, confidence, source, created_at, updated_at"
                " FROM traits WHERE user_id = ? AND trait = ?",
                (user_id, trait)
            ).fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "user_id": user_id,
                    "trait": trait,
                    "category": row[1],
                    "value": row[2],
                    "confidence": row[3],
                    "source": row[4],
                    "created_at": row[5],
                    "updated_at": row[6]
                }
            return None
    
    def delete_trait(self, user_id: str, trait: str) -> bool:
        """删除指定特征。"""
        with self._lock:
            # 检查是否存在
            existing = self._conn.execute(
                "SELECT id FROM traits WHERE user_id = ? AND trait = ?",
                (user_id, trait)
            ).fetchone()
            
            if not existing:
                return False
            
            with self._conn:
                self._conn.execute(
                    "DELETE FROM traits WHERE user_id = ? AND trait = ?",
                    (user_id, trait)
                )
            return True
    
    def clear(self, user_id: str) -> Dict[str, int]:
        """清空该用户的人格特征与摘要。返回 {traits, summaries} 删除数。"""
        with self._lock:
            with self._conn:
                traits = self._conn.execute(
                    "DELETE FROM traits WHERE user_id = ?", (user_id,)
                ).rowcount
                summaries = self._conn.execute(
                    "DELETE FROM summaries WHERE user_id = ?", (user_id,)
                ).rowcount
            return {"traits": traits, "summaries": summaries}
    
    def update_summary(self, user_id: str, summary: str) -> Dict:
        """更新人格摘要。
        
        Args:
            user_id: 用户ID
            summary: 人格摘要
            
        Returns:
            摘要信息
        """
        summary = (summary or "").strip()
        if not summary:
            return None
        
        now = datetime.now().isoformat(timespec="seconds")
        
        with self._lock:
            with self._conn:
                # 获取当前版本号
                current = self._conn.execute(
                    "SELECT MAX(version) FROM summaries WHERE user_id = ?",
                    (user_id,)
                ).fetchone()[0] or 0
                
                new_version = current + 1
                
                # 插入新版本
                cur = self._conn.execute(
                    "INSERT INTO summaries(user_id, summary, version, created_at)"
                    " VALUES(?, ?, ?, ?)",
                    (user_id, summary, new_version, now)
                )
                
                return {
                    "user_id": user_id,
                    "summary": summary,
                    "version": new_version,
                    "created_at": now
                }
    
    def get_latest_summary(self, user_id: str) -> Optional[Dict]:
        """获取最新的人格摘要。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT summary, version, created_at FROM summaries"
                " WHERE user_id = ? ORDER BY version DESC LIMIT 1",
                (user_id,)
            ).fetchone()
            
            if row:
                return {
                    "user_id": user_id,
                    "summary": row[0],
                    "version": row[1],
                    "created_at": row[2]
                }
            return None
    
    def get_summary_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取人格摘要历史。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT summary, version, created_at FROM summaries"
                " WHERE user_id = ? ORDER BY version DESC LIMIT ?",
                (user_id, limit)
            ).fetchall()
            
            return [
                {
                    "user_id": user_id,
                    "summary": row[0],
                    "version": row[1],
                    "created_at": row[2]
                }
                for row in rows
            ]
    
    def analyze_persona(self, user_id: str, facts: List[Dict]) -> Dict:
        """分析用户人格并生成摘要。
        
        Args:
            user_id: 用户ID
            facts: 事实列表
            
        Returns:
            分析结果
        """
        if not self.llm_client or not facts:
            return None
        
        # 构建事实摘要
        facts_text = "\n".join([f"- {f['fact']}" for f in facts[:15]])
        
        prompt = f"""基于以下关于用户的信息，请分析用户的人格特征并生成摘要。

用户事实：
{facts_text}

请分析并返回：
1. 主要人格特征（性格、爱好、价值观等）
2. 行为模式
3. 沟通偏好
4. 潜在需求

请以 JSON 格式返回，包含以下字段：
- traits: 特征列表（每个特征包含 trait, value, category, confidence）
- summary: 人格摘要（2-3句话）

示例：
{{
    "traits": [
        {{"trait": "性格", "value": "内向但友善", "category": "personality", "confidence": 0.8}},
        {{"trait": "爱好", "value": "阅读和编程", "category": "interests", "confidence": 0.9}}
    ],
    "summary": "用户是一个内向但友善的人，喜欢阅读和编程。在沟通中倾向于深入讨论技术话题。"
}}

分析结果："""
        
        try:
            response = self.llm_client.chat([
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            
            # 解析 JSON 响应
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            # 存储特征
            if "traits" in result:
                for trait_data in result["traits"]:
                    if isinstance(trait_data, dict) and "trait" in trait_data:
                        self.add_trait(
                            user_id=user_id,
                            trait=trait_data["trait"],
                            value=trait_data.get("value", ""),
                            category=trait_data.get("category", "general"),
                            confidence=float(trait_data.get("confidence", 0.5)),
                            source="analysis"
                        )
            
            # 存储摘要
            if "summary" in result:
                self.update_summary(user_id, result["summary"])
            
            return result
            
        except Exception as e:
            print(f"人格分析失败: {e}")
            return None
    
    def get_persona_context(self, user_id: str) -> str:
        """获取适合 LLM 上下文的人格信息。
        
        Args:
            user_id: 用户ID
            
        Returns:
            格式化的人格信息
        """
        traits = self.get_traits(user_id)
        summary = self.get_latest_summary(user_id)
        
        if not traits and not summary:
            return ""
        
        context_parts = []
        
        if summary:
            context_parts.append(f"人格摘要: {summary['summary']}")
        
        if traits:
            # 按分类组织特征
            categories = {}
            for trait in traits:
                cat = trait.get("category", "general")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(f"{trait['trait']}: {trait['value']}")
            
            for cat, trait_list in categories.items():
                context_parts.append(f"{cat}: {'; '.join(trait_list[:5])}")
        
        return "\n".join(context_parts)
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户人格统计信息。"""
        with self._lock:
            trait_count = self._conn.execute(
                "SELECT COUNT(*) FROM traits WHERE user_id = ?",
                (user_id,)
            ).fetchone()[0]
            
            summary_count = self._conn.execute(
                "SELECT COUNT(*) FROM summaries WHERE user_id = ?",
                (user_id,)
            ).fetchone()[0]
            
            categories = self._conn.execute(
                "SELECT category, COUNT(*) FROM traits WHERE user_id = ? GROUP BY category",
                (user_id,)
            ).fetchall()
            
            return {
                "user_id": user_id,
                "total_traits": trait_count,
                "total_summaries": summary_count,
                "categories": {cat: count for cat, count in categories}
            }
    
    def close(self):
        """关闭数据库连接。"""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass