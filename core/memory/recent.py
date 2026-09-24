"""工作记忆：管理短期对话历史，自动压缩和摘要。

简化版实现：
- 保存最近 N 条对话
- 当对话过多时，使用 LLM 生成摘要
- 支持按用户隔离
"""

import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional


class RecentMemory:
    """工作记忆管理器。"""
    
    def __init__(self, memory_dir: str, llm_client=None, max_items: int = 50):
        """
        Args:
            memory_dir: 记忆存储目录
            llm_client: LLM 客户端实例（用于生成摘要）
            max_items: 最大保存对话数
        """
        self.memory_dir = memory_dir
        self.llm_client = llm_client
        self.max_items = max_items
        self._lock = threading.RLock()
        
        # 确保目录存在
        os.makedirs(memory_dir, exist_ok=True)
        
        # 加载现有记忆
        self._memories: Dict[str, List[Dict]] = {}  # user_id -> [messages]
        self._load_all()
    
    def _get_path(self, user_id: str) -> str:
        """获取用户记忆文件路径。"""
        return os.path.join(self.memory_dir, f"recent_{user_id}.json")
    
    def _load_all(self):
        """加载所有用户的记忆。"""
        if not os.path.exists(self.memory_dir):
            return
            
        for filename in os.listdir(self.memory_dir):
            if filename.startswith("recent_") and filename.endswith(".json"):
                user_id = filename[6:-5]  # 去掉前缀和后缀
                filepath = os.path.join(self.memory_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        self._memories[user_id] = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self._memories[user_id] = []
    
    def _save(self, user_id: str):
        """保存用户记忆到文件。"""
        filepath = self._get_path(user_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._memories.get(user_id, []), f, ensure_ascii=False, indent=2)
    
    def add_message(self, user_id: str, role: str, content: str, **kwargs):
        """添加一条消息到工作记忆。
        
        Args:
            user_id: 用户ID
            role: 消息角色 (user/assistant/system)
            content: 消息内容
            **kwargs: 其他元数据
        """
        with self._lock:
            if user_id not in self._memories:
                self._memories[user_id] = []
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            
            self._memories[user_id].append(message)
            
            # 如果超过限制，尝试压缩
            if len(self._memories[user_id]) > self.max_items:
                self._compress(user_id)
            
            self._save(user_id)
    
    def get_messages(self, user_id: str, limit: Optional[int] = None) -> List[Dict]:
        """获取用户的对话历史。
        
        Args:
            user_id: 用户ID
            limit: 返回的最大消息数
            
        Returns:
            消息列表
        """
        with self._lock:
            messages = self._memories.get(user_id, [])
            if limit:
                return messages[-limit:]
            return messages.copy()
    
    def get_context(self, user_id: str, max_tokens: int = 2000) -> str:
        """获取适合 LLM 上下文的对话历史。
        
        Args:
            user_id: 用户ID
            max_tokens: 最大 token 数（估算）
            
        Returns:
            格式化的对话历史
        """
        messages = self.get_messages(user_id)
        if not messages:
            return ""
        
        # 简单估算：中文约 2 字符/token，英文约 4 字符/token
        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 3  # 粗略估算
        
        for msg in reversed(messages):
            role = msg["role"]
            content = msg["content"]
            part = f"{role}: {content}"
            
            if total_chars + len(part) > char_limit:
                break
            
            context_parts.insert(0, part)
            total_chars += len(part)
        
        return "\n".join(context_parts)
    
    def _compress(self, user_id: str):
        """压缩对话历史（使用 LLM 生成摘要）。"""
        if not self.llm_client or user_id not in self._memories:
            return
        
        messages = self._memories[user_id]
        if len(messages) <= self.max_items // 2:
            return
        
        # 取前半部分进行摘要
        half = len(messages) // 2
        to_summarize = messages[:half]
        remaining = messages[half:]
        
        # 生成摘要
        try:
            summary = self._generate_summary(to_summarize)
            # 用摘要替换前半部分
            compressed = [{"role": "system", "content": f"之前的对话摘要: {summary}"}]
            self._memories[user_id] = compressed + remaining
        except Exception:
            # 如果摘要失败，简单截断
            self._memories[user_id] = remaining
    
    def _generate_summary(self, messages: List[Dict]) -> str:
        """使用 LLM 生成对话摘要。"""
        if not self.llm_client:
            return "（对话历史已压缩）"
        
        # 构建摘要提示
        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages[-10:]  # 只取最近10条
        ])
        
        prompt = f"""请用中文简要总结以下对话的要点，保持简洁：

{conversation}

摘要:"""
        
        try:
            response = self.llm_client.chat([
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            return response.strip()
        except Exception:
            return "（对话摘要生成失败）"
    
    def clear(self, user_id: str):
        """清空用户的对话历史。"""
        with self._lock:
            n = len(self._memories.get(user_id, []))
            self._memories[user_id] = []
            self._save(user_id)
            return n
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户记忆统计信息。"""
        messages = self.get_messages(user_id)
        return {
            "user_id": user_id,
            "message_count": len(messages),
            "max_items": self.max_items,
            "last_message_time": messages[-1]["timestamp"] if messages else None
        }