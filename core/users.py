"""用户管理：用户级隔离（壳子多设备 / 多用户）。

每个 user_id 一个独立 Persona 实例：
- 同一用户的会话历史与观测笔记跨设备、跨会话共享——她"记得"你，不管你在哪台设备上；
- 不同用户之间完全隔离，互不可见。

未指定 user_id 的请求落到 "default"，保持向后兼容（相当于匿名用户）。
内存上限 max_users，超出按 FIFO 驱逐最旧（观测笔记已实时落盘，不丢数据）。
"""

import os
from collections import OrderedDict
from threading import RLock

from .persona import Persona
from .memory import MemoryManager
from .metrics import logger, metrics


class UserManager:
    def __init__(self, config, max_users=64):
        self.config = config
        self.max_users = max_users
        self._entries = OrderedDict()
        self._lock = RLock()

    def _make(self, user_id):
        cfg = dict(self.config)
        # 记忆目录按用户隔离：<note_root>/<user_id>
        memory = dict(cfg.get("memory", {}))
        note_root = memory.get("note_file", "notes")
        memory["note_file"] = os.path.join(note_root, user_id)
        cfg["memory"] = memory
        
        # 创建 LLM 客户端
        from .llm import LLMClient
        llm = LLMClient(cfg)
        
        # 创建五维记忆管理器
        memory_dir = os.path.join(note_root, user_id, "memory")
        memory_manager = MemoryManager(memory_dir=memory_dir, llm_client=llm)
        
        return Persona(cfg, llm=llm, memory_manager=memory_manager)

    def entry(self, user_id="default"):
        """取（或创建）某用户的 (persona, lock) 组合。线程安全。

        lock 用于串行化同一用户内的并发请求，避免历史/笔记写竞争。
        """
        user_id = (user_id or "default").strip() or "default"
        with self._lock:
            entry = self._entries.get(user_id)
            if entry is None:
                entry = {"persona": self._make(user_id), "lock": RLock()}
                self._entries[user_id] = entry
                metrics.inc("users_created")
                metrics.set_gauge("users_active", len(self._entries))
                metrics.emit("user_created", user_id=user_id)
                logger.info("user created id=%s active=%d", user_id, len(self._entries))
                if len(self._entries) > self.max_users:
                    evicted, _ = self._entries.popitem(last=False)  # FIFO 驱逐最旧
                    metrics.inc("users_evicted")
                    metrics.emit("user_evicted", user_id=evicted)
            else:
                self._entries.move_to_end(user_id)
            return entry

    def get(self, user_id="default"):
        """便捷取 Persona（不处理并发锁，仅读取/简单场景用）。"""
        return self.entry(user_id)["persona"]

    def total_notes(self):
        with self._lock:
            total = 0
            for entry in self._entries.values():
                persona = entry["persona"]
                if hasattr(persona, "memory_manager") and persona.memory_manager:
                    stats = persona.memory_manager.get_stats(user_id="default")
                    total += stats.get("facts", {}).get("total_facts", 0)
            return total

    def __len__(self):
        with self._lock:
            return len(self._entries)
