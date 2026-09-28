"""用户管理：用户级隔离（壳子多设备 / 多用户）+ 角色级隔离。

一级维度 user_id，二维维度角色（character）：
- 同一用户的同一角色，会话历史与观测笔记跨设备、跨会话共享——她"记得"你；
- 不同用户之间完全隔离；同一用户的不同角色也互相隔离
  （笔记落盘 `notes/<user_id>/<角色>/`，默认角色沿用一级目录 `notes/<user_id>/`）；
- 角色能力可开关：记忆（观测笔记）、自视身份卡等，未启用的角色自动跳过低负载。

未指定 user_id 的请求落到 "default"；未指定 character 落到默认角色。
内存上限 max_users，超出按 FIFO 驱逐最旧（观测笔记已实时落盘，不丢数据）。
"""

import os
from collections import OrderedDict
from threading import RLock

from .characters import default_character, load_characters
from .persona import Persona
from .memory import MemoryManager
from .metrics import logger, metrics


class UserManager:
    def __init__(self, config, max_users=64):
        self.config = config
        self.max_users = max_users
        self.characters = load_characters(config)
        self.default_character = default_character(self.characters)
        self._entries = OrderedDict()
        self._lock = RLock()

    def list_characters(self):
        """角色注册表清单（供 GET /characters 观测 / 壳子选择器）。"""
        return [spec.to_dict() for spec in self.characters.values()]

    def list_users(self):
        """已知 user_id 清单（供 GET /users 与壳子的用户下拉框）。

        以落盘目录为准（notes/<user_id>/ 被驱逐出内存后仍在），并入内存中的活跃键；
        "default" 恒在列。目录不存在时退化为仅内存清单。
        """
        found = {"default"}
        note_root = self.config.get("memory", {}).get("note_file", "notes")
        try:
            for name in os.listdir(note_root):
                if os.path.isdir(os.path.join(note_root, name)):
                    found.add(name)
        except OSError:
            pass  # 还没建过笔记目录：只报内存里的
        with self._lock:
            found.update(user_id for user_id, _ in self._entries)
        return sorted(found)

    @staticmethod
    def _char_dir(char, default_char):
        """二级命名空间：默认角色沿用一级目录（兼容旧数据），其余落到 <角色>/。"""
        return "" if char == default_char else char

    def _make(self, user_id, character):
        spec = self.characters.get(character) or \
            self.characters[self.default_character]
        cfg = dict(self.config)
        # 记忆目录按 用户 → 角色 隔离
        memory = dict(cfg.get("memory", {}))
        note_root = memory.get("note_file", "notes")
        char_dir = self._char_dir(spec.name, self.default_character)
        ns = user_id if not char_dir else os.path.join(user_id, char_dir)
        memory["note_file"] = os.path.join(note_root, ns)
        cfg["memory"] = memory

        # 创建 LLM 客户端
        from .llm import LLMClient
        llm = LLMClient(cfg)

        # 观测笔记是配给的：未启用记忆能力的角色不建记忆管理器
        memory_manager = None
        if spec.memory:
            memory_dir = os.path.join(note_root, ns, "memory")
            memory_manager = MemoryManager(memory_dir=memory_dir, llm_client=llm)

        return Persona(cfg, llm=llm, memory_manager=memory_manager,
                       character=spec, note_dir=memory["note_file"])

    def entry(self, user_id="default", character=None):
        """取（或创建）某用户某角色的 (persona, lock) 组合。线程安全。

        lock 用于串行化同一键内的并发请求，避免历史/笔记写竞争。
        """
        user_id = (user_id or "default").strip() or "default"
        char = (character or "").strip() or self.default_character
        key = (user_id, char)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = {"persona": self._make(user_id, char), "lock": RLock()}
                self._entries[key] = entry
                metrics.inc("users_created")
                metrics.set_gauge("users_active", len(self._entries))
                metrics.emit("user_created", user_id=user_id, character=char)
                logger.info("user created id=%s char=%s active=%d",
                            user_id, char, len(self._entries))
                if len(self._entries) > self.max_users:
                    evicted, _ = self._entries.popitem(last=False)  # FIFO 驱逐最旧
                    metrics.inc("users_evicted")
                    metrics.emit("user_evicted", user_id=evicted[0], character=evicted[1])
            else:
                self._entries.move_to_end(key)
            return entry

    def get(self, user_id="default", character=None):
        """便捷取 Persona（不处理并发锁，仅读取/简单场景用）。"""
        return self.entry(user_id, character)["persona"]

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
