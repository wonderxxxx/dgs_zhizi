"""人格装配：System Prompt（角色卡）+ 五维记忆 + 会话历史 → 完整消息序列。"""

import time
from collections import deque
from datetime import datetime

from .actions import split_reply
from .llm import LLMClient
from .metrics import logger, metrics


class Persona:
    def __init__(self, config, llm=None, note_dir=None, memory_manager=None):
        """llm / note_dir / memory_manager 可注入：
        - 多用户服务端共享一个 LLMClient，避免重复建连；
        - note_dir 指定该用户的独立笔记目录（用户级隔离）；
        - memory_manager 五维记忆管理器实例。"""
        with open(config["system_prompt"], encoding="utf-8") as fh:
            self.system_prompt = fh.read().strip()
        self.llm = llm if llm is not None else LLMClient(config)
        self.memory_manager = memory_manager
        self.history = []  # 短期记忆窗口
        self.recall_log = deque(maxlen=50)  # 每轮聊天的记忆召回 trace（供观测界面）

    def build_messages(self, user_input):
        system_content = self.system_prompt

        # 使用五维记忆系统获取上下文（查询驱动召回，附带回溯 trace）
        if self.memory_manager:
            context, trace = self.memory_manager.get_context_traced(
                user_id="default", max_tokens=2000, query=user_input
            )
            if context:
                # 记忆并入首条 system 消息：多数 chat template 只允许开头一条 system
                system_content += (
                    "\n\n【你记得的事：仅在相关内容自然浮现时调用，"
                    "不刻意提及、不逐条罗列】\n" + context
                )
            self._record_recall(user_input, context, trace)
        else:
            # 兼容旧版：使用简单历史
            messages = [{"role": "system", "content": system_content}]
            messages.extend(self.history[-10:])  # 只保留最近10条
            messages.append({"role": "user", "content": user_input})
            return messages

        messages = [{"role": "system", "content": system_content},
                    {"role": "user", "content": user_input}]
        return messages

    def _record_recall(self, query, context, trace):
        """把本轮对话的记忆召回过程记录下来（供 /memory 观测）。"""
        if not trace:
            return
        dims = []
        items = []
        for d in trace.get("dimensions", []):
            dim_items = d.get("items", [])
            dims.append({"dim": d.get("dim", ""), "label": d.get("label", ""),
                         "reason": d.get("reason", ""), "n": len(dim_items)})
            items.append({"dim": d.get("dim", ""), "label": d.get("label", ""),
                          "items": dim_items})
        self.recall_log.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "context_chars": len(context or ""),
            "dimensions": dims,
            "detail": items,
        })
        metrics.emit("recall", query=query,
                     dims=[d["dim"] for d in dims],
                     items=sum(d["n"] for d in dims))

    def get_recall_log(self, limit=50):
        """最近 N 轮聊天的召回 trace（新→旧）。"""
        return list(self.recall_log)[-limit:][::-1]

    def clear_memory(self):
        """清空该用户的全部记忆：五维持久化 + 短期窗口 + 召回日志。返回各维度删除计数。"""
        cleared = {}
        if self.memory_manager:
            cleared = self.memory_manager.clear(user_id="default")
        self.history = []
        self.recall_log.clear()
        return cleared

    def reply(self, user_input, images=None):
        t0 = time.perf_counter()
        metrics.inc("chat_turns")
        messages = self.build_messages(user_input)
        # 仅在确有附件时传 images，兼容旧 fake/接口
        reply_text = self.llm.chat(messages, images=images) if images \
            else self.llm.chat(messages)

        # 更新历史
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply_text})
        if len(self.history) > 40:  # 短期窗口上限，防上下文膨胀
            self.history = self.history[-40:]

        # 使用五维记忆系统处理消息（后置观测：回复已生成，再写笔记）
        if self.memory_manager:
            mem_t0 = time.perf_counter()
            try:
                result = self.memory_manager.process_message(
                    user_id="default",
                    role="user",
                    content=user_input
                )
                # 也可以处理助手回复
                self.memory_manager.process_message(
                    user_id="default",
                    role="assistant",
                    content=reply_text
                )
                mem_ms = (time.perf_counter() - mem_t0) * 1000.0
                metrics.observe_ms("memory_process", mem_ms)
                facts = len(result.get("facts") or [])
                if facts:
                    metrics.inc("facts_extracted", facts)
            except Exception as exc:
                metrics.inc("memory_errors")
                metrics.emit("memory_error", error=str(exc))
                logger.warning("memory process failed: %s", exc)

        total_ms = (time.perf_counter() - t0) * 1000.0
        metrics.observe_ms("chat_turn", total_ms)
        metrics.emit(
            "chat_turn",
            ms=round(total_ms, 1),
            reply_chars=len(reply_text or ""),
            history=len(self.history),
        )
        return reply_text

    def reply_structured(self, user_input, images=None):
        """壳子友好版：返回 {reply(正文), actions(动作列表), raw(原文)}。

        历史与笔记仍以原文(raw)记录，保证上下文连续；展示层按需取字段。
        """
        raw = self.reply(user_input, images=images)
        content, actions = split_reply(raw)
        return {"reply": content, "actions": actions, "raw": raw}

    def get_memory_stats(self):
        """获取记忆统计信息。"""
        if self.memory_manager:
            return self.memory_manager.get_stats(user_id="default")
        return {}

    def search_memory(self, query, top_k=10):
        """搜索记忆。"""
        if self.memory_manager:
            return self.memory_manager.search(user_id="default", query=query, top_k=top_k)
        return {}
