"""人格装配：System Prompt（角色卡）+ 五维记忆 + 会话历史 → 完整消息序列。"""

import base64
import time
from collections import deque
from datetime import datetime

from . import attachments
from .actions import split_reply
from .llm import LLMClient
from .metrics import logger, metrics


class Persona:
    def __init__(self, config, llm=None, note_dir=None, memory_manager=None,
                 character=None):
        """llm / note_dir / memory_manager 可注入：
        - 多用户服务端共享一个 LLMClient，避免重复建连；
        - note_dir 指定该用户的独立笔记目录（用户级隔离）；
        - memory_manager 五维记忆管理器实例（None = 不主动观测记忆）；
        - character 角色规格（core.characters.CharacterSpec）；None = 旧版单角色模式：
          读 config.system_prompt + 全局 visual_identity.card。"""
        if character is not None:
            self.character = character.name
            self.system_prompt = (character.prompt_text or "").strip() or \
                self._read_prompt_file(character.prompt_path)
        else:
            self.character = "default"
            self.system_prompt = self._read_prompt_file(config["system_prompt"])
        self.llm = llm if llm is not None else LLMClient(config)
        self.memory_manager = memory_manager
        # 附件（图片）落盘根目录 = 该用户 × 该角色的记忆命名空间
        self.note_dir = note_dir or (config.get("memory") or {}).get("note_file") or ""
        self.caption_images = bool((config.get("attachments") or {}).get("caption", True))
        self.visual_identity = None
        try:
            from .visual_identity import VisualIdentity

            # 按角色注入自视身份卡；空卡路径（如白鸥）→ 自识别关闭。
            # 旧版单角色模式原样传 config（visual_identity 段不变，兼容默认回退）。
            if character is not None:
                vis_cfg = dict(config.get("visual_identity") or {})
                vis_cfg["card"] = character.visual_identity or ""
                char_cfg = dict(config)
                char_cfg["visual_identity"] = vis_cfg
            else:
                char_cfg = config
            self.visual_identity = VisualIdentity(char_cfg, self.llm)
        except Exception as exc:
            # 自识别是可选项，加载失败不影响对话
            logger.warning("visual identity 装配失败: %s", exc)
            self.visual_identity = None
        self.history = []  # 短期记忆窗口
        self.last_attachments = {}  # 最近一轮的图片：{images:[{name,mime}], caption}
        self.recall_log = deque(maxlen=50)  # 每轮聊天的记忆召回 trace（供观测界面）

    @staticmethod
    def _read_prompt_file(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()

    def build_messages(self, user_input, self_context=""):
        system_content = self.system_prompt

        # 自我识别上下文（图片中「可能是我」）并入 system，尊重「首条 system」约定
        if self_context:
            system_content += "\n\n" + self_context

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
            # 兼容旧版：使用简单历史（带图消息补上图片内容说明）
            messages = [{"role": "system", "content": system_content}]
            for m in self.history[-10:]:  # 只保留最近10条
                messages.append({
                    "role": m["role"],
                    "content": attachments.with_caption(m.get("content", ""),
                                                        m.get("caption")),
                })
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
        """清空该用户的全部记忆：五维持久化 + 短期窗口 + 召回日志 + 图片。返回各维度删除计数。"""
        cleared = {}
        if self.memory_manager:
            cleared = self.memory_manager.clear(user_id="default")
        self.history = []
        self.recall_log.clear()
        cleared["images"] = attachments.clear(self.note_dir)  # 附件同属这轮记忆
        return cleared

    def reply(self, user_input, images=None):
        t0 = time.perf_counter()
        metrics.inc("chat_turns")
        # 图片自识别：VLM 裸描述 → 属性匹配 → 置信度 ≥ 阈值才注入「可能是我」上下文
        self_context = ""
        described = None
        if images and getattr(self, "visual_identity", None) is not None:
            self_context = self.visual_identity.analyze(images) or ""
            described = getattr(self.visual_identity, "last_described", None)
        messages = self.build_messages(user_input, self_context=self_context)
        # 仅在确有附件时传 images，兼容旧 fake/接口
        reply_text = self.llm.chat(messages, images=images) if images \
            else self.llm.chat(messages)

        # 图片：字节落盘（刷新后还在）+ 一句话说明（进记忆，她记得你给她看过什么）
        refs = attachments.save_images(self.note_dir, images) if images else []
        cap = ""
        if refs and self.caption_images:
            cap = attachments.caption(self.llm, images, described)
        self.last_attachments = ({"images": refs, "caption": cap} if refs else {})

        # 更新历史（图片引用与说明分列，原文保持干净）
        self.history.append(self._msg("user", user_input, refs, cap))
        self.history.append(self._msg("assistant", reply_text))
        if len(self.history) > 40:  # 短期窗口上限，防上下文膨胀
            self.history = self.history[-40:]

        # 使用五维记忆系统处理消息（后置观测：回复已生成，再写笔记）
        if self.memory_manager:
            mem_t0 = time.perf_counter()
            try:
                result = self.memory_manager.process_message(
                    user_id="default",
                    role="user",
                    content=user_input,
                    images=refs,
                    caption=cap,
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

    @staticmethod
    def _msg(role, content, images=None, caption=""):
        """一条消息：图片引用与内容说明只在有值时挂上（保持历史 JSON 干净）。"""
        msg = {"role": role, "content": content}
        if images:
            msg["images"] = images
        if caption:
            msg["caption"] = caption
        return msg

    def get_history(self, limit=50):
        """该用户 × 该角色的聊天记录（正序，供壳子回填界面）。

        启用观测笔记的角色读工作记忆落盘数据——跨设备、跨会话、跨进程重启都在；
        未启用的角色（memory=False）无落盘，退回进程内短期窗口（重启即失）。
        助手消息按原文重拆「正文 / 动作」；压缩产生的 system 摘要行只给模型看，不上界面。
        带图消息附 images（走 GET /image 取原图）与 caption（图片内容说明）。
        """
        if self.memory_manager:
            raw = self.memory_manager.recent.get_messages("default", limit=limit)
            persisted = True
        else:
            raw = self.history[-limit:]
            persisted = False
        messages = []
        for msg in raw:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            actions = []
            if role == "assistant":
                content, actions = split_reply(content)
            if not content and not actions and not msg.get("images"):
                continue  # 纯图片消息（无文字）也要留住
            out = {"role": role, "content": content, "actions": actions,
                   "time": msg.get("timestamp", "")}
            if msg.get("images"):
                out["images"] = msg["images"]
            if msg.get("caption"):
                out["caption"] = msg["caption"]
            messages.append(out)
        return {"messages": messages, "persisted": persisted}

    def reply_structured(self, user_input, images=None):
        """壳子友好版：返回 {reply(正文), actions(动作列表), raw(原文)}。

        历史与笔记仍以原文(raw)记录，保证上下文连续；展示层按需取字段。
        带图的轮次另附 attachments（落盘引用 + 图片内容说明）。
        """
        raw = self.reply(user_input, images=images)
        content, actions = split_reply(raw)
        result = {"reply": content, "actions": actions, "raw": raw}
        if self.last_attachments:
            result["attachments"] = self.last_attachments
        return result

    # ---------- 重新生成（换模型后就同一句话再问一次） ----------

    def _turns(self):
        """当前生效的短期窗口：有观测笔记 → 落盘工作记忆，否则进程内窗口。"""
        if self.memory_manager:
            return self.memory_manager.recent.get_messages("default")
        return self.history

    def _last_turn(self):
        """最后一轮 (user_msg, assistant_msg)；不是「先问后答」→ (None, None)。"""
        msgs = self._turns()
        if len(msgs) < 2:
            return None, None
        asst, user = msgs[-1], msgs[-2]
        if asst.get("role") != "assistant" or user.get("role") != "user":
            return None, None
        return user, asst

    def _drop_last_turn(self):
        """把最后一轮摘掉：先摘回复，再摘提问（顺序反了会摘错条）。"""
        if self.memory_manager:
            store = self.memory_manager.recent
            store.pop_last("default", "assistant")
            store.pop_last("default", "user")
        else:
            self.history.pop()
            self.history.pop()

    def _restore_last_turn(self, user_msg, asst_msg):
        """生成失败时原样放回这一轮（提问 + 旧回复），不能让用户丢了一句话。"""
        if self.memory_manager:
            store = self.memory_manager.recent
            extra = {}
            if user_msg.get("images"):
                extra["images"] = user_msg["images"]
            if user_msg.get("caption"):
                extra["caption"] = user_msg["caption"]
            store.add_message("default", "user", user_msg.get("content", ""), **extra)
            store.add_message("default", "assistant", asst_msg.get("content", ""))
        else:
            self.history.append(dict(user_msg))
            self.history.append(dict(asst_msg))

    def _recall_user_turn(self, user_msg):
        """提问放回工作记忆：不重复抽事实、不重复进时间索引（上一轮已经做过）。"""
        if self.memory_manager:
            extra = {}
            if user_msg.get("images"):
                extra["images"] = user_msg["images"]
            if user_msg.get("caption"):
                extra["caption"] = user_msg["caption"]
            self.memory_manager.recent.add_message(
                "default", "user", user_msg.get("content", ""), **extra)
        else:
            self.history.append(dict(user_msg))

    def _remember_assistant(self, reply_text):
        """新回复入账：与 reply() 同一条路径（工作记忆 + 时间索引）。"""
        if self.memory_manager:
            self.memory_manager.process_message(user_id="default", role="assistant",
                                                content=reply_text)
        else:
            self.history.append(self._msg("assistant", reply_text))

    def _load_ref_images(self, refs):
        """已落盘的图片引用 → base64：重新生成要把上一轮的图再喂给模型。"""
        out = []
        for ref in refs or []:
            if not isinstance(ref, dict):
                continue
            got = attachments.read_image(self.note_dir, ref.get("name") or "")
            if not got:
                continue
            data, mime = got
            out.append({"mime": mime or ref.get("mime") or "image/png",
                        "data": base64.b64encode(data).decode("ascii")})
        return out

    def regenerate(self):
        """丢掉最后一条助手回复，就同一句话重新回答（换模型后重试用）。

        只回滚工作记忆里那一轮；事实/反思/时间索引里旧回复的痕迹保留——
        跨维度级联删除代价远大于收益，也不影响下一轮上下文。
        图片沿用上一轮已落盘的引用（读回字节喂模型），不重复写盘、不重复描述。
        """
        user_msg, asst_msg = self._last_turn()
        if user_msg is None:
            raise ValueError("没有可重新生成的回复（需要先有一问一答）")
        user_input = str(user_msg.get("content") or "")
        images = self._load_ref_images(user_msg.get("images"))
        if not user_input and not images:
            raise ValueError("没有可重新生成的回复（上一轮没有内容）")

        self._drop_last_turn()
        t0 = time.perf_counter()
        metrics.inc("chat_turns")
        try:
            self_context = ""
            if images and self.visual_identity is not None:
                self_context = self.visual_identity.analyze(images) or ""
            messages = self.build_messages(user_input, self_context=self_context)
            reply_text = self.llm.chat(messages, images=images) if images \
                else self.llm.chat(messages)
        except Exception:
            self._restore_last_turn(user_msg, asst_msg)   # 失败就当没重试过
            raise

        self._recall_user_turn(user_msg)
        self._remember_assistant(reply_text)

        total_ms = (time.perf_counter() - t0) * 1000.0
        metrics.observe_ms("chat_turn", total_ms)
        metrics.emit(
            "chat_turn",
            ms=round(total_ms, 1),
            reply_chars=len(reply_text or ""),
            history=len(self._turns()),
            regenerated=True,
        )
        return reply_text

    def regenerate_structured(self):
        """regenerate() 的壳子友好版，字段同 reply_structured。

        不带 attachments：上一轮的图还在原消息上，别让壳子重复渲染。
        """
        raw = self.regenerate()
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
