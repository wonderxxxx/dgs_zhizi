"""人格装配：System Prompt（角色卡）+ 五维记忆 + 会话历史 → 完整消息序列。"""

import base64
import time
from collections import deque
from datetime import datetime

from . import attachments
from . import websearch
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
        # 按需联网搜索：全局 web_search.enabled × 角色 search 开关，都开才装配。
        # 白鸥这类角色 search=false → 无工具说明、无搜索执行，人格不受影响。
        ws_cfg = config.get("web_search") or {}
        char_search = True if character is None else bool(
            getattr(character, "search", True))
        self.websearch = None
        self.search_prompt = ""
        if bool(ws_cfg.get("enabled", True)) and char_search:
            self.websearch = websearch.WebSearch(config)
            self.search_prompt = websearch.tool_instructions()
        self.last_searched = False  # 最近一轮是否真的发起过搜索（供壳子展示）
        self.history = []  # 短期记忆窗口
        self.last_attachments = {}  # 最近一轮的图片：{images:[{name,mime}], caption}
        self.recall_log = deque(maxlen=50)  # 每轮聊天的记忆召回 trace（供观测界面）

    @staticmethod
    def _read_prompt_file(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()

    def build_messages(self, user_input, self_context="", search_context="",
                       with_tools=True):
        """拼完整消息序列。

        search_context：本轮搜索结果块（两段式的第二遍才传；重建而非追加，
        工具信封文本永远不进消息序列）。
        with_tools：False = 撤掉工具说明（第二遍还硬要查时的最后兜底）。
        """
        system_content = self.system_prompt

        # 自我识别上下文（图片中「可能是我」）并入 system，尊重「首条 system」约定
        if self_context:
            system_content += "\n\n" + self_context

        # 使用五维记忆系统获取上下文（查询驱动召回，附带回溯 trace）
        history = None
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
            history = self.history[-10:]  # 兼容旧版：简单历史（最近10条）

        # 工具说明（每轮）与搜索结果（仅第二遍）都并入首条 system
        if with_tools and self.search_prompt:
            system_content += "\n\n" + self.search_prompt
        if search_context:
            system_content += "\n\n" + search_context

        messages = [{"role": "system", "content": system_content}]
        for m in history or []:
            # 带图消息补上图片内容说明
            messages.append({
                "role": m["role"],
                "content": attachments.with_caption(m.get("content", ""),
                                                    m.get("caption")),
            })
        messages.append({"role": "user", "content": user_input})
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

    def _chat_once(self, messages, images=None):
        """一次模型调用（仅在确有附件时传 images，兼容旧 fake/接口）。"""
        return self.llm.chat(messages, images=images) if images \
            else self.llm.chat(messages)

    def _generate(self, user_input, images=None, self_context="",
                  allow_search=True):
        """生成一轮回复（含按需搜索的两段式）。返回 (reply_text, searched)。

        第一遍：正常生成。没吐 <search> 信封 → 直接就是回复（绝大多数轮次
        只多一次正则判断，零额外开销）。
        吐了信封 → 执行搜索（额度内），**重建** messages（带结果块）再生成
        第二遍——第一遍的信封文本被丢弃，最终回复、历史、记忆永远看不到
        工具文本。第二遍还硬要查 → 撤掉工具说明再要一遍（封顶 3 次调用）。
        """
        self.last_searched = False
        active = bool(allow_search and self.websearch)
        reply = self._chat_once(
            self.build_messages(user_input, self_context=self_context), images)
        calls = websearch.parse_tool_calls(reply) if active else []
        if not calls:
            return websearch.strip_tool_calls(reply), False

        # 执行额度内的搜索；注册表外的信封（模型瞎编的工具名）忽略
        results, executed = [], 0
        for call in calls:
            if executed >= self.websearch.max_calls_per_turn:
                break
            if call["name"] not in websearch.TOOLS:
                continue
            executed += 1
            results.extend(self.websearch.search(call["args"]))
        if not executed:
            return websearch.strip_tool_calls(reply), False
        self.last_searched = True

        ctx = websearch.render_results(results)
        reply2 = self._chat_once(
            self.build_messages(user_input, self_context=self_context,
                                search_context=ctx),
            images)
        if websearch.parse_tool_calls(reply2):
            # 额度已用完：带结果再问一遍，同时撤掉工具说明（没得查就只能答）
            if not websearch.strip_tool_calls(reply2):
                reply2 = self._chat_once(
                    self.build_messages(user_input, self_context=self_context,
                                        search_context=ctx, with_tools=False),
                    images)
        return websearch.strip_tool_calls(reply2), True

    def reply(self, user_input, images=None, allow_search=True):
        t0 = time.perf_counter()
        metrics.inc("chat_turns")
        # 图片自识别：VLM 裸描述 → 属性匹配 → 置信度 ≥ 阈值才注入「可能是我」上下文
        self_context = ""
        described = None
        if images and getattr(self, "visual_identity", None) is not None:
            self_context = self.visual_identity.analyze(images) or ""
            described = getattr(self.visual_identity, "last_described", None)
        reply_text, _ = self._generate(user_input, images=images,
                                       self_context=self_context,
                                       allow_search=allow_search)

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

    def reply_structured(self, user_input, images=None, allow_search=True):
        """壳子友好版：返回 {reply(正文), actions(动作列表), raw(原文), searched}。

        历史与笔记仍以原文(raw)记录，保证上下文连续；展示层按需取字段。
        带图的轮次另附 attachments（落盘引用 + 图片内容说明）。
        searched：本轮是否真的发起了联网搜索（壳子可选择展示，默认不渲染）。
        """
        raw = self.reply(user_input, images=images, allow_search=allow_search)
        content, actions = split_reply(raw)
        result = {"reply": content, "actions": actions, "raw": raw,
                  "searched": bool(self.last_searched)}
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
        """把最后一轮摘掉：先摘回复，再摘提问（顺序反了会摘错条）。

        工作记忆（recent）与时间索引（timeindex）一起回滚——只摘 recent 会
        让被放弃的那句回复留在 BM25 召回源里，措辞和聊天记录对不上，反复
        重新生成还会成倍堆积。时间索引按内容删（hash 唯一，同内容只可能一条），
        前提是剩下的工作记忆里没有同样的内容——有就说明别的轮次还在用它。
        """
        if not self.memory_manager:
            self.history.pop()
            self.history.pop()
            return
        store = self.memory_manager.recent
        asst = store.pop_last("default", "assistant")
        store.pop_last("default", "user")
        if asst:
            self._forget_timeindex(asst.get("content", ""))

    def _forget_timeindex(self, content):
        """把一条消息从时间索引里抹掉；内容仍被别的轮次引用时保留。"""
        text = (content or "").strip()
        if not text:
            return
        store = self.memory_manager.recent
        for msg in store.get_messages("default"):
            if (msg.get("content") or "").strip() == text:
                return
        try:
            if self.memory_manager.timeindex.forget(
                    "default", text, category="conversation"):
                metrics.inc("timeindex_forgotten")
        except Exception as exc:   # 回滚不该拖垮重新生成
            logger.warning("时间索引回滚失败: %s", exc)

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
            content = asst_msg.get("content", "")
            store.add_message("default", "assistant", content)
            # _drop_last_turn 已把旧回复从时间索引抹了，这里补回来，
            # 否则恢复后的这一轮召不到（提问那条没被删，不用补）
            if (content or "").strip():
                try:
                    self.memory_manager.timeindex.add(
                        user_id="default", content=content,
                        category="conversation", importance=3)
                except Exception as exc:
                    logger.warning("时间索引恢复失败: %s", exc)
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

    def regenerate(self, allow_search=True):
        """丢掉最后一条助手回复，就同一句话重新回答（换模型后重试用）。

        回滚范围：工作记忆 + 时间索引里旧回复那一条都清掉，不留「已被放弃的
        回答」继续参与 BM25 召回。事实/反思/人格不动——它们只从你的提问抽取，
        而提问本身没变，留着是对的；跨维度级联删除的代价远大于收益。
        图片沿用上一轮已落盘的引用（读回字节喂模型），不重复写盘、不重复描述。
        按需搜索与 reply() 同一条路径（_generate 两段式）。
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
            reply_text, _ = self._generate(user_input, images=images,
                                           self_context=self_context,
                                           allow_search=allow_search)
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

    def regenerate_structured(self, allow_search=True):
        """regenerate() 的壳子友好版，字段同 reply_structured。

        不带 attachments：上一轮的图还在原消息上，别让壳子重复渲染。
        """
        raw = self.regenerate(allow_search=allow_search)
        content, actions = split_reply(raw)
        return {"reply": content, "actions": actions, "raw": raw,
                "searched": bool(self.last_searched)}

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
