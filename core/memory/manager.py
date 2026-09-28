"""统一记忆管理器：协调五个维度的记忆系统。

提供统一的接口来管理：
- 工作记忆 (Recent)
- 近期记忆 (TimeIndexed)
- 事实记忆 (Facts)
- 反思记忆 (Reflection)
- 人格记忆 (Persona)
"""

import os
from typing import List, Dict, Any, Optional

from .recent import RecentMemory
from .timeindex import TimeIndexedMemory
from .facts import FactMemory
from .reflection import ReflectionMemory
from .persona import PersonaMemory
from ..attachments import with_caption


class MemoryManager:
    """统一记忆管理器。"""
    
    def __init__(self, memory_dir: str, llm_client=None):
        """
        Args:
            memory_dir: 记忆存储根目录
            llm_client: LLM 客户端实例
        """
        self.memory_dir = memory_dir
        self.llm_client = llm_client
        
        # 确保目录存在
        os.makedirs(memory_dir, exist_ok=True)
        
        # 初始化五个维度的记忆
        self.recent = RecentMemory(
            memory_dir=os.path.join(memory_dir, "recent"),
            llm_client=llm_client
        )
        
        self.timeindex = TimeIndexedMemory(
            memory_dir=os.path.join(memory_dir, "timeindex")
        )
        
        self.facts = FactMemory(
            memory_dir=os.path.join(memory_dir, "facts"),
            llm_client=llm_client
        )
        
        self.reflection = ReflectionMemory(
            memory_dir=os.path.join(memory_dir, "reflection"),
            llm_client=llm_client
        )
        
        self.persona = PersonaMemory(
            memory_dir=os.path.join(memory_dir, "persona"),
            llm_client=llm_client
        )
    
    def process_message(self, user_id: str, role: str, content: str,
                        images: List[Dict[str, str]] = None,
                        caption: str = None) -> Dict[str, Any]:
        """处理一条消息，更新所有相关记忆。
        
        Args:
            user_id: 用户ID
            role: 消息角色 (user/assistant)
            content: 消息内容
            images: 图片引用 [{name, mime}]（只存工作记忆，取图走 /image）
            caption: 图片内容说明：工作记忆单列一个字段给界面看，
                     检索与事实提取则用「原文 + 说明」的文本
            
        Returns:
            处理结果摘要
        """
        result = {
            "recent": None,
            "facts": [],
            "reflections": [],
            "persona": None
        }
        
        # 1. 添加到工作记忆（图片引用与说明分列，原文保持干净）
        extra = {}
        if images:
            extra["images"] = images
        if caption:
            extra["caption"] = caption
        self.recent.add_message(user_id, role, content, **extra)
        result["recent"] = True
        
        # 2. 添加到近期记忆（带图片说明的文本，BM25 检索才搜得到"那张图"）
        recall_text = with_caption(content, caption)
        self.timeindex.add(
            user_id=user_id,
            content=recall_text,
            category="conversation",
            importance=5 if role == "user" else 3
        )
        
        # 3. 如果是用户消息，尝试提取事实
        if role == "user":
            # 获取最近的对话上下文
            recent_messages = self.recent.get_messages(user_id, limit=5)
            conversation = "\n".join([
                f"{msg['role']}: {with_caption(msg.get('content', ''), msg.get('caption'))}"
                for msg in recent_messages
            ])
            
            # 提取事实
            facts = self.facts.extract_facts(user_id, conversation)
            result["facts"] = facts
            
            # 4. 如果有足够的事实，生成反思
            if facts:
                all_facts = self.facts.get_facts(user_id, limit=20)
                reflections = self.reflection.generate_reflection(
                    user_id, all_facts, conversation
                )
                result["reflections"] = reflections
            
            # 5. 定期更新人格分析（每10条消息）
            messages = self.recent.get_messages(user_id)
            if len(messages) % 10 == 0 and len(messages) > 0:
                all_facts = self.facts.get_facts(user_id, limit=30)
                if all_facts:
                    persona_result = self.persona.analyze_persona(user_id, all_facts)
                    result["persona"] = persona_result
        
        return result
    
    def recall_trace(self, user_id: str, query: str = None, top_k: int = 5) -> Dict[str, Any]:
        """召回过程明细：每个记忆维度检索发生在何时、命中什么、为什么命中。

        供观测界面 /memory/search 与 /memory（真实聊天召回日志）使用。

        Returns:
            {
                "query": 查询原句,
                "tokens": 查询分词（BM25 用）,
                "dimensions": [
                    {dim, label, reason, items: [...每项带 matched_terms / score / confidence...]},
                    ...
                ]
            }
        """
        query = (query or "").strip()
        trace: Dict[str, Any] = {"query": query, "tokens": [], "dimensions": []}

        try:
            from .timeindex import _query_tokens
            trace["tokens"] = sorted(_query_tokens(query))
        except Exception:
            pass

        # 1. 工作记忆（最近对话）：按时间窗口注入，不做语义评分
        recent_msgs = self.recent.get_messages(user_id, limit=30)
        trace["dimensions"].append({
            "dim": "recent",
            "label": "工作记忆",
            "reason": "最近对话按时间窗口注入，不做语义评分",
            "items": [
                {"role": m.get("role", ""),
                 "content": with_caption(m.get("content", ""), m.get("caption")),
                 "caption": m.get("caption", ""),
                 "images": m.get("images") or [],
                 "time": m.get("timestamp", ""), "matched_terms": []}
                for m in recent_msgs
            ],
        })

        # 2. 人格记忆：恒定注入（用户是谁、怎么互动）
        persona_items = []
        for t in self.persona.get_traits(user_id):
            persona_items.append({
                "kind": "trait", "trait": t.get("trait", ""), "value": t.get("value", ""),
                "category": t.get("category", ""), "confidence": t.get("confidence"),
            })
        summary = self.persona.get_latest_summary(user_id)
        if summary:
            persona_items.append({
                "kind": "summary", "trait": "人格摘要", "value": summary.get("summary", ""),
                "confidence": 1.0,
            })
        trace["dimensions"].append({
            "dim": "persona",
            "label": "人格记忆",
            "reason": "人格画像恒定注入，不做检索",
            "items": persona_items,
        })

        # 3. 近期记忆：BM25 打分（中文 bigram / 英文·数字整词）+ importance 微调
        if query:
            time_items = self.timeindex.search_traced(user_id, query, top_k=top_k)
        else:
            time_items = [
                {**d, "score": None, "matched_terms": []}
                for d in self.timeindex.all(user_id, limit=top_k)
            ]
        trace["dimensions"].append({
            "dim": "timeindex",
            "label": "近期记忆",
            "reason": "BM25 打分：中文 bigram / 英文·数字整词，重要度微调",
            "items": [
                {"content": i.get("content", ""), "time": i.get("time", ""),
                 "importance": i.get("importance"), "score": i.get("score"),
                 "matched_terms": i.get("matched_terms", [])}
                for i in time_items
            ],
        })

        # 4. 事实记忆：LIKE 子串匹配，按置信度排序
        if query:
            facts = self.facts.search_facts(user_id, query, top_k=top_k)
        else:
            facts = self.facts.get_facts(user_id, limit=top_k)
        trace["dimensions"].append({
            "dim": "facts",
            "label": "事实记忆",
            "reason": "LIKE 子串匹配，按置信度排序",
            "items": [
                {"fact": f.get("fact", ""), "confidence": f.get("confidence"),
                 "source": (f.get("source") or "")[:80],
                 "matched_terms": [query] if query and query in f.get("fact", "") else []}
                for f in facts
            ],
        })

        # 5. 反思记忆：关键词命中已确认洞察
        confirmed = self.reflection.get_confirmed_reflections(user_id, limit=50)
        if query:
            refs = [r for r in confirmed if query in r.get("reflection", "")]
        else:
            refs = confirmed[:top_k]
        trace["dimensions"].append({
            "dim": "reflection",
            "label": "反思记忆",
            "reason": "关键词命中已确认反思",
            "items": [
                {"reflection": r.get("reflection", ""), "insight": r.get("insight", ""),
                 "confidence": r.get("confidence"),
                 "matched_terms": [query] if query and query in r.get("reflection", "") else []}
                for r in refs[:top_k]
            ],
        })

        return trace

    def get_context_traced(self, user_id: str, max_tokens: int = 2000,
                           query: str = None, top_k: int = 5) -> tuple:
        """获取适合 LLM 的完整上下文，并返回本次召回过程 trace。

        query 非空时为查询驱动召回（事实/反思按相关性检索）；
        无查询时退回「最近 + 人格 + 近期记忆」，仍产出 trace 供观测。
        """
        trace = self.recall_trace(user_id=user_id, query=query, top_k=top_k)
        context_parts = []

        # 1. 工作记忆（最近对话）
        recent_context = self.recent.get_context(user_id, max_tokens=max(max_tokens // 3, 240))
        if recent_context:
            context_parts.append(f"最近对话:\n{recent_context}")

        # 2. 人格信息
        persona_context = self.persona.get_persona_context(user_id)
        if persona_context:
            context_parts.append(f"用户人格:\n{persona_context}")

        by_dim = {d["dim"]: d for d in trace["dimensions"]}

        # 3. 相关事实（仅注入命中项）
        facts_items = [i for i in by_dim.get("facts", {}).get("items", []) if i.get("matched_terms")]
        if facts_items:
            context_parts.append(
                "相关事实:\n" + "\n".join(f"- {i['fact']}" for i in facts_items)
            )

        # 4. 洞察（命中项；无查询时注入最新几条）
        ref_items = by_dim.get("reflection", {}).get("items", [])
        if query:
            ref_items = [i for i in ref_items if i.get("matched_terms")]
        if ref_items:
            context_parts.append(
                "洞察:\n" + "\n".join(
                    f"- {i['reflection']}" + (f"（{i['insight']}）" if i.get("insight") else "")
                    for i in ref_items
                )
            )

        # 5. 相关往事（近期记忆命中，供回溯上下文）
        time_items = [i for i in by_dim.get("timeindex", {}).get("items", []) if i.get("matched_terms")]
        if time_items and query:
            context_parts.append(
                "相关往事:\n" + "\n".join(f"- {i['content']}" for i in time_items)
            )

        return "\n\n".join(context_parts), trace

    def get_context(self, user_id: str, max_tokens: int = 3000) -> str:
        """获取适合 LLM 的完整上下文（供旧调用方）。"""
        context, _ = self.get_context_traced(user_id, max_tokens=max_tokens, query=None)
        return context
    
    def search(self, user_id: str, query: str, top_k: int = 10) -> Dict[str, List]:
        """跨维度搜索记忆。
        
        Args:
            user_id: 用户ID
            query: 搜索查询
            top_k: 每个维度返回的最大结果数
            
        Returns:
            各维度的搜索结果
        """
        results = {
            "timeindex": self.timeindex.search(user_id, query, top_k=top_k),
            "facts": self.facts.search_facts(user_id, query, top_k=top_k),
            "reflections": [
                r for r in self.reflection.get_reflections(user_id, limit=top_k)
                if query.lower() in r.get("reflection", "").lower()
            ]
        }
        
        return results
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户记忆统计信息。"""
        return {
            "recent": self.recent.get_stats(user_id),
            "timeindex": self.timeindex.get_stats(user_id),
            "facts": self.facts.get_stats(user_id),
            "reflections": self.reflection.get_stats(user_id),
            "persona": self.persona.get_stats(user_id)
        }
    
    def clear(self, user_id: str):
        """清空该用户的全部记忆（五维）。返回各维度删除计数。"""
        return {
            "recent": self.recent.clear(user_id),
            "timeindex": self.timeindex.clear(user_id),
            "facts": self.facts.clear(user_id),
            "reflections": self.reflection.clear(user_id),
            "persona": self.persona.clear(user_id),
        }
    
    def close(self):
        """关闭所有记忆存储。"""
        self.recent.close() if hasattr(self.recent, 'close') else None
        self.timeindex.close()
        self.facts.close()
        self.reflection.close()
        self.persona.close()