#!/usr/bin/env python3
"""千夜智子 · HTTP JSON API（壳子对接层）

供 Flutter / Swift / Web 壳子调用的轻量服务，零新依赖（标准库实现）。
接口不绑定任何前端框架，壳子只负责 UI，人格与记忆全在这里。

运行：
    uv run python api.py --port 8765

接口：
    POST /chat      {"message": "你好", "user_id": "小明", "character": "白鸥", "hide_actions": false}
                    → {"reply": "正文", "actions": ["动作1"], "raw": "原文",
                       "user_id": "小明", "character": "白鸥"}
                    character 缺省 = 默认角色（config.characters 的 default）。
                    一级用户 / 二级角色：不同角色独立人格与记忆命名空间。
    GET  /characters → {"default": "千夜智子", "characters": [{name, prompt, visual_identity, memory, default}]}
    GET  /users      → {"default": "default", "users": ["default", "alice", ...]}
                    已知 user_id 清单（notes 落盘目录 ∪ 内存活跃键），供壳子做用户下拉框
    GET  /history    → {"user_id","character","persisted","messages":[{role,content,actions,time}]}
                    该 user_id × character 的聊天记录（正序），供壳子打开会话时回填
                    带图消息附 images:[{name,mime}]（走 GET /image）与 caption（图片内容说明）
    GET  /image      → ?user_id=&character=&name= 回读历史里的原图字节（鉴权同 /history）
    GET  /health    → {"status": "ok", "model": "...", "vision": bool, "users": N, "notes": M}
    GET  /model     → 当前模型 + 可切换模型清单（含是否多模态）
    POST /model     → {"model": "<路径/名称>"} 切换现行模型
    GET  /metrics   → 进程内指标（计数器/延迟/事件）
    GET  /events    → SSE 实时事件流（监控用）
    GET  /chat      → 聊天单页（static/chat.html）
    GET  /memory    → 记忆观察单页（static/memory.html）
    GET  /memory/api→ user_id + character 维度：五维记忆全量 + 最近召回 trace
    POST /memory/search → {"user_id","character","query","top_k"} → 该查询的召回过程明细
    POST /memory/clear   → {"user_id","character"} → 清空该用户该角色全部记忆（不可恢复）
    GET  /dashboard → 单页实时监控（observability.dashboard: true 时）

鉴权（可选）：config.yaml 的 server.api_key 非空时，
/chat 需携带请求头：Authorization: Bearer <api_key>
"""

import argparse
import json
import os
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import yaml

from core import attachments
from core import llm as llm_mod
from core.metrics import logger, metrics, setup_logging
from core.users import UserManager

DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "static", "dashboard.html")
CHAT_PATH = os.path.join(os.path.dirname(__file__), "static", "chat.html")
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "static", "memory.html")


def dump_memory(persona):
    """某用户五维记忆全量快照（供 GET /memory）。"""
    mm = persona.memory_manager
    if mm is None:
        return {}
    uid = "default"  # 每用户的 Persona 内统一使用 default 命名空间
    return {
        "recent": mm.recent.get_messages(uid),
        "timeindex": mm.timeindex.all(uid),
        "facts": mm.facts.get_facts(uid, limit=500),
        "reflections": mm.reflection.get_reflections(uid, limit=500),
        "persona": {
            "traits": mm.persona.get_traits(uid),
            "summaries": mm.persona.get_summary_history(uid, limit=20),
        },
        "stats": {
            "recent": mm.recent.get_stats(uid),
            "timeindex": mm.timeindex.get_stats(uid),
            "facts": mm.facts.get_stats(uid),
            "reflections": mm.reflection.get_stats(uid),
            "persona": mm.persona.get_stats(uid),
        },
    }


def persona_identity(persona):
    """某用户的「自视身份」快照（供 GET /identity 观测）。"""
    vi = getattr(persona, "visual_identity", None)
    if vi is None:
        return {"enabled": False}
    return {
        "enabled": bool(vi.enabled),
        "threshold": vi.threshold,
        "identity": vi.get_card(),
    }


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_handler(users, api_key="", dashboard_enabled=True):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ZhiziAPI/0.4"

        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()
            self.wfile.write(body)

        def _send_raw(self, code, content_type, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            if not api_key:
                return True
            auth = self.headers.get("Authorization", "")
            return auth == f"Bearer {api_key}"

        def _query_args(self):
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            return {k: (v[-1] if v else "") for k, v in parse_qs(qs).items()}

        @staticmethod
        def _parse_attachments(body):
            """解析 attachments: [{"mime","data"}] → [{"mime","data"}]（校验 + 总量限制）。"""
            atts = body.get("attachments") or body.get("images") or []
            if not isinstance(atts, list):
                return []
            out = []
            total = 0
            for att in atts[:4]:  # 单轮最多 4 张
                if not isinstance(att, dict):
                    continue
                mime = str(att.get("mime") or "").lower()
                data = str(att.get("data") or "")
                if mime.startswith("image/") and data:
                    total += len(data)
                    if total > 30_000_000:  # ~22MB 上限
                        logger.warning("附件总量超限（base64 已 %d 字符），丢弃剩余图片", total)
                        break
                    out.append({"mime": mime, "data": data})
            return out

        def do_OPTIONS(self):
            self._send(204, {})

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/health":
                llm = users.get("default").llm
                self._send(200, {
                    "status": "ok",
                    "model": llm_mod.get_active_model() or llm.model,
                    "vision": llm_mod.active_vision(),
                    "users": len(users),
                    "notes": users.total_notes(),
                })
            elif path == "/model":
                self._model_list()
            elif path == "/metrics":
                snap = metrics.snapshot()
                snap["model"] = llm_mod.get_active_model() or users.get("default").llm.model
                snap["users"] = len(users)
                self._send(200, snap)
            elif path == "/events":
                self._sse()
            elif path == "/chat":
                try:
                    with open(CHAT_PATH, "rb") as fh:
                        body = fh.read()
                    self._send_raw(200, "text/html; charset=utf-8", body)
                except FileNotFoundError:
                    self._send(404, {"error": "chat.html not found"})
            elif path == "/characters":
                self._send(200, {
                    "default": users.default_character,
                    "characters": users.list_characters(),
                })
            elif path == "/users":
                self._send(200, {
                    "default": "default",
                    "users": users.list_users(),
                })
            elif path == "/history":
                self._history()
            elif path == "/image":
                self._image()
            elif path == "/memory":
                try:
                    with open(MEMORY_PATH, "rb") as fh:
                        body = fh.read()
                    self._send_raw(200, "text/html; charset=utf-8", body)
                except FileNotFoundError:
                    self._send(404, {"error": "memory.html not found"})
            elif path == "/memory/api":
                args = self._query_args()
                user_id = (args.get("user_id") or "default").strip() or "default"
                character = (args.get("character") or "").strip() or None
                persona = users.entry(user_id, character)["persona"]
                data = {
                    "user_id": user_id,
                    "character": persona.character,
                    "recalls": persona.get_recall_log(),
                    "dimensions": dump_memory(persona),
                }
                self._send(200, data)
            elif path == "/identity":
                args = self._query_args()
                user_id = (args.get("user_id") or "default").strip() or "default"
                character = (args.get("character") or "").strip() or None
                self._send(200, persona_identity(users.get(user_id, character)))
            elif path == "/dashboard":
                if not dashboard_enabled:
                    self._send(404, {"error": "dashboard disabled"})
                    return
                try:
                    with open(DASHBOARD_PATH, "rb") as fh:
                        body = fh.read()
                    self._send_raw(200, "text/html; charset=utf-8", body)
                except FileNotFoundError:
                    self._send(404, {"error": "dashboard.html not found"})
            else:
                self._send(404, {"error": "not found"})

        def _history(self):
            """GET /history：某用户某角色的聊天记录（供壳子回填界面）。

            同一 (user_id, character) 打开会话即拉到此前全部往来——工作记忆落盘，
            跨设备、跨会话、跨进程重启都在。鉴权同 /chat（含私密对话，不放行裸读）。
            """
            if not self._authorized():
                metrics.inc("auth_failures")
                self._send(401, {"error": "unauthorized"})
                return
            args = self._query_args()
            user_id = (args.get("user_id") or "default").strip() or "default"
            character = (args.get("character") or "").strip() or None
            try:
                limit = int(args.get("limit") or 100)
            except (TypeError, ValueError):
                limit = 100
            limit = max(1, min(limit, 500))
            try:
                entry = users.entry(user_id, character)
                with entry["lock"]:  # 与 /chat 同键串行，避免读到半轮对话
                    data = entry["persona"].get_history(limit=limit)
            except Exception as exc:
                logger.warning("history error: %s", exc)
                self._send(500, {"error": str(exc)})
                return
            data["user_id"] = user_id
            data["character"] = entry["persona"].character
            self._send(200, data)

        def _image(self):
            """GET /image：回读该用户该角色历史里的某张原图（?name=）。

            名字只认 [A-Za-z0-9._-]，解析后必须落在该命名空间的 images/ 内
            （跨用户、跨角色、路径穿越都取不到）。鉴权同 /history。
            """
            if not self._authorized():
                metrics.inc("auth_failures")
                self._send(401, {"error": "unauthorized"})
                return
            args = self._query_args()
            user_id = (args.get("user_id") or "default").strip() or "default"
            character = (args.get("character") or "").strip() or None
            name = args.get("name") or ""
            persona = users.get(user_id, character)
            blob = attachments.read_image(persona.note_dir, name)
            if blob is None:
                self._send(404, {"error": "image not found"})
                return
            data, mime = blob
            self._send_raw(200, mime, data)

        def _sse(self):
            """SSE：先推最近事件，再持续转发新事件；断开即停。"""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q: queue.Queue = queue.Queue(maxsize=256)

            def on_event(ev):
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass

            metrics.add_listener(on_event)
            try:
                # 回放最近事件
                for ev in list(metrics.snapshot().get("recent_events", []))[-50:]:
                    data = json.dumps(ev, ensure_ascii=False, default=str)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        ev = q.get(timeout=15.0)
                        data = json.dumps(ev, ensure_ascii=False, default=str)
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                metrics.remove_listener(on_event)

        def _model_list(self):
            """GET /model：现行模型 + 可切换的模型清单（含各自是否多模态）。"""
            reg = llm_mod.get_model_registry()
            self._send(200, {
                "current": llm_mod.get_active_model(),
                "vision": llm_mod.active_vision(),
                "models": [
                    {"path": m, "name": os.path.basename(m),
                     "vision": llm_mod.detect_multimodal(m, "auto")}
                    for m in reg
                ],
            })

        def _model_switch(self):
            """POST /model：切换现行模型（mlx 场景释放旧引擎，下次请求惰性重载）。"""
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length).decode("utf-8")
                body = json.loads(raw) if raw else {}
                target = llm_mod.resolve_model_switch(str(body.get("model", "")))
                llm_mod.set_active_model(target)
                vision = llm_mod.active_vision()
                metrics.emit("model_switch", model=target, vision=vision)
                logger.info("model switched to=%s vision=%s", target, vision)
                self._send(200, {"status": "ok", "model": target, "vision": vision})
            except ValueError as exc:
                metrics.inc("model_switch_errors")
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                metrics.inc("model_switch_errors")
                logger.warning("model switch error: %s", exc)
                self._send(500, {"error": str(exc)})

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/model":
                self._model_switch()
                return
            if path == "/memory/search":
                self._memory_search("default")
                return
            if path == "/memory/clear":
                self._memory_clear()
                return
            if path not in ("/chat", "/regenerate"):
                self._send(404, {"error": "not found"})
                return
            if not self._authorized():
                metrics.inc("auth_failures")
                self._send(401, {"error": "unauthorized"})
                return

            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length).decode("utf-8")
                body = json.loads(raw) if raw else {}
                user_id = str(body.get("user_id", "default")).strip() or "default"
                character = (str(body.get("character", "")).strip() or None)
                metrics.inc("http_chat_requests")
                entry = users.entry(user_id, character)
                persona = entry["persona"]
                images = []
                with entry["lock"]:  # 同一键内串行，避免历史/笔记写竞争
                    if path == "/regenerate":
                        # 换模型后就同一句话再要一个回答（丢掉旧回复）
                        result = persona.regenerate_structured()
                    else:
                        message = str(body.get("message", "")).strip()
                        images = self._parse_attachments(body)
                        if not message and not images:
                            metrics.inc("chat_bad_requests")
                            self._send(400, {"error": "message 不能为空"})
                            return
                        result = persona.reply_structured(message, images=images)
                # reply 恒为正文；hide_actions=true 时不返回动作列表（省流量）
                if body.get("hide_actions", False):
                    result.pop("actions", None)
                result["user_id"] = user_id
                result["character"] = persona.character
                result["image_acked"] = bool(images) and getattr(
                    persona.llm, "multimodal", False
                )
                metrics.inc("http_chat_ok")
                self._send(200, result)
            except json.JSONDecodeError:
                metrics.inc("chat_bad_requests")
                self._send(400, {"error": "JSON 解析失败"})
            except ValueError as exc:   # 没有可重生成的轮次等用户侧问题
                metrics.inc("chat_bad_requests")
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                metrics.inc("http_chat_errors")
                metrics.emit("http_chat_error", error=str(exc))
                logger.warning("chat error: %s", exc)
                self._send(500, {"error": str(exc)})

        def _memory_search(self, _unused=""):
            """POST /memory/search：观测某条查询会召回什么记忆（user × character）。"""
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length).decode("utf-8")
                body = json.loads(raw) if raw else {}
                user_id = str(body.get("user_id", "default")).strip() or "default"
                character = (str(body.get("character", "")).strip() or None)
                query = str(body.get("query", "")).strip()
                top_k = int(body.get("top_k", 5) or 5)
                if not query:
                    self._send(400, {"error": "query 不能为空"})
                    return
                persona = users.entry(user_id, character)["persona"]
                if persona.memory_manager is None:
                    self._send(200, {"query": query, "user_id": user_id,
                                     "character": persona.character,
                                     "error": "该角色未启用观测笔记（智子专有能力）"})
                    return
                trace = persona.memory_manager.recall_trace(
                    user_id="default", query=query, top_k=top_k
                )
                trace["user_id"] = user_id
                trace["character"] = persona.character
                self._send(200, trace)
            except Exception as exc:
                logger.warning("memory search error: %s", exc)
                self._send(500, {"error": str(exc)})

        def _memory_clear(self):
            """POST /memory/clear：清空某用户某角色的全部记忆（不可恢复）。鉴权同 /chat。"""
            if not self._authorized():
                metrics.inc("auth_failures")
                self._send(401, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length).decode("utf-8")
                body = json.loads(raw) if raw else {}
                user_id = str(body.get("user_id", "default")).strip() or "default"
                character = (str(body.get("character", "")).strip() or None)
                persona = users.entry(user_id, character)["persona"]
                if persona.memory_manager is None:
                    self._send(200, {"user_id": user_id,
                                     "character": persona.character,
                                     "cleared": True,
                                     "error": "该角色未启用观测笔记，无记忆可清"})
                    return
                cleared = persona.clear_memory()
                metrics.inc("memory_clear_times")
                metrics.emit("memory_clear", user_id=user_id,
                             character=persona.character)
                logger.warning("memory cleared user_id=%s char=%s counts=%s",
                               user_id, persona.character, cleared)
                self._send(200, {"user_id": user_id,
                                 "character": persona.character,
                                 "cleared": True, **cleared})
            except Exception as exc:
                logger.warning("memory clear error: %s", exc)
                self._send(500, {"error": str(exc)})

        def log_message(self, fmt, *args):
            # 走结构化日志，避免和指标事件刷屏冲突
            logger.debug("http " + fmt, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="千夜智子 HTTP API")
    parser.add_argument("--port", type=int, default=None, help="覆盖 config 的 server.port")
    parser.add_argument("--host", default=None, help="覆盖 config 的 server.host")
    args = parser.parse_args()

    try:
        config = load_config("config.yaml")
    except FileNotFoundError:
        print("找不到 config.yaml（请在 zhizi/ 目录下运行）")
        sys.exit(1)

    obs = config.get("observability") or {}
    setup_logging(log_file=obs.get("log_file") or None, level=obs.get("level") or "INFO")

    server_cfg = config.get("server", {}) or {}
    host = args.host or server_cfg.get("host", "127.0.0.1")
    port = args.port or int(server_cfg.get("port", 8765))
    api_key = server_cfg.get("api_key", "") or ""
    dashboard_enabled = bool(obs.get("dashboard", True))

    # 模型切换器：扫描本地模型目录，激活 config 指定的默认模型
    provider_cfg = config.get("provider") or {}
    llm_mod.init_model_registry(provider_cfg.get("model_dir") or "")
    default_model = provider_cfg.get("model") or ""
    if default_model:
        llm_mod.set_active_model(llm_mod.resolve_model_switch(default_model))
    logger.info("api starting host=%s port=%s dashboard=%s", host, port, dashboard_enabled)
    try:
        users = UserManager(config)
        users.get("default")  # 预热默认用户（尽早暴露配置错误）
    except RuntimeError as exc:
        logger.error("init failed: %s", exc)
        print(f"初始化失败：{exc}")
        sys.exit(1)

    metrics.set_gauge("users_active", len(users))
    metrics.emit("api_start", host=host, port=port, model=users.get("default").llm.model)

    handler = make_handler(users, api_key, dashboard_enabled=dashboard_enabled)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"千夜智子 API 已上线：http://{host}:{port}")
    print(f"  GET  /chat     聊天页面")
    print(f"  POST /chat     对话（user_id 用户级 × character 角色级隔离；hide_actions 隐藏动作描写）")
    print(f"  GET  /characters  角色清单（默认={users.default_character}）")
    print(f"  GET  /users     已知用户清单（用户下拉框数据源）")
    print(f"  GET  /history   聊天记录（?user_id=&character=&limit=，打开会话即回填）")
    print(f"  GET  /image     回读历史里的原图（?user_id=&character=&name=，鉴权同 /history）")
    print(f"  GET  /memory   记忆观察页面")
    print(f"  GET  /memory/api   五维记忆快照 + 最近召回 trace（?user_id=&character=）")
    print(f"  POST /memory/search 查询 → 召回过程明细")
    print(f"  POST /memory/clear   清空某用户某角色全部记忆（鉴权同 /chat）")
    for spec in users.list_characters():
        extra = []
        if not spec["memory"]:
            extra.append("无观测笔记")
        if not spec["visual_identity"]:
            extra.append("无自视身份")
        print(f"    - 角色「{spec['name']}」{'(默认) ' if spec['default'] else ''}"
              f"{('· '.join(extra)) if extra else ''}")
    print(f"  GET  /health    健康检查")
    print(f"  GET  /identity   自视身份卡 + 自识别开关状态")
    print(f"  GET  /model     模型切换器：现行模型 + 可切换清单")
    print(f"  POST /model     JSON('model': 路径) → 切换现行模型（mlx 释放旧引擎惰性重载）")
    print(f"  GET  /metrics   指标 JSON")
    print(f"  GET  /events    SSE 事件流")
    print(f"  GET  /chat     聊天页面")
    if dashboard_enabled:
        print(f"  GET  /dashboard 实时监控页")
    if api_key:
        print(f"  鉴权已开启：请求需带 Authorization: Bearer <api_key>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("api shutdown by keyboard")
        print("\n—— 智子去睡了 ——")


if __name__ == "__main__":
    main()
