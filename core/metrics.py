"""运行指标 + 结构化日志（零第三方依赖）。

- 指标：进程内计数器/延迟桶，供 /metrics 与 dashboard 展示
- 日志：stderr + 可选滚动文件 logs/zhizi.log（config: observability.log_file）
- 事件：环形缓冲最近 N 条，SSE /events 推给前端
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, Deque, List, Optional


class _LatencyStats:
    __slots__ = ("count", "total", "min", "max", "last")

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.min = 0.0
        self.max = 0.0
        self.last = 0.0

    def observe(self, ms: float):
        self.count += 1
        self.total += ms
        self.last = ms
        if self.min == 0.0 or ms < self.min:
            self.min = ms
        if ms > self.max:
            self.max = ms

    def snapshot(self) -> Dict[str, float]:
        avg = self.total / self.count if self.count else 0.0
        return {
            "count": self.count,
            "last_ms": round(self.last, 2),
            "avg_ms": round(avg, 2),
            "min_ms": round(self.min, 2),
            "max_ms": round(self.max, 2),
        }


class Metrics:
    """线程安全的进程内指标聚合。"""

    def __init__(self, max_events: int = 200):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, _LatencyStats] = {}
        self.gauges: Dict[str, float] = {}
        self.events: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._listeners: List[Any] = []  # callables(event_dict)

    # ── 计数 ──
    def inc(self, name: str, value: int = 1):
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + value

    # ── 仪表 ──
    def set_gauge(self, name: str, value: float):
        with self._lock:
            self.gauges[name] = value

    # ── 延迟（毫秒） ──
    def observe_ms(self, name: str, ms: float):
        with self._lock:
            timer = self.timers.get(name)
            if timer is None:
                timer = self.timers[name] = _LatencyStats()
            timer.observe(ms)

    @contextmanager
    def timer(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe_ms(name, (time.perf_counter() - t0) * 1000.0)

    # ── 事件（给 SSE / 最近日志） ──
    def add_listener(self, fn):
        with self._lock:
            self._listeners.append(fn)
        return fn

    def remove_listener(self, fn):
        with self._lock:
            try:
                self._listeners.remove(fn)
            except ValueError:
                pass

    def emit(self, kind: str, **fields):
        event = {"ts": time.time(), "kind": kind, **fields}
        listeners = []
        with self._lock:
            self.events.append(event)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(event)
            except Exception:
                pass
        # 同步打一条结构化日志
        try:
            logger.info("event=%s %s", kind, json.dumps(fields, ensure_ascii=False, default=str))
        except Exception:
            pass
        return event

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "uptime_sec": round(time.time() - self.started_at, 1),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timers": {k: v.snapshot() for k, v in self.timers.items()},
                "recent_events": list(self.events)[-50:],
            }


# 全局单例
metrics = Metrics()

# ── 日志 ──────────────────────────────────────────────
logger = logging.getLogger("zhizi")
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(log_file: Optional[str] = None, level: str = "INFO"):
    """配置 stderr + 可选文件日志。幂等。"""
    logger.setLevel(getattr(logging, (level or "INFO").upper(), logging.INFO))
    logger.propagate = False

    # 避免重复 add
    have_stderr = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not have_stderr:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(h)

    if log_file:
        log_file = os.path.expanduser(log_file)
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        abs_path = os.path.abspath(log_file)
        have_file = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == abs_path
            for h in logger.handlers
        )
        if not have_file:
            fh = logging.FileHandler(abs_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_LOG_FORMAT))
            logger.addHandler(fh)
