"""五维记忆系统：工作记忆 / 近期记忆 / 事实记忆 / 反思记忆 / 人格记忆。

简化版实现，基于 N.E.K.O 的设计思路。
"""

from .recent import RecentMemory
from .timeindex import TimeIndexedMemory
from .facts import FactMemory
from .reflection import ReflectionMemory
from .persona import PersonaMemory
from .manager import MemoryManager

__all__ = [
    "RecentMemory",
    "TimeIndexedMemory", 
    "FactMemory",
    "ReflectionMemory",
    "PersonaMemory",
    "MemoryManager",
]