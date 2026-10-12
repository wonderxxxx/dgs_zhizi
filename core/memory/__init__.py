"""五维记忆系统：工作记忆 / 近期记忆 / 事实记忆 / 反思记忆 / 人格记忆。

简化版实现，基于 N.E.K.O 的设计思路。
"""

from .recent import RecentMemory
from .timeindex import TimeIndexedMemory
from .facts import FactMemory
from .reflection import ReflectionMemory
from .persona import PersonaMemory
from .manager import MemoryManager

# 每个 MemoryManager 都严格绑定一条 (user_id × character) 的 memory_dir
# （users._make() 按 notes/<user_id>/<角色>/memory/ 生成，隔离靠目录完成）。
# 内部各维度存储虽然也用 user_id 分键，但那只是「这个实例自己的命名空间」——
# 对一个实例永远传同一个键即可，所以统一为常量，别散落魔法字符串。
# 千万不要把一个 MemoryManager 复用到多个用户/角色（那会让几个人的记忆写进
# 同一张表，隔离立刻穿帮）；也不要拿请求里的真实 user_id 顶替，否则同一实例
# 内会出现两个键、历史/笔记从同一个目录下读到两套并存的数据。
MEMORY_NAMESPACE = "default"

__all__ = [
    "MEMORY_NAMESPACE",
    "RecentMemory",
    "TimeIndexedMemory",
    "FactMemory",
    "ReflectionMemory",
    "PersonaMemory",
    "MemoryManager",
]