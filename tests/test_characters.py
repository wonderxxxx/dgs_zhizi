#!/usr/bin/env python3
"""角色注册表验证：一级用户 × 二级角色。

覆盖：人物卡两种格式读取 / 能力开关（观测笔记、自视身份）/
      UserManager 二级隔离（默认角色沿用一级目录，其余落到 <用户>/<角色>/）/
      记忆能力门控（白鸥无记忆管理器）。
运行方式：.venv/bin/python tests/test_characters.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台默认 GBK，✅ 会崩
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.characters import (  # noqa: E402
    default_character,
    load_characters,
    load_prompt_text,
)
from core.persona import Persona  # noqa: E402
from core.users import UserManager  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZHIZI_MD = os.path.join(BASE, "prompts", "zhizi_v1.md")
BAIOU_TXT = os.path.join(BASE, "prompts", "白鸥人物卡-万物皆数定制版.txt")
IDENTITY = os.path.join(BASE, "visual", "identity.json")

PASS = []


def check(name, cond, detail=""):
    status = "✅" if cond else "❌"
    PASS.append(cond)
    print(f"{status} {name}" + (f"  ({detail})" if detail and not cond else ""))


class FakeLLM:
    model = "fake"
    multimodal = False

    def __init__(self, config=None):
        pass

    def chat(self, messages, temperature=None, images=None, enable_thinking=None):
        return "（点头）嗯。"


def make_cfg(tmp):
    return {
        "characters": {
            "千夜智子": {
                "prompt": ZHIZI_MD,
                "visual_identity": IDENTITY,
                "memory": True,
                "default": True,
            },
            "白鸥": {
                "prompt": BAIOU_TXT,
                "visual_identity": "",
                "memory": False,
            },
        },
        "memory": {"note_file": os.path.join(tmp, "notes"), "top_k": 5},
        "provider": {"type": "local", "base_url": "http://x", "model": "fake",
                     "api_key": "x", "temperature": 0.85},
    }


def main():
    tmp = tempfile.mkdtemp(prefix="zhizi_chars_")

    # ── 1. 人物卡读取：Python 字面量包装 与 纯 Markdown ─────────────
    t = load_prompt_text(BAIOU_TXT)
    check("包装卡：去掉 SYS_PROMPT = r''' 外壳",
          isinstance(t, str) and t.startswith("# 白鸥") and "SYS_PROMPT" not in t)
    check("包装卡：内容保留", "乌鸦" in t and "黑鹰" in t)
    t2 = load_prompt_text(ZHIZI_MD)
    check("纯 Markdown 卡：原样读取", t2.startswith("# 角色：千夜智子"))

    # ── 2. 注册表解析 + 能力开关 ──────────────────────────────────
    chars = load_characters(make_cfg(tmp))
    check("注册表：两个角色", set(chars) == {"千夜智子", "白鸥"})
    check("默认角色：千夜智子（标记 default）",
          default_character(chars) == "千夜智子")
    baiou = chars["白鸥"]
    check("白鸥：无自视身份卡", baiou.visual_identity == "")
    check("白鸥：不主动观测（memory=False）", baiou.memory is False)
    check("智子：memory=True", chars["千夜智子"].memory is True)

    # ── 3. 兼容旧配置：无 characters 段 → 单角色 default ──────────
    legacy = load_characters({
        "system_prompt": ZHIZI_MD,
        "visual_identity": {"card": IDENTITY},
    })
    check("旧配置：单角色 default", set(legacy) == {"default"})
    check("旧配置：default 启用记忆与身份卡",
          legacy["default"].memory is True and legacy["default"].visual_identity == IDENTITY)

    # ── 4. UserManager 二级隔离 + 能力门控（白鸥无记忆） ──────────
    cfg = make_cfg(tmp)
    um = UserManager(cfg, max_users=64)
    check("UserManager 默认角色", um.default_character == "千夜智子")

    z = um.entry("alice")["persona"]
    check("智子实例：character 正确", z.character == "千夜智子")
    check("智子：有观测记忆管理器", z.memory_manager is not None)
    check("智子：记忆落在一级目录 notes/alice/memory",
          z.memory_manager.memory_dir == os.path.join(tmp, "notes", "alice", "memory"))
    check("智子：自视身份已启用", getattr(z.visual_identity, "enabled", False) is True)

    b = um.entry("alice", "白鸥")["persona"]
    check("白鸥实例：character 正确", b.character == "白鸥")
    check("白鸥：无观测记忆管理器", b.memory_manager is None)
    check("白鸥：自视身份未启用", getattr(b.visual_identity, "enabled", False) is False)
    check("白鸥：不落盘记忆目录", not os.path.exists(os.path.join(tmp, "notes", "alice", "白鸥")))

    # 同一键复用同一实例（历史/记忆连续）
    check("键复用：同一 (user, char) 返回同一实例", um.entry("alice")["persona"] is z)
    check("键隔离：不同角色各自独立", um.entry("alice", "白鸥")["persona"] is b)

    # ── 4b. 已知用户清单（用户下拉框的数据源） ────────────────
    os.makedirs(os.path.join(tmp, "notes", "carol"), exist_ok=True)  # 落盘但未进内存
    listed = um.list_users()
    check("用户清单：含内存中的用户 alice", "alice" in listed)
    check("用户清单：含落盘未激活的用户 carol", "carol" in listed)
    check("用户清单：恒含 default", "default" in listed)
    check("用户清单：角色子目录不算独立用户 / 有序去重",
          "白鸥" not in listed and listed == sorted(set(listed)))

    # ── 5. Persona 装配（注入 fake LLM，避免建连） ────────────────
    p = Persona(cfg, llm=FakeLLM(), character=chars["白鸥"])
    check("Persona：白鸥 prompt 已装配", "白鸥" in p.system_prompt and "乌鸦" in p.system_prompt)
    check("Persona：白鸥 无观察记忆", p.memory_manager is None)

    print()
    print(f"通过 {sum(PASS)}/{len(PASS)}")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()