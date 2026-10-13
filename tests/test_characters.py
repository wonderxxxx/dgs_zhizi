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

from core.characters import (  # noqa: E402
    default_character,
    load_characters,
    load_prompt_text,
    normalize_preset,
)
from core.guardrails import shared_guardrails  # noqa: E402
from core.persona import Persona  # noqa: E402
from core.presets import render_preset_prompt  # noqa: E402
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
                "preview_line": "喂——哥哥，帮我拿下薯片嘛，我在打 boss 呢。",
            },
            "白鸥": {
                "prompt": BAIOU_TXT,
                "visual_identity": "",
                "memory": False,
                "preview_line": "又没吃饭？……坐下，我端过来。",
                "guardrails": True,
                "preset": {
                    "性格原型": "全能管家",
                    "口头禅": "未知键应被丢弃",
                    "一句话台词": "交给我。",
                },
            },
            "毒舌学妹": {
                "preview_line": "哟，又累成狗了？",
                "memory": False,
                "search": False,
                "preset": {
                    "性格原型": "毒舌学妹",
                    "性格": "攻击性很强，但行动可靠。",
                    "一句话台词": "让开，我来。",
                },
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
    check("注册表：三个角色", set(chars) == {"千夜智子", "白鸥", "毒舌学妹"})
    check("默认角色：千夜智子（标记 default）",
          default_character(chars) == "千夜智子")
    baiou = chars["白鸥"]
    check("白鸥：无自视身份卡", baiou.visual_identity == "")
    check("白鸥：不主动观测（memory=False）", baiou.memory is False)
    check("智子：memory=True", chars["千夜智子"].memory is True)

    # ── 2b. 声明式元数据：preview_line / preset / guardrails ──────
    check("preview_line：智子已解析",
          chars["千夜智子"].preview_line.startswith("喂——"))
    check("preview_line：白鸥已解析",
          baiou.preview_line == "又没吃饭？……坐下，我端过来。")
    bdict = baiou.to_dict()
    check("to_dict：含 preview_line / guardrails",
          bdict.get("preview_line") == "又没吃饭？……坐下，我端过来。"
          and bdict.get("guardrails") is True)
    check("preset：未知键丢弃 + 已知键保留",
          baiou.preset == {"性格原型": "全能管家", "一句话台词": "交给我。"})
    check("preset：智子无 preset → to_dict 不含该键",
          "preset" not in chars["千夜智子"].to_dict())
    check("guardrails：默认关（智子零改动）",
          chars["千夜智子"].guardrails is False)
    check("normalize_preset：按 PRESET_FIELDS 顺序 + 去空值",
          list(normalize_preset(
              {"一句话台词": "x", "性格": "y", "爱好": ""}).keys())
          == ["性格", "一句话台词"])
    check("shared_guardrails：含四条纪律关键词",
          all(k in shared_guardrails()
              for k in ("不复读", "用量克制", "不讨好", "守边界")))

    # ── 2c. 预设型角色：无整卡 → 七字段渲染成提示词 + 纪律块自动并入 ──
    jb = chars["毒舌学妹"]
    check("预设角色：无整卡 prompt", jb.prompt_path == "" and jb.prompt_text == "")
    check("预设角色：preset 已解析",
          jb.preset == {"性格原型": "毒舌学妹", "性格": "攻击性很强，但行动可靠。",
                        "一句话台词": "让开，我来。"})
    check("预设角色：guardrails 自动开启（有效值）", jb.guardrails is True)
    sp = jb.system_prompt()
    check("预设角色：system prompt 由七字段渲染",
          "# 角色：毒舌学妹" in sp and "## 性格" in sp
          and "让开，我来。" in sp and "## 表达" in sp)
    check("预设角色：缺字段段落自动跳过",
          "## 口癖" not in sp and "## 爱好" not in sp)
    check("预设角色：自动拼公共纪律块",
          "不复读" in sp and "守边界" in sp and "用量克制" in sp)
    jd = jb.to_dict()
    check("预设角色：to_dict 含 preset、prompt 空、guardrails=true",
          jd["prompt"] == "" and "preset" in jd and jd["guardrails"] is True)
    check("渲染器：全空预设 → 空串",
          render_preset_prompt("x", {}) == "")

    # ── 3. 兼容旧配置：无 characters 段 → 单角色 default ──────────
    legacy = load_characters({
        "system_prompt": ZHIZI_MD,
        "visual_identity": {"card": IDENTITY},
    })
    check("旧配置：单角色 default", set(legacy) == {"default"})
    check("旧配置：default 启用记忆与身份卡",
          legacy["default"].memory is True and legacy["default"].visual_identity == IDENTITY)
    check("旧配置：default 无 preview_line、guardrails 关",
          legacy["default"].preview_line == ""
          and legacy["default"].guardrails is False)

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

    # ── 5b. guardrails 注入：显式开启才拼，默认角色零改动 ─────────
    check("guardrails：开启角色拼上纪律块",
          "不复读" in p.system_prompt and "守边界" in p.system_prompt)
    p_plain = Persona(cfg, llm=FakeLLM(), character=chars["千夜智子"])
    check("guardrails：默认角色不拼（零改动）",
          "不复读" not in p_plain.system_prompt)

    p_jb = Persona(cfg, llm=FakeLLM(), character=chars["毒舌学妹"])
    check("Persona：预设角色装配渲染提示词 + 纪律块",
          "毒舌学妹" in p_jb.system_prompt and "不复读" in p_jb.system_prompt)

    print()
    print(f"通过 {sum(PASS)}/{len(PASS)}")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()