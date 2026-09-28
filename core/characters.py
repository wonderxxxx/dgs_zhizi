"""角色注册表：二级制——一级用户，二级角色。

每个角色有独立的 System Prompt（人物卡）与记忆命名空间；
能力按角色启用：自视身份卡、观测笔记（主动记忆）等，未配置的自动关闭。

人物卡支持两种格式：
- 纯文本 / Markdown：直接作为 System Prompt 内容；
- Python 字面量包装（`SYS_PROMPT = r'''...'''`）：用 ast 解析赋值，
  只取 SYS_PROMPT 的字符串值——人物卡本身零修改即可接入。
"""

import ast
import re

DEFAULT_CHARACTER = "default"


class CharacterSpec:
    """单角色规格：prompt 文本 + 能力开关 + 命名空间标记。"""

    def __init__(self, name, prompt_path="", prompt_text="", visual_identity="",
                 memory=True, default=False):
        self.name = name
        self.prompt_path = prompt_path
        self.prompt_text = prompt_text
        self.visual_identity = visual_identity  # 自视身份卡路径；空=无（不自识别）
        self.memory = memory                    # 观测笔记（主动记忆）是否可用
        self.default = default

    def to_dict(self):
        return {
            "name": self.name,
            "prompt": self.prompt_path,
            "visual_identity": self.visual_identity,
            "memory": self.memory,
            "default": self.default,
        }


def load_prompt_text(path):
    """读取人物卡为 System Prompt 纯文本。

    兼容纯文本 / Markdown 文件，以及 `SYS_PROMPT = r'''...'''` 的 Python
    字面量包装（用 ast.literal_eval 取值，支持全部引号与转义，不改原卡）。
    """
    if not path:
        return ""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if re.search(r"^\s*SYS_PROMPT\s*=", text, re.M):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "SYS_PROMPT":
                        try:
                            return ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            continue
    return text


def load_characters(config):
    """从 config 解析角色注册表，返回 {name: CharacterSpec}。

    无 `characters` 段时向后兼容旧配置：单角色「default」，
    用全局 system_prompt + visual_identity.card，记忆开启。
    """
    raw = (config or {}).get("characters")
    if raw:
        specs = {}
        for name, cfg in raw.items():
            if not isinstance(cfg, dict) or not cfg.get("prompt"):
                continue
            prompt_path = str(cfg["prompt"]).strip()
            specs[name] = CharacterSpec(
                name=name,
                prompt_path=prompt_path,
                prompt_text=load_prompt_text(prompt_path),
                visual_identity=str(cfg.get("visual_identity") or "").strip(),
                memory=bool(cfg.get("memory", True)),
                default=bool(cfg.get("default", False)),
            )
        if specs:
            return specs

    prompt_path = str((config or {}).get("system_prompt") or "").strip()
    vis = (config or {}).get("visual_identity") or {}
    return {
        DEFAULT_CHARACTER: CharacterSpec(
            name=DEFAULT_CHARACTER,
            prompt_path=prompt_path,
            prompt_text=load_prompt_text(prompt_path),
            visual_identity=str(vis.get("card") or "").strip(),
            memory=True,
            default=True,
        )
    }


def default_character(specs):
    """默认角色名：标记 default 者优先，否则取注册表首位。"""
    for name, spec in specs.items():
        if spec.default:
            return name
    return next(iter(specs), DEFAULT_CHARACTER)