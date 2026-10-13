"""角色注册表：二级制——一级用户，二级角色。

每个角色有独立的 System Prompt（人物卡）与记忆命名空间；
能力按角色启用：自视身份卡、观测笔记（主动记忆）、按需联网搜索等，未配置的自动关闭。

人物卡支持两种格式：
- 纯文本 / Markdown：直接作为 System Prompt 内容；
- Python 字面量包装（`SYS_PROMPT = r'''...'''`）：用 ast 解析赋值，
  只取 SYS_PROMPT 的字符串值——人物卡本身零修改即可接入。
"""

import ast
import re

from .guardrails import shared_guardrails
from .presets import render_preset_prompt

DEFAULT_CHARACTER = "default"

# 声明式人格预设的字段（顺序即推荐展示顺序）。
# 取自 N.E.K.O. 的 preset schema，供「无整卡 / 轻量」角色使用：结构化字段即可
# 拼出人格，不必每角色手写一整篇卡。有整卡的角色字段仍可填，仅作元数据/预览用。
PRESET_FIELDS = (
    "性格原型", "性格", "口癖", "爱好", "雷点", "隐藏设定", "一句话台词",
)


def normalize_preset(preset):
    """结构化人格字段 → 只留已知键、按 PRESET_FIELDS 顺序、去空值。"""
    if not isinstance(preset, dict):
        return {}
    out = {}
    for key in PRESET_FIELDS:
        val = preset.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            out[key] = text
    return out


class CharacterSpec:
    """单角色规格：prompt 文本 + 能力开关 + 命名空间标记 + 声明式人格元数据。"""

    def __init__(self, name, prompt_path="", prompt_text="", visual_identity="",
                 memory=True, search=True, default=False,
                 preview_line="", preset=None, guardrails=False):
        self.name = name
        self.prompt_path = prompt_path
        self.prompt_text = prompt_text
        self.visual_identity = visual_identity  # 自视身份卡路径；空=无（不自识别）
        self.memory = memory                    # 观测笔记（主动记忆）是否可用
        self.search = search                    # 按需联网搜索是否可用
        self.default = default
        self.preview_line = str(preview_line or "").strip()  # 选择器一句话预告
        self.preset = normalize_preset(preset)  # 七字段声明式人格（可为空）
        # 预设型角色（无整卡只有预设）自动吃公共纪律块；整卡角色按显式开关（默认关，
        # 避免通用纪律与卡片自带的表达风格打架——白鸥长镜头叙事即反例）。
        card_text = (prompt_text or "").strip()
        self.preset_only = bool(self.preset) and not card_text
        self.guardrails = bool(guardrails) or self.preset_only

    def to_dict(self):
        out = {
            "name": self.name,
            "prompt": self.prompt_path,
            "visual_identity": self.visual_identity,
            "memory": self.memory,
            "search": self.search,
            "default": self.default,
            "preview_line": self.preview_line,
            "guardrails": self.guardrails,  # 有效值：预设型角色恒 true
        }
        if self.preset:
            out["preset"] = dict(self.preset)
        return out

    def system_prompt(self):
        """该角色最终的 system prompt：整卡优先，否则预设渲染，均并入纪律块。"""
        card = (self.prompt_text or "").strip()
        if card:
            base = card
        elif self.preset:
            base = render_preset_prompt(self.name, self.preset)
        else:
            base = ""
        if base and self.guardrails:
            base = f"{base}\n\n{shared_guardrails()}"
        return base


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
            if not isinstance(cfg, dict):
                continue
            prompt_path = str(cfg.get("prompt") or "").strip()
            preset = normalize_preset(cfg.get("preset"))
            if not prompt_path and not preset:
                continue  # 既无整卡也无预设字段 → 不是有效角色，跳过
            specs[name] = CharacterSpec(
                name=name,
                prompt_path=prompt_path,
                prompt_text=load_prompt_text(prompt_path),
                visual_identity=str(cfg.get("visual_identity") or "").strip(),
                memory=bool(cfg.get("memory", True)),
                search=bool(cfg.get("search", True)),
                default=bool(cfg.get("default", False)),
                preview_line=str(cfg.get("preview_line") or "").strip(),
                preset=preset,
                guardrails=bool(cfg.get("guardrails", False)),
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
            search=True,
            default=True,
        ),
    }


def default_character(specs):
    """默认角色名：标记 default 者优先，否则取注册表首位。"""
    for name, spec in specs.items():
        if spec.default:
            return name
    return next(iter(specs), DEFAULT_CHARACTER)