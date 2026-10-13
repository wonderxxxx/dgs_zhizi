"""声明式人格 → 人格提示词渲染器（纯函数，供无整卡的轻量角色用）。

结构致敬 N.E.K.O. 的 preset schema（性格原型/性格/口癖/爱好/雷点/隐藏设定/
一句话台词）。表达纪律（不复读/用量克制/不讨好/守边界）由 core.guardrails
公共块统一在后端注入，不在这里重复写进每个角色。
"""

_SECTION_TITLES = (
    ("性格原型", "## 性格原型"),
    ("性格", "## 性格"),
    ("口癖", "## 口癖（说话方式）"),
    ("爱好", "## 爱好"),
    ("雷点", "## 雷点（绝不越过的线）"),
    ("隐藏设定", "## 隐藏设定（底色，不主动点破）"),
    ("一句话台词", "## 一句话台词（风格参考，禁止逐字复读）"),
)


def render_preset_prompt(name, preset):
    """把七字段人格渲染成完整 system prompt。

    - 缺字段的段落自动跳过；预设全空返回空串（调用方据此决定是否回退）。
    - 只渲染人格本体；表达纪律由 core.guardrails 统一追加。
    """
    preset = preset or {}
    if not preset:
        return ""
    parts = [
        f"# 角色：{name}",
        "",
        "## 身份",
        f"- {name} 是一个独立的人，不是系统、不是助手。以角色自居，"
        "不把自己当 AI 或技术产物提及或承认。",
    ]
    for key, title in _SECTION_TITLES:
        val = (preset.get(key) or "").strip()
        if val:
            parts.extend(["", title, val])
    parts.extend([
        "",
        "## 表达",
        "- 口语化应答，输出你能实际说出口的话。",
        "- 不用 Markdown 加粗/列表、不刷 Emoji、不加剧本式旁白或"
        "内心独白，除非对方明确要求角色扮演。",
    ])
    return "\n".join(parts)