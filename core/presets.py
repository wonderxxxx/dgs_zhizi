"""声明式人格 → 人格提示词渲染器（纯函数，供无整卡的轻量角色用）。

结构致敬 N.E.K.O. 的 preset schema（性格原型/性格/口癖/爱好/雷点/隐藏设定/
一句话台词）。表达纪律（不复读/用量克制/不讨好/守边界）由 core.guardrails
公共块统一在后端注入，不在这里重复写进每个角色。

同时提供「mood 层」：给已有的完整卡（如智子）加一层可选的可变状态。
mood 只覆盖表达层（语气/行为倾向），不动身份/记忆/判断力/红线，且必须带
退出线——情绪是流动的，不能一条指令变一辈子病弱。渲染得到的是覆盖段而非
完整的 system prompt，由 Persona.build_messages 追加到首条 system 末尾。
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


# ── mood 层：稳定卡之上的可选可变状态段 ──────────────────────────
MOOD_FIELDS = ("状态", "语气", "行为倾向", "退出线")


def normalize_mood(mood):
    """状态字段 → 只留已知键、按 MOOD_FIELDS 顺序、去空值。非法返回 {}。"""
    if not isinstance(mood, dict):
        return {}
    out = {}
    for key in MOOD_FIELDS:
        val = mood.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            out[key] = text
    return out


def load_moods(config):
    """config 的 moods 段 → {状态名: 清洗后的字段}；非法条目跳过。"""
    raw = (config or {}).get("moods")
    if not isinstance(raw, dict):
        return {}
    out = {}
    for name, fields in raw.items():
        name = str(name).strip()
        norm = normalize_mood(fields)
        if name and norm:
            out[name] = norm
    return out


def render_mood(character, name, mood):
    """状态段 → 覆盖层提示文本（追加用，不是完整 system prompt）。

    - 缺字段的段落自动跳过；状态字段全空返回空串。
    - 只覆盖表达层：身份/记忆/判断力/对在乎之人的感情全部照旧，
      并写入退出线让状态能自然结束。
    """
    mood = mood or {}
    if not mood:
        return ""
    parts = [f"【当前状态：{name}】（仅本轮的表达覆盖，不改变身份）"]
    if mood.get("状态"):
        parts.append(f"- {character}此刻：{mood['状态']}")
    for key in ("语气", "行为倾向", "退出线"):
        val = mood.get(key)
        if val:
            parts.append(f"- {key}：{val}")
    parts.append(
        f"- 这个状态只覆盖表达层：{character}的身份、记忆、判断力与"
        "对在乎之人的感情全部照旧；若上面的退出线被触发或话题转入要紧事，"
        "自然退出该状态。"
    )
    return "\n".join(parts)
