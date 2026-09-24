"""回复解析：把智子的回复拆成「正文」与「动作/神情描写」。

约定：System Prompt 已规范——动作、神情、场景描写统一用全角括号（…）括起。
壳子（Flutter / Swift / Web）可按需决定是否展示动作行。
"""

import re

# 全角括号优先，半角兜底；交替匹配保持文本顺序
_PATTERN = re.compile(r"（[^（）]*）|\([^()]*\)")
_SPACES = re.compile(r"[ \t]{2,}")


def split_reply(raw):
    """返回 (content, actions)。

    - content：去掉动作描写后的正文（用于纯对话展示）
    - actions：按出现顺序提取的动作/神情列表（不含括号本身）
    """
    actions = []

    def take(match):
        actions.append(match.group(0).strip("（）() \t"))
        return ""

    content = _PATTERN.sub(take, raw)
    content = _SPACES.sub(" ", content)
    content = content.strip()
    return content, actions


def parse(raw, hide_actions=False):
    """壳子友好的便捷入口。

    hide_actions=True 时返回纯正文；否则返回 (content, actions) 二元组。
    """
    content, actions = split_reply(raw)
    if hide_actions:
        return content
    return content, actions
