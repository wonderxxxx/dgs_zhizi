"""路由层（后置，默认关闭）：小模型做语域/模式判断与插件调度。

设计意图：切换由智子「自己判断」，路由只做轻量辅助，不强制注入指令。
当前为占位实现；启用后接入便宜模型返回结构化决策。
"""


class Router:
    def __init__(self, config):
        router = config.get("router", {})
        self.enabled = bool(router.get("enabled", False))
        self.model = router.get("model", "doubao-lite")

    def decide(self, user_input):
        """返回 {"mode": "zhonger"|"doctor"|"coquetry"|None, "plugin": 名称|None}"""
        if not self.enabled:
            return None
        # TODO: 接入小模型（复用 LLMClient 指向便宜模型），
        #       输出结构化 JSON：语域判断 + 是否需要调用能力插件。
        return {"mode": None, "plugin": None}
