#!/usr/bin/env python3
"""千夜智子 · 命令行入口（人格核心验证形态，界面壳之前的交互方式）。

运行：
    pip install -r requirements.txt
    export ARK_API_KEY=你的密钥        # 或按 config.yaml 的 provider 配置
    python main.py

参数：
    --no-actions   隐藏（动作/神情）描写，只显示台词正文
"""

import argparse
import sys

import yaml

from core.metrics import logger, setup_logging
from core.persona import Persona
from core.router import Router

CONFIG_PATH = "config.yaml"


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main():
    parser = argparse.ArgumentParser(description="千夜智子 CLI")
    parser.add_argument("--no-actions", action="store_true",
                        help="隐藏（动作/神情）描写，只显示正文")
    args = parser.parse_args()

    try:
        config = load_config(CONFIG_PATH)
    except FileNotFoundError:
        print(f"找不到配置文件：{CONFIG_PATH}（请在 zhizi/ 目录下运行）")
        sys.exit(1)

    obs = config.get("observability") or {}
    setup_logging(log_file=obs.get("log_file") or None, level=obs.get("level") or "INFO")

    try:
        persona = Persona(config)
        router = Router(config)
    except RuntimeError as exc:
        logger.error("init failed: %s", exc)
        print(f"初始化失败：{exc}")
        sys.exit(1)

    print("千夜智子 已上线（场景：同居日常）。输入 exit / quit 退出。")
    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        router.decide(user_input)  # 后置：当前为空操作
        try:
            if args.no_actions:
                reply = persona.reply_structured(user_input)["reply"]
            else:
                reply = persona.reply(user_input)
        except RuntimeError as exc:
            logger.warning("chat error: %s", exc)
            print(f"[错误] {exc}")
            continue
        print(f"智子> {reply}")

    print("—— 智子去睡了 ——")


if __name__ == "__main__":
    main()
