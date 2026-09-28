#!/usr/bin/env python3
"""命令行入口（人格核心验证形态，界面壳之前的交互方式）。

运行：
    pip install -r requirements.txt
    export ARK_API_KEY=你的密钥        # 或按 config.yaml 的 provider 配置
    python main.py

参数：
    --no-actions   隐藏（动作/神情）描写，只显示台词正文
    --character    选择启用角色（默认角色的 prompt），如：--character 白鸥
"""

import argparse
import sys

import yaml

from core.characters import default_character, load_characters
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
    parser.add_argument("--character", default=None,
                        help="启用角色名（默认 config.characters 的默认角色）")
    args = parser.parse_args()

    try:
        config = load_config(CONFIG_PATH)
    except FileNotFoundError:
        print(f"找不到配置文件：{CONFIG_PATH}（请在 zhizi/ 目录下运行）")
        sys.exit(1)

    obs = config.get("observability") or {}
    setup_logging(log_file=obs.get("log_file") or None, level=obs.get("level") or "INFO")

    chars = load_characters(config)
    char_name = (args.character or "").strip() or default_character(chars)
    if char_name not in chars:
        print(f"未知角色：{char_name}（可选：{', '.join(chars)}）")
        sys.exit(1)

    try:
        persona = Persona(config, character=chars[char_name])
        router = Router(config)
    except RuntimeError as exc:
        logger.error("init failed: %s", exc)
        print(f"初始化失败：{exc}")
        sys.exit(1)

    print(f"「{char_name}」已上线。输入 exit / quit 退出。")
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
        print(f"{char_name}> {reply}")

    print(f"—— {char_name} 走了 ——")


if __name__ == "__main__":
    main()
