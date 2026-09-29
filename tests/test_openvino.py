#!/usr/bin/env python3
"""OpenVINO provider 验证：目录识别 / 引擎判定 / 图片占位符 / 模型切换器。

不加载真实模型——GPU 首次编译 1~2 分钟、权重 5~12GB，真机推理的手测步骤见
README「OpenVINO 分支」一节。这里只覆盖纯逻辑，无模型的机器也能跑。
运行方式：PYTHONUTF8=1 .venv/bin/python tests/test_openvino.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台默认 GBK，✅ 会崩
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core import llm  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    status = "✅" if cond else "❌"
    PASS.append(cond)
    print(f"{status} {name}" + (f"  ({detail})" if detail and not cond else ""))


def make_ov_dir(root, name, weights="openvino_model.xml", vision=False):
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, weights), "w").close()
    if vision:
        open(os.path.join(path, "openvino_vision_embeddings_model.xml"), "w").close()
    # Optimum 导出总带 HF 的 config.json，但它不是判定依据
    with open(os.path.join(path, "config.json"), "w") as fh:
        fh.write('{"model_type": "qwen2"}')
    return path


def main():
    tmp = tempfile.mkdtemp(prefix="zhizi_ov_")
    try:
        # ── 1. IR 目录识别（单文件式 / 分图式 / 视觉塔） ────────────────
        text_dir = make_ov_dir(tmp, "qwen2-7b-int4-ov")
        vlm_dir = make_ov_dir(tmp, "gemma3-12b-int8-ov",
                              weights="openvino_language_model.xml", vision=True)
        plain = os.path.join(tmp, "not_a_model")
        os.makedirs(plain, exist_ok=True)
        with open(os.path.join(plain, "config.json"), "w") as fh:
            fh.write("{}")

        check("识别单文件式 IR（openvino_model.xml）",
              llm.is_openvino_model_dir(text_dir))
        check("识别分图式 IR（openvino_language_model.xml）",
              llm.is_openvino_model_dir(vlm_dir))
        check("无权重文件的目录不算模型", not llm.is_openvino_model_dir(plain))
        check("不存在的路径不报错", not llm.is_openvino_model_dir(
            os.path.join(tmp, "nope")))

        # ── 2. 多模态判定：按 IR 目录结构判定 ──────────────────────────
        # 这是本分支最容易回归的点：带视觉塔的模型不能被误判成纯文本
        # 旧实现 import 失败直接 return False，gemma3 的视觉塔会被误判成纯文本
        check("文本 IR：auto 判定为非多模态",
              llm.detect_multimodal(text_dir, "auto") is False)
        check("带视觉塔的 IR：auto 判定为多模态",
              llm.detect_multimodal(vlm_dir, "auto") is True)
        check("multimodal=false 强制关掉视觉塔",
              llm.detect_multimodal(vlm_dir, "false") is False)
        check("multimodal=true 强制开启",
              llm.detect_multimodal(text_dir, True) is True)
        check("带视觉塔目录有 vision 标记", llm.ov_has_vision(vlm_dir))
        check("纯文本目录无 vision 标记", not llm.ov_has_vision(text_dir))

        # ── 3. 引擎选择（不实例化 pipeline，只走静态判定） ───────────────
        check("vision + auto → vlm 引擎",
              llm._OpenVINOExecutor.engine_kind(vlm_dir, multimodal=True) == "vlm")
        check("vision + 关多模态 → lm 引擎（省 13GB 视觉塔内存）",
              llm._OpenVINOExecutor.engine_kind(vlm_dir, multimodal=False) == "lm")
        check("无视觉塔 → lm 引擎",
              llm._OpenVINOExecutor.engine_kind(text_dir, multimodal=True) == "lm")

        # ── 4. 模型切换器：registry 能扫到 IR 目录 ────────────────────
        # 模拟真实布局：gemma3 目录里有个 1/ 子目录（下载器留下的重复副本），
        # 父目录已匹配时不应再往里扫
        os.makedirs(os.path.join(vlm_dir, "1"), exist_ok=True)
        open(os.path.join(vlm_dir, "1", "openvino_language_model.xml"), "w").close()
        reg = llm.init_model_registry(tmp)
        check("registry 收录两个 IR 模型", len(reg) == 2, f"got={reg}")
        check("重复副本子目录不重复注册",
              sum(1 for m in reg if m.rstrip("/\\").endswith("1")) == 0, f"got={reg}")
        check("_looks_like_model_dir 认 IR 目录", llm._looks_like_model_dir(text_dir))
        check("_looks_like_model_dir 不认空目录",
              not llm._looks_like_model_dir(plain))

        # ── 5. 图片占位符：只改本轮副本 ─────────────────────────────────
        msgs = [
            {"role": "system", "content": "你是千夜智子"},
            {"role": "user", "content": "我回来了"},
            {"role": "assistant", "content": "（笑）欢迎回来"},
            {"role": "user", "content": "你看这张图"},
        ]
        out = llm.LLMClient._ov_image_placeholders(msgs, 2)
        check("占位符挂到最后一条 user 上",
              out[-1]["content"] == "你看这张图\n<ov_genai_image_0><ov_genai_image_1>",
              f"got={out[-1]['content']!r}")
        check("不改动原消息列表（历史仍是纯文本）",
              msgs[-1]["content"] == "你看这张图",
              f"got={msgs[-1]['content']!r}")
        check("system / 历史轮不受影响",
              out[0] == msgs[0] and out[1] == msgs[1] and out[2] == msgs[2])
        check("无图片时不注入占位符",
              llm.LLMClient._ov_image_placeholders(msgs, 0) == msgs)
        check("没有 user 消息时原样返回",
              llm.LLMClient._ov_image_placeholders([msgs[0]], 1)[0] == msgs[0])

        # ── 6. 切模型只清当前 provider 的执行器 ─────────────────────────
        class FakeExec:
            def __init__(self):
                self.cleared = 0

            def clear(self):
                self.cleared += 1

        fake_ov = FakeExec()
        saved = (llm._ov_executor, llm._ACTIVE_PROVIDER)
        llm._ov_executor, llm._ACTIVE_PROVIDER = fake_ov, "openvino"
        try:
            llm.set_active_model(os.path.join(tmp, "qwen2-7b-int4-ov"))
            check("openvino 下切模型清掉 OV 引擎缓存", fake_ov.cleared == 1,
                  f"cleared={fake_ov.cleared}")
            check("现行模型已更新", llm.get_active_model().endswith("qwen2-7b-int4-ov"))
        finally:
            llm._ov_executor, llm._ACTIVE_PROVIDER = saved

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"通过 {sum(PASS)}/{len(PASS)}")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
