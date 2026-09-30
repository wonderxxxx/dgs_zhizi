#!/usr/bin/env python3
"""视觉能力探针：config 声明多模态 ≠ 权重里有视觉塔。

背景：AtomicChat 那份 gemma-4-31B-it-MLX-4bit 的 config.json 照旧带
image_token_id / audio_config，但转换时漏掉了视觉塔权重，mlx_vlm 加载时把
vision_tower 置 None（transformers 打出 "Checkpoint has no weights for
module(s): vision_tower"），随后每次带图推理都炸成
`'NoneType' object is not callable`。所以「能不能看图」必须以权重为准。

覆盖：权重清单读取（index / 单文件 safetensors / npz / 读不到）/
      vision_capability 的判据优先级（强制开关 > 权重 > config）/
      加载后复查 / client_vision 门控。
运行方式：.venv/bin/python tests/test_vision_probe.py
"""

import json
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import llm as L  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    cond = bool(cond)
    PASS.append(cond)
    print(f"{'✅' if cond else '❌'} {name}" + (f"  ({detail})" if detail and not cond else ""))


def make_model(root, name, cfg=None, keys=None, index=True, npz=False):
    """造一个模型目录：config.json + 权重清单。"""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    if cfg is not None:
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
    if keys is not None and index:
        with open(os.path.join(d, "model.safetensors.index.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"weight_map": {k: "model.safetensors" for k in keys}}, fh)
    elif keys is not None and npz:
        with zipfile.ZipFile(os.path.join(d, "model.npz"), "w") as zf:
            for k in keys:
                zf.writestr(k + ".npy", b"")
    return d


MODAL_CFG = {"model_type": "gemma4", "image_token_id": 258880, "vision_config": {}}
TEXT_CFG = {"model_type": "qwen3", "text_config": {}}
VISION_KEYS = ["language_model.model.embed_tokens.weight",
               "vision_tower.encoder.layers.0.weight",
               "embed_vision.embedding_projection.weight"]
TEXT_KEYS = ["language_model.model.embed_tokens.weight",
             "language_model.model.layers.0.self_attn.q_proj.weight"]


def main():
    tmp = tempfile.mkdtemp(prefix="zhizi_vision_")
    L._vision_probe_cache.clear()

    # ── 权重清单读取 ──────────────────────────────────────────────
    d_v = make_model(tmp, "pub-a/vlm-model", MODAL_CFG, VISION_KEYS)
    d_t = make_model(tmp, "pub-b/vlm-model", MODAL_CFG, TEXT_KEYS)
    d_n = make_model(tmp, "pub-c/vlm-model", MODAL_CFG, None)
    d_p = make_model(tmp, "pub-d/vlm-model", MODAL_CFG, VISION_KEYS, index=False, npz=True)
    check("探针：index 里有 vision_tower → True", L._has_vision_weights(d_v) is True)
    check("探针：只有语言模型张量 → False", L._has_vision_weights(d_t) is False)
    check("探针：无权重清单 → None（不下判断）", L._has_vision_weights(d_n) is None)
    check("探针：npz 清单同样能读", L._has_vision_weights(d_p) is True)

    # 兄弟命名：Qwen 系视觉塔叫 model.visual.*，不含 "vision"
    d_visual = make_model(tmp, "pub-e/qwen-vl", MODAL_CFG,
                          ["model.language_model.layers.0.mlp.weight",
                           "model.visual.merger.weight"])
    check("探针：model.visual.* 也算有视觉塔", L._has_vision_weights(d_visual) is True)

    # ── vision_capability 判据优先级 ──────────────────────────────
    check("capability：声明多模态 + 有视觉塔 → True",
          L.vision_capability(d_v) is True)
    check("capability：声明多模态但无视觉塔 → False（这就是那个 bug）",
          L.vision_capability(d_t) is False)
    check("capability：判不出来时沿用 config → True",
          L.vision_capability(d_n) is True)
    d_text = make_model(tmp, "pub-f/text-model", TEXT_CFG, TEXT_KEYS)
    check("capability：config 没声明模态 → False",
          L.vision_capability(d_text) is False)
    check("capability：显式 true 覆盖权重缺失",
          L.vision_capability(d_t, True) is True)
    check("capability：显式 false 覆盖有权重",
          L.vision_capability(d_v, False) is False)
    check("capability：非目录路径（远端仓库）→ False",
          L.vision_capability("org/model", "auto") is False)

    # 引擎选择与看图能力解耦：mlx_vlm 仍负责加载 multimodal 架构
    check("引擎：声明多模态的架构仍走 mlx_vlm",
          L.detect_multimodal(d_t) is True)
    check("引擎：纯文本架构走 mlx_lm",
          L.detect_multimodal(d_text) is False)

    # ── 加载后复查 ───────────────────────────────────────────────
    class Broken:  # transformers 禁用视觉塔后的样子
        vision_tower = None
        embed_vision = None
        audio_tower = None

    class Healthy:
        vision_tower = object()
        embed_vision = object()
        audio_tower = None  # 音频塔缺失不影响视觉判定

    check("复查：视觉子模块为 None → 判缺",
          L._vision_disabled(Broken()) is True)
    check("复查：视觉子模块正常 → 判不缺",
          L._vision_disabled(Healthy()) is False)

    # ── client_vision 门控 ───────────────────────────────────────
    class MlxClient:
        provider_type = "mlx"
        model = d_t
        multimodal = True
        vision = False

    class OkClient(MlxClient):
        model = d_v
        vision = True

    class ServerClient:  # api / local：由服务端决定
        provider_type = "local"
        model = "doubao-seed"
        multimodal = True

    class LegacyFake:  # 旧测试替身：只有 multimodal
        model = "fake"
        multimodal = True

    check("client_vision：mlx 无视觉塔 → False（不启线程）",
          L.client_vision(MlxClient()) is False)
    check("client_vision：None 客户端 → False", L.client_vision(None) is False)
    check("client_vision：老替身回落到 multimodal → True",
          L.client_vision(LegacyFake()) is True)
    check("client_vision：OpenAI 兼容 provider → 透传 multimodal",
          L.client_vision(ServerClient()) is True)

    ex = L._MlxExecutor.__new__(L._MlxExecutor)  # 不起 MLX 线程
    ex._no_vision = {d_t}
    check("client_vision：加载后复查到缺权重 → 降级",
          L._MlxExecutor.vision_missing(ex, d_t) is True)
    check("client_vision：未加载的模型不误判",
          L._MlxExecutor.vision_missing(ex, d_v) is False)
    ok = OkClient()
    ok.vision = True
    ex._no_vision = set()
    check("client_vision：有权重的模型 → True", ex.vision_missing(ok.model) is False)

    # ── 兄弟版本提示 ─────────────────────────────────────────────
    L._MODEL_REGISTRY = [d_t, d_v]
    hint = L._vision_sibling_hint(d_t)
    check("提示：指名带视觉塔的兄弟版本", d_v in hint, hint)
    L._MODEL_REGISTRY = []

    # 探针有缓存：重复调用不重复告警
    before = len(L._vision_probe_cache)
    L.vision_capability(d_t)
    check("探针结果有缓存", len(L._vision_probe_cache) == before)

    # ── 缺视觉塔告警：每个模型每进程只出一次 ────────────────────
    # 模型切换器每列一次模型、每次带图请求都会重新判一遍；不去重会反复刷屏，
    # 且 metrics.emit 每次都往内存 events 列表追加，涨个没完。
    L._vision_missing_warned.clear()
    calls = {"warn": 0, "emit": 0, "inc": 0}
    real_warn, real_emit, real_inc = L.logger.warning, L.metrics.emit, L.metrics.inc

    def spy_warn(*a, **k):
        calls["warn"] += 1
        return real_warn(*a, **k)

    def spy_emit(kind, **f):
        calls["emit"] += 1
        return real_emit(kind, **f)

    def spy_inc(name, **f):
        calls["inc"] += 1
        return real_inc(name, **f)

    L.logger.warning, L.metrics.emit, L.metrics.inc = spy_warn, spy_emit, spy_inc
    try:
        for _ in range(5):
            L.vision_capability(d_t)
        check("告警去重：缺视觉塔连判 5 次只警告 1 次", calls["warn"] == 1, f"warn={calls['warn']}")
        check("告警去重：结构化事件也只发 1 次", calls["emit"] == 1, f"emit={calls['emit']}")
        check("告警去重：指标只累加 1 次", calls["inc"] == 1, f"inc={calls['inc']}")
        check("告警去重：判据本身不受影响（仍返回 False）",
              L.vision_capability(d_t) is False)
        # 有视觉塔的模型不受牵连，也不该产生告警
        n = calls["warn"]
        L.vision_capability(d_v)
        L.vision_capability(d_v)
        check("告警去重：有视觉塔的模型不误报", calls["warn"] == n, f"warn={calls['warn']}")
        # 另一个缺塔模型要能各自警告一次（去重是按模型记账，不是全局一次性）
        d_t2 = make_model(tmp, "pub-c/vlm-model", MODAL_CFG, TEXT_KEYS)
        L.vision_capability(d_t2)
        check("告警去重：换个模型仍会警告", calls["warn"] == n + 1, f"warn={calls['warn']}")
        L.vision_capability(d_t2)
        check("告警去重：第二个模型也只警告一次", calls["warn"] == n + 1, f"warn={calls['warn']}")
        check("告警去重：去重集合按模型分别记账",
              d_t in L._vision_missing_warned and d_t2 in L._vision_missing_warned)
        check("告警去重：有视觉塔的模型不进集合", d_v not in L._vision_missing_warned)
    finally:
        L.logger.warning, L.metrics.emit, L.metrics.inc = real_warn, real_emit, real_inc

    print(f"\n通过 {sum(PASS)}/{len(PASS)}")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())


