# TODO · 人设融合（Neko × zhizi）+ 智子 mood 层

> 阶段性暂停点：step 1-3 与 A 路线（mood 最小验证）都已落地、测试全绿。
> 下面记录"已完成 / 验证结论 / 候选下一步"，免得下次忘了上下文。

## 已完成 ✅

1. **角色元数据**（`core/characters.py`）：`CharacterSpec` 新增 `preview_line`（选择器预告）、
   `preset`（七字段声明式人格）、`guardrails` 开关；`GET /characters` 一并带出。
2. **公共纪律块**（`core/guardrails.py`）：不复读 / 用量克制（每轮最多一种，拿不准就不用）
   / 不讨好 / 守边界。融合 Neko 量化保守 × zhizi 现想不复用。
3. **预设渲染器**（`core/presets.py`）：`render_preset_prompt(name, preset)` 把七字段
   渲染成完整 system prompt（身份/性格原型/性格/口癖/爱好/雷点/隐藏设定/一句话台词/表达），
   缺字段段落自动跳过。`CharacterSpec.system_prompt()`：**整卡优先，预设兜底**。
4. **演示角色**（`config.yaml`）：毒舌学妹 = 纯 preset、无整卡，已验证人设立住。
5. **A 路线 · mood 层（最小验证，已落地）**：
   - `core/presets.py`：`MOOD_FIELDS`(状态/语气/行为倾向/退出线) + `normalize_mood` /
     `load_moods` / `render_mood`（状态段 → 覆盖层提示文本）。
   - `core/persona.py`：`self.moods` 装载；`_mood_block` 显式声明即生效、未知/空 → 无状态；
     `build_messages` 把状态段追加到**首条 system 末尾**（三处 build 全穿线）；
     `reply/reply_structured` 新增 `mood` 参数；状态名挂用户提问（`_msg`/`process_message`/
     `_recall_user_turn`/`_restore_last_turn`）；`regenerate` 沿用上一轮状态。
   - `config.yaml`：`moods:` 注册表（病弱/困倦/低落，均带退出线；智子卡零改动）。
   - `api.py`：POST `/chat` 接受 `"mood": "<名>"`，响应回带 `mood` 字段。
   - `tests/test_moods.py`：25 项（注入位置 / 随轮流动 / 未知降级 / 不回灌上下文 / regenerate 沿用 / HTTP 透传）。
6. 测试 10 个文件全绿。

## 关键结论（别忘）⚠️

- **智子零改动**：整卡角色不套通用纪律块；她的卡本身就是"稳定核 + 可变语域/人格面"范式。
- **万能纪律块 ≠ 整卡**：白鸥实测一轮 4 段长镜头叙事，直接顶掉"动作每轮最多一种"。
  结论：`guardrails` 只属于**预设型（无整卡）角色**；显式 `guardrails: true` 才是整卡的出路。
- **mood 纪律**（落地时已内建）：每轮至多一个（单字段）；不声明 = 无状态基线；只覆盖表达层、
  必须带退出线；mood 只挂提问元数据——不进模型历史正文、不进上下文回灌。
- 验证命令：`curl /characters`；`--character 毒舌学妹` 对话；
  `curl -X POST .../chat -d '{"message":"...","mood":"病弱"}'` 看 `mood` 回带。

## 候选下一步（按兴趣排序）

### A'. mood 层增强（最小验证已落地，这里是后续）
- 触发②router 自动判定（按轮由路由层选状态）与 ③P2 CognitionMode 状态机，尚未做。
- 壳子发现入口：`GET /characters` 暂未暴露 `moods` 清单，可加。

### B. 把 Neko 其余 3 个活跃预设注册成 demo 角色
病弱妹妹 / 知心姐姐 / 沙雕网友，直接抄 user 提供的 preset 原文填 `config.yaml`，零代码。

### C. P2 中二/博士状态机（README 原路线，借鉴 N.E.K.O. CognitionMode）
与 mood 层其实同构——状态机的"进入/维持/退出"就是 mood 的落地形态，可合并推进。

## 相关文件速查

- `core/characters.py`：`PRESET_FIELDS` / `normalize_preset` / `CharacterSpec.system_prompt()`
- `core/presets.py`：渲染器 + `MOOD_FIELDS` / `normalize_mood` / `load_moods` / `render_mood`（新）
- `core/guardrails.py`：公共纪律块（新）
- `core/persona.py`：`__init__` 走 `character.system_prompt()`；`_mood_block` / `build_messages` mood 注入
- `core/memory/manager.py`：`process_message(..., mood=)` 挂工作记忆
- `config.yaml`：`characters` 段（毒舌学妹为演示）+ `moods:` 段（病弱/困倦/低落）
- `tests/test_moods.py`：mood 层 25 项（新）