# TODO · 人设融合（Neko × zhizi）

> 阶段性暂停点：step 1-3 已落地、测试全绿。下面记录"已完成 / 验证结论 / 候选下一步"，
> 免得下次忘了上下文。

## 已完成 ✅

1. **角色元数据**（`core/characters.py`）：`CharacterSpec` 新增 `preview_line`（选择器预告）、
   `preset`（七字段声明式人格）、`guardrails` 开关；`GET /characters` 一并带出。
2. **公共纪律块**（`core/guardrails.py`）：不复读 / 用量克制（每轮最多一种，拿不准就不用）
   / 不讨好 / 守边界。融合 Neko 量化保守 × zhizi 现想不复用。
3. **预设渲染器**（`core/presets.py`）：`render_preset_prompt(name, preset)` 把七字段
   渲染成完整 system prompt（身份/性格原型/性格/口癖/爱好/雷点/隐藏设定/一句话台词/表达），
   缺字段段落自动跳过。`CharacterSpec.system_prompt()`：**整卡优先，预设兜底**。
4. **演示角色**（`config.yaml`）：毒舌学妹 = 纯 preset、无整卡，已验证人设立住。
5. 测试 9 个文件全绿。

## 关键结论（别忘）⚠️

- **智子零改动**：整卡角色不套通用纪律块；她的卡本身就是"稳定核 + 可变语域/人格面"范式。
- **通用纪律块 ≠ 整卡**：白鸥实测一轮 4 段长镜头叙事，直接顶掉"动作每轮最多一种"。
  结论：`guardrails` 只属于**预设型（无整卡）角色**；显式 `guardrails: true` 才是整卡的出路。
- 验证命令：`curl /characters`；`--character 毒舌学妹` 对话。

## 候选下一步（按兴趣排序）

### A. 智子 mood 层（上次聊到的方向，方案已成形）
目标：智子"可变量情绪 ≠ 换皮"。架构 = 稳定核心卡 + **可选 mood 状态段**（复用 preset 渲染机，
新增 mood 模式；只覆盖表达层，不动身份/红线）。
- 注入点现成：`build_messages()` 已能把 `memory / search_context / self_context` 并入首条 system。
- 纪律沿用：**每轮至多一个 mood**，防多状态叠加稀释人格。
- 触发三档：① 显式（请求传 `"mood": "病弱"`）→ ② router 自动判定 → ③ P2 CognitionMode 状态机。
- 必须克制：mood 有进入代价就得有退出条件（情绪是流动的，不能一条指令变"一辈子病弱"）。
- 最后一步：`render_preset_prompt` 加 mood-mode；先做显式触发的最小验证。

### B. 把 Neko 其余 3 个活跃预设注册成 demo 角色
病弱妹妹 / 知心姐姐 / 沙雕网友，直接抄 user 提供的 preset 原文填 `config.yaml`，零代码。

### C. P2 中二/博士状态机（README 原路线，借鉴 N.E.K.O. CognitionMode）
与 mood 层其实同构——状态机的"进入/维持/退出"就是 mood 的落地形态，可合并推进。

## 相关文件速查

- `core/characters.py`：`PRESET_FIELDS` / `normalize_preset` / `CharacterSpec.system_prompt()`
- `core/presets.py`：渲染器（新）
- `core/guardrails.py`：公共纪律块（新）
- `core/persona.py`：`__init__` 走 `character.system_prompt()`
- `config.yaml`：`characters` 段（毒舌学妹为演示）