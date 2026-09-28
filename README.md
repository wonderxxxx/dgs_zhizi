# 千夜智子 · 自研工程

人格核心优先，界面壳后置。当前形态：命令行对话 + HTTP JSON API（壳子对接层），
用于验证「像不像智子」与「记不记得住事」。

## 目录结构

```
zhizi/
├── config.yaml            # 模型/记忆/提示词配置（模型无关的关键）
├── prompts/
│   └── zhizi_v1.md        # 自研版 System Prompt（人格层唯一事实来源）
├── core/
│   ├── llm.py             # 统一调用层（API/local 走 OpenAI 协议；mlx 进程内推理）
│   ├── metrics.py         # 运行指标 + 结构化日志（/metrics、SSE 事件源）
│   ├── memory.py          # 观测笔记 v2：SQLite 持久化 + BM25 倒排检索 + 重要度 + 去重
│   ├── persona.py         # 人格装配：System Prompt + 记忆 + 历史 + 结构化回复
│   ├── characters.py      # 角色注册表：prompt 读取（含字面量包装）+ 能力开关
│   ├── visual_identity.py # 自视身份层：VLM 裸描述 → 属性匹配 → 「图片中可能是我」
│   ├── actions.py         # 动作/神情描写剥离（按需隐藏）
│   ├── users.py           # 用户级 + 角色级隔离：每 (user, char) 独立 Persona + 并发锁
│   └── router.py          # 路由层（后置占位：小模型语域/插件判断）
├── visual/                # 自视身份卡 + 参考图（模型自识别的视觉基准）
│   ├── identity.json      # 角色视觉特征（发色/瞳色/画风/年龄段/参考图清单）
│   └── references/        # 参考图（可选：放对应图片后开启「视觉比对」加权）
├── static/
│   └── dashboard.html     # 实时监控单页（GET /dashboard）
├── logs/                  # 结构化日志（observability.log_file，自动生成）
├── notes/                 # 观测笔记持久化（自动生成）
├── api.py                 # HTTP JSON API（Flutter/Swift/Web 壳子对接）
├── main.py                # 命令行对话入口（--no-actions 可隐藏动作描写）
└── requirements.txt
```

## 快速开始

### 方式 A：uv（推荐，依赖隔离 + 版本锁定）

```bash
cd zhizi
uv init --bare        # 首次：生成 pyproject.toml（工程已内置）
uv add openai pyyaml  # 首次：创建 .venv 并安装依赖（工程已内置 uv.lock）
uv run python main.py
```

密钥两种配法（任选）：
- 环境变量：`export ARK_API_KEY=你的密钥`
- `.env` 文件：在 zhizi/ 下建 `.env` 写 `ARK_API_KEY=你的密钥`，`uv run` 会自动加载

以后加依赖（如升级记忆向量库）：`uv add chromadb`，不需要手动管虚拟环境。

### 方式 B：pip

```bash
cd zhizi
pip install -r requirements.txt
export ARK_API_KEY=你的密钥
python main.py
```

### 方式 C：快速临时跑（不落依赖，uv 特色）

```bash
cd zhizi
uv run --with openai --with pyyaml python main.py
```

### 本地模型切换（三种方式通用）

改 `config.yaml`：`provider.type` 改 `local`，`base_url` 改 `http://localhost:11434/v1`，`model` 改 `qwen2.5:7b`。

## HTTP API（Flutter / Swift / Web 壳子对接）

```bash
uv run python api.py --port 8765
```

```bash
# 对话（默认返回正文 + 动作列表；user_id 缺省落 "default"）
curl -X POST http://127.0.0.1:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "哥哥回来了吗"}'
# → {"reply": "正文", "actions": ["动作1"], "raw": "原文", "user_id": "default"}

# 用户级隔离：同一 user_id 跨设备/会话共享记忆；不同 user_id 完全隔离
curl -X POST http://127.0.0.1:8765/chat \
  -d '{"message": "还记得我们的暗号吗", "user_id": "alice"}'
# → 换个设备用同一个 user_id 也能接上记忆；bob 看不到 alice 的任何内容

# 角色级切换：一级用户 / 二级角色，角色互相隔离（人格 + 记忆）
curl -X POST http://127.0.0.1:8765/chat \
  -d '{"message": "我现在在危险区，怎么撤离", "user_id": "alice", "character": "白鸥"}'
curl http://127.0.0.1:8765/characters
# → {"default": "千夜智子", "characters": [{"name": "千夜智子", ...}, {"name": "白鸥", "memory": false, ...}]}
# 未指定 character 时落到 config.characters 的默认角色（沿用旧目录 notes/<user_id>/）；
# 非默认角色落在 notes/<user_id>/<角色名>/，彼此不可见。

# 已知用户清单（notes 落盘目录 ∪ 内存活跃键）——聊天页/记忆页的用户下拉框数据源
curl http://127.0.0.1:8765/users
# → {"default": "default", "users": ["default", "dgs", "dgs2", ...]}

# 聊天记录回填：同一 user_id × character 打开会话即拉到此前全部往来（鉴权同 /chat）
curl "http://127.0.0.1:8765/history?user_id=alice&character=千夜智子&limit=200"
# → {"user_id":"alice","character":"千夜智子","persisted":true,
#    "messages":[{"role":"user","content":"…","actions":[],"time":"2026-09-20T21:00:00"},
#                {"role":"assistant","content":"正文","actions":["（笑）"],"time":"…"}]}
# 记录落在 notes/<user_id>/<角色>/memory/recent/recent_default.json：跨设备、跨会话、
# 跨进程重启都在。persisted=false 表示该角色未启用观测笔记（memory: false），
# 记录只在进程内，重启即失。limit 缺省 100、上限 500（取最近 N 条，正序返回）。

# 图片看得见 + 记得住：发图那一轮 /history 会多两个字段，原文保持干净
#   images: [{"name":"20260928-111148-7f93bf.png","mime":"image/png"}]  字节在
#           notes/<user_id>/<角色>/images/，历史只存文件名（不进上下文）
#   caption: "一个女孩穿着白裙站在樱花树下"  图片内容说明，随该轮进五维记忆
curl "http://127.0.0.1:8765/history?user_id=alice&character=千夜智子"
# 拿回原图（鉴权同 /history；name 必须是不含分隔符的纯文件名）
curl -o out.png "http://127.0.0.1:8765/image?user_id=alice&character=千夜智子&name=20260928-111148-7f93bf.png"
# 路径穿越（../ 、 x/y.png）一律 404。说明由 VLM 生成：有身份卡时复用「自识别」那次
# 描述（零额外视觉调用），无卡时单独描述一次，失败降级为空。attachments.caption=false
# 可关掉（只记 [图片]，图仍然落盘可回看）。清记忆会连带清掉该用户的图片。

# 隐藏动作/神情描写（只拿正文）
curl -X POST http://127.0.0.1:8765/chat \
  -d '{"message": "哥哥回来了吗", "user_id": "alice", "hide_actions": true}'
# → {"reply": "正文", "raw": "原文", "user_id": "alice"}

# 重新生成：换模型后就同一句话再要一个回答（鉴权同 /chat）
# 丢掉工作记忆里那条旧回复再重问，历史不会多出一条重复提问；
# 带图轮次的图从落盘读回再喂给模型，引用与说明原样保留。
# 生成失败则上一轮原样放回，不会让你丢了一句话。
curl -X POST http://127.0.0.1:8765/regenerate \
  -d '{"user_id": "alice"}'
# → {"reply": "正文", "actions": ["动作1"], "raw": "原文", "user_id": "alice"}
# 没有一问一答时 400。换模型用 POST /model {"model": "<路径>"}，再调本接口。
# 聊天页里她最后一条回答下方有「重新生成」按钮。

curl http://127.0.0.1:8765/health
# → {"status": "ok", "model": "...", "users": N, "notes": M}

# 自视身份卡（模型自识别基准，GET 观测用）
curl http://127.0.0.1:8765/identity
# → {"enabled": true, "threshold": 0.55, "identity": {"character": "千夜智子", "visual_identity": {...}}}

# 发图对话：贴图后智子会先做「自识别」，置信度过阈值才注入“图片中可能是我”
curl -X POST http://127.0.0.1:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你看这张图", "user_id": "default", "images": [{"mime": "image/png", "data": "<base64>"}]}'
# → {"reply": "...", "raw": "...", "image_acked": true,
#    "attachments": {"images": [{"name": "20260928-111148-7f93bf.png", "mime": "image/png"}],
#                    "caption": "一个女孩穿着白裙站在樱花树下"}}

# 运行监控
curl http://127.0.0.1:8765/metrics   # 指标 JSON（计数器/延迟/tok/s/事件）
# 浏览器打开实时页：
open http://127.0.0.1:8765/dashboard
# SSE 事件流（可接自定义面板）：GET /events

# 记忆观察（查看五维记忆 + 召回过程）
open http://127.0.0.1:8765/memory          # 记忆观察页（user_id 隔离）
curl "http://127.0.0.1:8765/memory/api?user_id=alice"   # 五维记忆全量 + 最近召回 trace
curl -X POST http://127.0.0.1:8765/memory/search \
  -d '{"user_id": "alice", "query": "薯片", "top_k": 5}'
# → 该查询的召回过程明细（tokens / 各维度命中项 + BM25 分 + 命中词）

# 清空某用户全部记忆（不可恢复；鉴权同 /chat，服务端启动时可留空）
curl -X POST http://127.0.0.1:8765/memory/clear \
  -d '{"user_id": "alice"}'
# → {"user_id":"alice","cleared":true,"recent":N,"timeindex":N,"facts":N,"reflections":N,"persona":{...}}
```

- **动作描写约定**：System Prompt 已规范智子回复中动作/神情统一用全角括号（…）包裹，
  服务端按此剥离——壳子侧显示/隐藏只是取哪个字段的问题。
- **用户级隔离**：`user_id` 是隔离维度（缺省 `default`）。同一用户的「会话历史 + 观测笔记」
  跨设备、跨会话共享——她记得你，不管你在哪台设备上；不同用户之间互不可见
  （笔记落盘到 `notes/<user_id>/`）。服务进程内最多常驻 64 个用户，超出按 FIFO 驱逐
  （笔记已落盘，不丢数据）。
- **角色级隔离（二级）**：`character` 是第二维（缺省默认角色）。每个角色独立 System Prompt
  与记忆命名空间——默认角色沿用 `notes/<user_id>/` 一级目录，其余角色落盘到
  `notes/<user_id>/<角色名>/`，互不串味。角色能力可开关（`config.yaml` 的 `characters` 段）：
  `memory` 关掉即不主动观测用户（观测笔记是智子的专属能力），`visual_identity` 留空即无
  自视身份卡（白鸥如此）。人物卡支持纯文本/Markdown 或 `SYS_PROMPT = r'''...'''` 包装
  （无需改卡，零修改接入）。
- **鉴权（可选）**：`config.yaml` 的 `server.api_key` 非空时，`/chat` 与
  `/regenerate` 需带请求头 `Authorization: Bearer <api_key>`（`/health` 放行，方便探活）。
  `/history`（聊天记录是私事）与 `/memory/clear` 同样要求鉴权。
- **壳子只做 UI**：人格、记忆、红线全在服务端，换壳不动人格。
- **聊天记录回填**：`GET /history`（`user_id` × `character`）是该会话的「历史真相」——
  壳子打开会话、切换用户或切换角色时调一次，把 `messages` 灌进界面即可续上上下文；
  同一对不重复拉（`static/chat.html` 按 `user\x00character` 键去重）。

## 设计要点

- **模型无关**：云端 API 与本地模型都走 OpenAI 兼容协议，切换只改 `config.yaml`。
- **可观测**：`core/metrics.py` 采集聊天轮次、LLM 延迟、mlx tok/s 与峰值内存、记忆耗时/错误；日志到 stderr + `logs/zhizi.log`；`GET /metrics`、`GET /events`（SSE）、`GET /dashboard` 实时页（`observability` 段配置）。
- **观测笔记 = 长期记忆（v2）**：每轮对话后由智子自己用科研口吻写一条笔记
  （角色化存储），自评「重要度 1-10」，相同内容自动去重（只更新不新增）；
  下次聊到相关内容时按 BM25 打分注入——她真的"记得"，且越重要的记得越牢。
  存储为 SQLite（`notes/<user_id>/notes.db`），旧 JSONL 首次启动自动迁移（原文件
  保留为 `notes.jsonl.migrated`）。
- **短期窗口**：会话历史上限 40 条，防止上下文膨胀。
- **自视身份层（Self-Identity）**：带 VLM 的模型"认不出自己"时，系统不依赖模型自觉，而是走
  一条确定性的外挂识别链——用户图片 → 中立 VLM 裸描述（不喂身份信息，避免确认偏误）→
  属性匹配（visual/identity.json：发色/瞳色/画风/年龄段加权比对）→ 置信度 →
  超阈值才注入「图片中可能是我」的 system 上下文，让智子按性格自然接话；
  有参考图时叠加一张「视觉比对」加权融合（0.6 属性 + 0.4 比对）。粒度见 /metrics 的
  identity_* 事件；任何一步失败都静默降级，不拖垮对话（config: visual_identity 段）。
- **路由层后置**：默认关闭，小模型语域/插件判断后续接入；当前人格切换完全由主模型
  按 System Prompt 自主执行，不注入外部指令。
- **红线已在 System Prompt 内置**：模糊边界（不暴露技术本质）、呈现克制（谜团/渴望
  不念叨）、中二设定低频触发、同居在场场景、无固定例句（防复读）、动作描写统一括号化。

## 下一步（按序）

1. 跑通对话 → 2. ✅ 记忆升级 P0（SQLite + BM25 + 重要度 + 去重，已落地）→
3. P1 反思层（定期把笔记综合成洞察：事实→反思→人格）→ 4. P2 中二/博士状态机
（借鉴 N.E.K.O. CognitionMode）→ 5. 接入路由小模型 → 6. Plugin 能力层
（RVC 语音、图像生成等，单独排期）→ 7. 界面壳（API 已就位，Flutter/Swift 壳
直接 POST /chat 即可，动作描写按需取字段）。
