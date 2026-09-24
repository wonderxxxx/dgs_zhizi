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
│   ├── actions.py         # 动作/神情描写剥离（按需隐藏）
│   ├── users.py           # 用户级隔离：每用户独立 Persona + 并发锁
│   └── router.py          # 路由层（后置占位：小模型语域/插件判断）
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

# 隐藏动作/神情描写（只拿正文）
curl -X POST http://127.0.0.1:8765/chat \
  -d '{"message": "哥哥回来了吗", "user_id": "alice", "hide_actions": true}'
# → {"reply": "正文", "raw": "原文", "user_id": "alice"}

curl http://127.0.0.1:8765/health
# → {"status": "ok", "model": "...", "users": N, "notes": M}

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
- **鉴权（可选）**：`config.yaml` 的 `server.api_key` 非空时，`/chat` 需带请求头
  `Authorization: Bearer <api_key>`（`/health` 放行，方便探活）。
- **壳子只做 UI**：人格、记忆、红线全在服务端，换壳不动人格。

## 设计要点

- **模型无关**：云端 API 与本地模型都走 OpenAI 兼容协议，切换只改 `config.yaml`。
- **可观测**：`core/metrics.py` 采集聊天轮次、LLM 延迟、mlx tok/s 与峰值内存、记忆耗时/错误；日志到 stderr + `logs/zhizi.log`；`GET /metrics`、`GET /events`（SSE）、`GET /dashboard` 实时页（`observability` 段配置）。
- **观测笔记 = 长期记忆（v2）**：每轮对话后由智子自己用科研口吻写一条笔记
  （角色化存储），自评「重要度 1-10」，相同内容自动去重（只更新不新增）；
  下次聊到相关内容时按 BM25 打分注入——她真的"记得"，且越重要的记得越牢。
  存储为 SQLite（`notes/<user_id>/notes.db`），旧 JSONL 首次启动自动迁移（原文件
  保留为 `notes.jsonl.migrated`）。
- **短期窗口**：会话历史上限 40 条，防止上下文膨胀。
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
