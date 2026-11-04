# 天宫医疗智能体（tiangong-agent）

基于 FastAPI + LangGraph/LangChain 的医疗领域 Agent 项目，覆盖智慧问诊、报告解读、药物咨询、知识文档和运营数据五类场景。

> 当前定位：**可运行的领域 Agent MVP / 内部验证项目**，还不是生产级医疗系统。

## 当前能力

- **智慧问诊 Agent**：多轮症状采集、置信度评估、科室推荐、挂号移交
- **知识文档 Agent**：MinerU 解析 PDF → 页码切块 → Embedding → Milvus → RAG 问答
- **药物 Agent**：药品说明书检索、药物-疾病关系、药物相互作用、处方审核
- **运营数据 Agent**：自然语言 → SQL → PostgreSQL → 数据回答
- **报告解读 Agent**：当前为模拟实现，尚未接入 MinerU/视觉模型
- **多通道检索**：文档 RAG + GraphRAG + NL2SQL 融合
- **质量增强**：Query 改写、HyDE、Reranker、幻觉检测、引用页码

## 技术栈

| 层 | 技术 |
|---|---|
| Web | FastAPI、Uvicorn |
| Agent | LangGraph、LangChain、DeepSeek |
| 文档解析 | MinerU 云 API（mineru.net v4） |
| 向量库 | Milvus 2.6 |
| 图数据库 | Neo4j 5 |
| 关系库 | PostgreSQL 17 |
| 缓存/会话 | Redis Stack |
| 对象存储 | MinIO |
| Embedding | DashScope text-embedding-v3 / 本地 BGE |

## 架构

```text
FastAPI
  └─ Supervisor Agent
       ├─ Inquiry Agent      问诊
       ├─ Report Agent       报告解读（模拟）
       ├─ Drug Agent         药物咨询 + 处方审核
       ├─ Knowledge Agent    文档 RAG / GraphRAG / NL2SQL / 多通道
       └─ Operation Agent    NL2SQL 运营数据

Knowledge Agent
  ├─ search_knowledge_docs    文档 RAG
  ├─ search_knowledge_graph   GraphRAG
  ├─ search_knowledge_sql     NL2SQL
  ├─ search_knowledge_multi   多通道融合
  └─ review_prescription_tool 处方审核
```

## 目录结构

```text
src/
  api/                 FastAPI 路由
  agents/
    supervisor_agent.py
    workers/           问诊/报告/药物/知识/运营 Agent
    knowledge/         文档入库、RAG、GraphRAG、NL2SQL、处方审核
    tools/             Supervisor 工具与上下文
  infra/               Milvus/Neo4j/Redis/MinIO/Embedding
  modules/             SQLAlchemy 数据模型
scripts/               初始化与入库脚本
alembic/               数据库迁移
test/                  测试
docker-compose.yml     基础设施
```

## 快速开始

### 1. 创建环境

```bash
conda create -n tiangong-agent python=3.13
conda activate tiangong-agent
pip install -r requirements.txt
```

### 2. 启动基础设施

```bash
docker compose up -d
docker compose ps
```

包含：

- PostgreSQL `localhost:5432`
- Redis Stack `localhost:6379`
- MinIO `localhost:9000` / 控制台 `9001`
- Milvus `localhost:19530`
- Neo4j `localhost:7474` / `7687`

### 3. 配置环境变量

复制 `.env.example` 为 `.env`（如果仓库暂时没有 `.env.example`，请手动创建 `.env`），至少配置：

```ini
DEEPSEEK_API_KEY=...
DASHSCOPE_API_KEY=...

# MinerU 云解析
MINERU_API_BASE=https://mineru.net
MINERU_API_TOKEN=...
MINERU_MODEL_VERSION=vlm
MINERU_LANGUAGE=ch
MINERU_POLL_TIMEOUT=900

# Embedding 后端
EMBEDDING_BACKEND=dashscope
EMBEDDING_MODEL=text-embedding-v3
```

`.env` 已在 `.gitignore` 中，不要提交。

### 4. 数据库迁移

```bash
alembic upgrade head
```

### 5. 初始化数据（按需）

```bash
# 结构化基础数据
python -m scripts.init_postgres

# 医学知识图谱
python -m scripts.init_neo4j

# 症状向量索引
python -m scripts.init_symptom_index

# 药品说明书 → Milvus
python -m scripts.init_knowledge_docs
```

具体参数以各脚本头部说明为准。

### 6. 启动 API

```bash
uvicorn src.main:app --port 8080 --reload
```

打开：

- Swagger：<http://localhost:8080/docs>
- 健康检查：<http://localhost:8080/health>
- 依赖检查：<http://localhost:8080/health/deps>

## API

### 非流式对话

```http
POST /api/v1/chat
Content-Type: application/json

{
  "user_id": "u1",
  "session_id": "s1",
  "message": "糖尿病的常用药有哪些？"
}
```

响应：

```json
{
  "reply": "...",
  "session_id": "s1"
}
```

### 流式对话（SSE）

```http
POST /api/v1/chat/stream
Content-Type: application/json

{
  "user_id": "u1",
  "session_id": "s1",
  "message": "高血压患者平时要注意什么？"
}
```

输出：

```text
data: {"type":"token","content":"..."}
data: {"type":"done","session_id":"s1"}
```

## 知识库入库

### 本地 PDF

```bash
python -m scripts.ingest_knowledge \
  --file data/raw/guide.pdf \
  --doc-type guideline \
  --category 心内科
```

### MinIO 对象

```bash
python -m scripts.ingest_knowledge \
  --object guidelines/guide.pdf \
  --doc-type guideline
```

### 扫描件

```bash
python -m scripts.ingest_knowledge \
  --object guidelines/scan.pdf \
  --doc-type guideline \
  --ocr
```

入库流程：

```text
PDF → MinerU 云解析 → content_list.json → 按页切块
→ Embedding → Milvus knowledge_docs → flush
```

`content_list.json` 提供 `page_idx`，因此检索结果可以引用真实页码。

## Embedding 配置

默认使用 DashScope：

```ini
EMBEDDING_BACKEND=dashscope
EMBEDDING_MODEL=text-embedding-v3
```

本地 BGE：

```ini
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=models/bge-large-zh-v1.5
LOCAL_EMBEDDING_DEVICE=cpu
```

启用本地 BGE 需要额外安装：

```bash
pip install sentence-transformers
```

> 切换 Embedding 后端后，必须重新入库 `knowledge_docs` 和长期记忆，因为向量空间不同。

## 测试与检查

```bash
python -m compileall -q src scripts test
pytest
```

当前测试以手工 smoke test 为主，完整自动化测试仍在补充中。

## 已知限制

- **没有登录和 RBAC**：`user_id/role` 目前由请求方传入，生产环境必须接统一认证。
- **PII 风险**：NL2SQL 只做了基础黑名单，仍可能通过 `SELECT *` 返回敏感字段。
- **GraphRAG 写风险**：LLM 生成的 Cypher 尚未做只读白名单，建议使用只读 Neo4j 账号。
- **报告解读 Agent** 仍是模拟实现，未接 MinerU/视觉模型。
- **反馈/通知/知识对话上下文** 已有代码，但尚未暴露 API。
- **MinerU 任务恢复**：超时后保留 `batch_id`，但还没有持久化任务表。
- **本地 BGE** 依赖 `sentence-transformers`，当前未写入 `requirements.txt`。

## 安全与合规

- 患者报告、病历、影像等敏感数据**不要上传公有云 MinerU**，应使用私有化部署或先脱敏。
- `.env` 不提交 Git；Token 如曾硬编码或泄露，请立即轮换。
- 生产环境需要补充：登录鉴权、角色权限、审计、PII 脱敏、限流和成本控制。