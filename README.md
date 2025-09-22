# RAG Sample

一个端到端的中文 RAG（检索增强生成）样例：
- 解析 MinerU 导出的 JSON，生成带 metadata 的段落级 chunk
- 可选：向量化并写入 Chroma 持久化库
- FastAPI 服务：FAISS（向量）+ TF‑IDF（关键词）双路召回 → 合并 → 交给 LLM 作答
- JSONL 结构化日志，记录检索与回答的全链路信息

## 目录结构
```
rag_sample/
  data/
    minerU_output/            # MinerU 输出（每校一个目录，auto 下含 *_content_list.json）
    chunk/                    # 生成的 per-school chunk JSON
    chroma_db/                # 可选：Chroma 持久化路径
  prompts/
    answer_prompt.txt         # 提示词模板
  scripts/
    mineru_to_chunks.py       # 解析 MinerU JSON → 生成 chunk JSON
    embed_to_chroma.py        # 可选：向量化并写入 Chroma
  server/
    main.py                   # FastAPI 服务（FAISS + TF‑IDF 检索）
    templates/
      index.html              # 简单 Web 页面
  logs/
    rag_server-YYYYMMDD.jsonl # 每日滚动的 JSONL 日志
  rag_config.yaml             # 主配置
  .env                        # LLM 环境变量（需自建）
```

## 架构概览
1) 数据准备（MinerU 流程）
- 输入：`data/minerU_output/<学校>/auto/<学校>_content_list.json`
- 运行 `python scripts/mineru_to_chunks.py` → 输出 `data/chunk/<学校>.json`
  - 第一项为全局 metadata（含 `school_name`）
  - 其余为 chunk：`{ text, metadata }`

2) 检索与问答
- 服务启动时加载 `data/chunk/*.json`，构建：
  - FAISS 向量索引（SentenceTransformer 模型）
  - TF‑IDF 关键词向量
- 查询时分别召回各 `top_k`（默认 10），做简单 rank 融合，按模板把“school + chunk 正文”交给 LLM。

3) 日志
- 每次请求写入 `logs/rag_server-YYYYMMDD.jsonl`，含 request_id、query、检索命中、用于提示的片段、LLM 配置、耗时、答案等，便于检索/审计。

4) Chroma（可选）
- `python scripts/embed_to_chroma.py`：将 `data/chunk/*` 编码写入 Chroma（默认 `data/chroma_db`）。
- 当前在线检索使用内存 FAISS + TF‑IDF，如需替换为 Chroma 检索可扩展 Retriever。

## 配置（rag_config.yaml）
- 基础
  - `output_dir`: chunk 输出目录（默认 `data/chunk`）
  - `mineru_input_dir`: MinerU 根目录（默认 `data/minerU_output`）
- 切分与归一化
  - `chunk_size` / `chunk_overlap` / `min_chunk_chars` / `max_chunk_chars`
  - `normalize_whitespace` / `paragraph_separator` / `preserve_line_breaks`
- 向量
  - `embedding.model_name`（默认 `paraphrase-multilingual-MiniLM-L12-v2`）
  - `embedding.use_gpu` / `device` / `batch_size`
- Chroma
  - `chroma.persist_dir`（默认 `data/chroma_db`）
  - `chroma.collection_name`（默认 `schools`）
- Server
  - `server.host` / `port` / `temperature` / `prompt_path` / `templates_dir` / `static_dir`
- 检索
  - `retrieval.faiss_top_k`、`retrieval.keyword_top_k`、`retrieval.tfidf_max_features`
- 日志
  - `logging.dir`（默认 `logs`）
  - `logging.filename_prefix`（默认 `rag_server`）
  - `logging.include_context_text`（默认 true）
  - `logging.max_context_chars`（默认 500）

## 环境变量（.env）
在项目根创建 `.env`：
```
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-xxxx
```
或本地/自建：
```
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=Qwen2.5-7B-Instruct
LLM_API_KEY=EMPTY
```

## 安装
```
pip install -U fastapi uvicorn python-dotenv pyyaml sentence-transformers scikit-learn faiss-cpu openai chromadb
# GPU: 安装与你 CUDA 匹配的 torch 版本（可选）
```

## 使用
1) 生成 chunk
```
python scripts/mineru_to_chunks.py
```
2)（可选）写入 Chroma
```
python scripts/embed_to_chroma.py
```
3) 启动服务
```
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 1
# 开发可加 --reload
```
4) 访问
- 打开浏览器 `http://localhost:8000`

## 调优
- 降低 `faiss_top_k` / `keyword_top_k` 可加速
- 更小的向量模型或更小 `embedding.batch_size` 降低显存
- 调整 `chunk_size` / `chunk_overlap` 控制 chunk 数量

## FAQ
- 缺少 LLM 配置：检查 `.env` 的 `LLM_BASE_URL` 与 `LLM_MODEL`（必要时 `LLM_API_KEY`）
- 首次加载慢：需向量化并建索引，可降低 topK 或更换轻量模型
