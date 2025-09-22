import json
import os
from typing import Any, Dict, List, Tuple
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer


def load_config(project_root: str) -> dict:
    yaml_path = os.path.join(project_root, "rag_config.yaml")
    yml_path = os.path.join(project_root, "rag_config.yml")
    json_path = os.path.join(project_root, "rag_config.json")
    if os.path.exists(yaml_path):
        path = yaml_path
    elif os.path.exists(yml_path):
        path = yml_path
    elif os.path.exists(json_path):
        path = json_path
    else:
        raise FileNotFoundError("Config not found. Expected one of: rag_config.yaml, rag_config.yml, rag_config.json")

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            if yaml is None:
                raise RuntimeError("PyYAML is required. Install with: python -m pip install pyyaml")
            return yaml.safe_load(f) or {}
        return json.load(f)


def load_prompt(project_root: str, cfg: dict) -> str:
    rel_path = cfg.get("server", {}).get("prompt_path", "prompts/answer_prompt.txt")
    path = os.path.join(project_root, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def resolve_device(cfg: dict) -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    want_gpu = bool(cfg.get("embedding", {}).get("use_gpu", True))
    requested = str(cfg.get("embedding", {}).get("device", "auto")).lower()
    if requested == "cpu":
        return "cpu"
    if want_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Retriever:
    def __init__(self, project_root: str, cfg: dict) -> None:
        self.project_root = project_root
        self.cfg = cfg
        self.embedding_model_name = str(cfg.get("embedding", {}).get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
        self.device = resolve_device(cfg)
        self.model = SentenceTransformer(self.embedding_model_name, device=self.device)

        self.batch_size = int(cfg.get("embedding", {}).get("batch_size", 64))
        self.faiss_top_k = int(cfg.get("retrieval", {}).get("faiss_top_k", 10))
        self.keyword_top_k = int(cfg.get("retrieval", {}).get("keyword_top_k", 10))

        self.chunks: List[Dict[str, Any]] = []
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []
        self.chunk_metas: List[Dict[str, Any]] = []

        self.faiss_index = None
        self.embeddings: np.ndarray | None = None
        self.tfidf = None
        self.tfidf_matrix = None

    def _load_chunks(self) -> None:
        chunks_dir = os.path.join(self.project_root, self.cfg.get("output_dir", "data/chunk"))
        files = [
            os.path.join(chunks_dir, f)
            for f in os.listdir(chunks_dir)
            if f.lower().endswith(".json")
        ]
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            # list format: [ {metadata: {...}}, {text, metadata}, ... ]
            if isinstance(data, list) and data and isinstance(data[0], dict) and "metadata" in data[0]:
                school_name = str(data[0]["metadata"].get("school_name") or os.path.splitext(os.path.basename(fp))[0])
                for idx, ch in enumerate(data[1:]):
                    text = str(ch.get("text", ""))
                    meta = dict(ch.get("metadata", {}))
                    meta.setdefault("school_name", school_name)
                    cid = f"{school_name}::chunk::{idx}"
                    self.chunk_ids.append(cid)
                    self.chunk_texts.append(text)
                    self.chunk_metas.append(meta)
            # dict format: {document, chunks}
            elif isinstance(data, dict) and "document" in data and "chunks" in data:
                school_name = str(data.get("document", {}).get("source", {}).get("file_name") or os.path.splitext(os.path.basename(fp))[0])
                for idx, ch in enumerate(data.get("chunks", [])):
                    text = str(ch.get("text", ""))
                    meta = dict(ch.get("metadata", {}))
                    meta.setdefault("school_name", school_name)
                    cid = f"{school_name}::chunk::{idx}"
                    self.chunk_ids.append(cid)
                    self.chunk_texts.append(text)
                    self.chunk_metas.append(meta)

    def _build_faiss(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install with: pip install faiss-cpu")
        if not self.chunk_texts:
            return
        vectors = self.model.encode(self.chunk_texts, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True)
        mat = np.asarray(vectors, dtype=np.float32)
        dim = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        self.faiss_index = index
        self.embeddings = mat

    def _build_tfidf(self) -> None:
        if not self.chunk_texts:
            return
        max_features = int(self.cfg.get("retrieval", {}).get("tfidf_max_features", 50000))
        self.tfidf = TfidfVectorizer(max_features=max_features)
        self.tfidf_matrix = self.tfidf.fit_transform(self.chunk_texts)

    def initialize(self) -> None:
        self._load_chunks()
        self._build_faiss()
        self._build_tfidf()

    def search_faiss(self, query: str, top_k: int | None = None) -> List[Tuple[int, float]]:
        if self.faiss_index is None:
            return []
        k = top_k or self.faiss_top_k
        qvec = self.model.encode([query], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
        q = np.asarray(qvec, dtype=np.float32)
        scores, idxs = self.faiss_index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def search_keyword(self, query: str, top_k: int | None = None) -> List[Tuple[int, float]]:
        if self.tfidf is None or self.tfidf_matrix is None:
            return []
        k = top_k or self.keyword_top_k
        qv = self.tfidf.transform([query])
        sims = linear_kernel(qv, self.tfidf_matrix).ravel()
        if k >= len(sims):
            idxs = np.argsort(-sims)
        else:
            idxs = np.argpartition(-sims, k)[:k]
            idxs = idxs[np.argsort(-sims[idxs])]
        return [(int(i), float(sims[i])) for i in idxs[:k]]

    def merge_results(self, faiss_res: List[Tuple[int, float]], kw_res: List[Tuple[int, float]], limit: int) -> List[int]:
        score_map: Dict[int, float] = {}
        # Normalize scores roughly via rank-based aggregation
        for rank, (i, s) in enumerate(faiss_res):
            score_map[i] = score_map.get(i, 0.0) + 1.0 / (1 + rank)
        for rank, (i, s) in enumerate(kw_res):
            score_map[i] = score_map.get(i, 0.0) + 1.0 / (1 + rank)
        sorted_idx = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
        return sorted_idx[:limit]


class LLMClient:
    def __init__(self) -> None:
        load_dotenv()
        self.base_url = os.getenv("LLM_BASE_URL", "")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "")
        if not self.base_url or not self.model:
            raise RuntimeError("Missing LLM_BASE_URL or LLM_MODEL in environment variables")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package is required. Install with: pip install openai") from e
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


def build_prompt(template: str, question: str, contexts: List[Dict[str, Any]]) -> Tuple[str, str]:
    # The template contains placeholder markers: {question} and {contexts}
    # Only include school and chunk text per context.
    blocks: List[str] = []
    for _, c in enumerate(contexts, start=1):
        meta = c.get("metadata", {})
        school = meta.get("school_name", "")
        lines = [
            f"school: {school}",
            c.get('text', ''),
        ]
        blocks.append("\n".join(lines))
    ctx_text = "\n\n".join(blocks)
    filled = template.replace("{question}", question).replace("{contexts}", ctx_text)
    # Split first line as system, rest as user if template provides a marker
    system_prompt = "你是一个严谨的中文助理，基于提供的检索片段回答用户问题。"
    user_prompt = filled
    return system_prompt, user_prompt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _log_file_path(project_root: str, cfg: dict) -> str:
    log_cfg = cfg.get("logging", {})
    logs_dir = os.path.join(project_root, str(log_cfg.get("dir", "logs")))
    _ensure_dir(logs_dir)
    prefix = str(log_cfg.get("filename_prefix", "rag_server"))
    date_str = datetime.utcnow().strftime("%Y%m%d")
    return os.path.join(logs_dir, f"{prefix}-{date_str}.jsonl")


def _append_jsonl(path: str, obj: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        # best-effort, do not raise
        pass


def create_app() -> FastAPI:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(project_root)
    prompt_template = load_prompt(project_root, cfg)

    app = FastAPI()

    templates_dir = os.path.join(project_root, cfg.get("server", {}).get("templates_dir", "server/templates"))
    static_dir = os.path.join(project_root, cfg.get("server", {}).get("static_dir", "server/static"))
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=templates_dir)

    retriever = Retriever(project_root, cfg)
    llm_client = LLMClient()

    @app.on_event("startup")
    def _startup() -> None:
        retriever.initialize()

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/api/chat")
    async def chat(payload: Dict[str, Any]) -> JSONResponse:
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        question = str(payload.get("query", "")).strip()
        if not question:
            return JSONResponse({"error": "empty query", "request_id": request_id}, status_code=400)

        log_path = _log_file_path(project_root, cfg)
        include_text = bool(cfg.get("logging", {}).get("include_context_text", True))
        max_chars = int(cfg.get("logging", {}).get("max_context_chars", 500))

        try:
            top_k = int(payload.get("top_k", max(retriever.faiss_top_k, retriever.keyword_top_k)))
            t_ret = time.perf_counter()
            faiss_hits = retriever.search_faiss(question, retriever.faiss_top_k)
            kw_hits = retriever.search_keyword(question, retriever.keyword_top_k)
            merged_idx = retriever.merge_results(faiss_hits, kw_hits, top_k)
            retrieval_ms = int((time.perf_counter() - t_ret) * 1000)

            contexts: List[Dict[str, Any]] = []
            for i in merged_idx:
                text_full = retriever.chunk_texts[i]
                text = text_full
                if include_text and max_chars > 0 and len(text_full) > max_chars:
                    text = text_full[:max_chars]
                contexts.append({
                    "id": retriever.chunk_ids[i],
                    "text": text,
                    "metadata": retriever.chunk_metas[i],
                })

            system_prompt, user_prompt = build_prompt(prompt_template, question, contexts)
            temp = float(cfg.get("server", {}).get("temperature", 1))
            t_llm = time.perf_counter()
            answer = llm_client.chat(system_prompt, user_prompt, temperature=temp)
            llm_ms = int((time.perf_counter() - t_llm) * 1000)
            total_ms = int((time.perf_counter() - t0) * 1000)

            # Build structured logs
            faiss_log = [
                {"idx": i, "score": s, "id": retriever.chunk_ids[i], "school": retriever.chunk_metas[i].get("school_name")}
                for i, s in faiss_hits
            ]
            kw_log = [
                {"idx": i, "score": s, "id": retriever.chunk_ids[i], "school": retriever.chunk_metas[i].get("school_name")}
                for i, s in kw_hits
            ]
            ctx_log = [
                {
                    "id": c["id"],
                    "school": c["metadata"].get("school_name"),
                    "num_chars": len(c["text"]),
                    **({"text": c["text"]} if include_text else {}),
                }
                for c in contexts
            ]
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_id": request_id,
                "query": question,
                "retrieval": {
                    "faiss_hits": faiss_log,
                    "keyword_hits": kw_log,
                    "merged_indices": merged_idx,
                    "retrieval_ms": retrieval_ms,
                },
                "llm": {
                    "model": llm_client.model,
                    "temperature": temp,
                },
                "contexts": ctx_log,
                "answer": answer,
                "timing_ms": {"retrieval": retrieval_ms, "llm": llm_ms, "total": total_ms},
                "status": "ok",
            }
            _append_jsonl(log_path, log_entry)

            return JSONResponse({
                "answer": answer,
                "contexts": contexts,
                "request_id": request_id,
                "timing_ms": {"retrieval": retrieval_ms, "llm": llm_ms, "total": total_ms},
            })
        except Exception as e:
            total_ms = int((time.perf_counter() - t0) * 1000)
            _append_jsonl(
                log_path,
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "request_id": request_id,
                    "query": question,
                    "error": str(e),
                    "timing_ms": {"total": total_ms},
                    "status": "error",
                },
            )
            return JSONResponse({"error": "internal error", "request_id": request_id}, status_code=500)

    return app


app = create_app()


