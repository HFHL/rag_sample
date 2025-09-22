import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


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


def build_embedder(cfg: dict):
    from sentence_transformers import SentenceTransformer  # type: ignore

    model_name = str(cfg.get("embedding", {}).get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    device = resolve_device(cfg)
    model = SentenceTransformer(model_name, device=device)
    return model, device


def list_chunk_files(chunks_dir: str) -> List[str]:
    return [
        os.path.join(chunks_dir, f)
        for f in os.listdir(chunks_dir)
        if f.lower().endswith(".json") and os.path.isfile(os.path.join(chunks_dir, f))
    ]


def load_chunks(file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], dict) and "metadata" in data[0]:
        meta = data[0]["metadata"]
        chunks = data[1:]
        return meta, chunks
    # fallback: old format with {document, chunks}
    if isinstance(data, dict) and "document" in data and "chunks" in data:
        meta = data.get("document", {})
        chunks = data.get("chunks", [])
        # map to unified format
        mapped = []
        for ch in chunks:
            mapped.append({
                "text": ch.get("text", ""),
                "metadata": ch.get("metadata", {}),
            })
        return meta, mapped
    raise ValueError(f"Unrecognized chunk file format: {file_path}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_collection(client, name: str):
    # Try to create with cosine space; ignore if not supported in version
    try:
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    except Exception:
        return client.get_or_create_collection(name=name)


def upsert_chunks(
    coll,
    school_name: str,
    chunks: List[Dict[str, Any]],
    model,
    batch_size: int,
) -> int:
    total = 0
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    ids: List[str] = []

    def flush_batch() -> None:
        nonlocal total, docs, metas, ids
        if not docs:
            return
        vectors = model.encode(docs, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=False)
        # chroma expects lists
        embeddings = [v.tolist() for v in vectors]
        coll.add(documents=docs, metadatas=metas, ids=ids, embeddings=embeddings)
        total += len(docs)
        docs, metas, ids = [], [], []

    for idx, ch in enumerate(chunks):
        text = ch.get("text", "").strip()
        if not text:
            continue
        meta = ch.get("metadata", {}) or {}
        meta = dict(meta)
        meta.setdefault("school_name", school_name)
        chunk_id = f"{school_name}::chunk::{idx}"

        docs.append(text)
        metas.append(meta)
        ids.append(chunk_id)

        if len(docs) >= batch_size:
            flush_batch()

    flush_batch()
    return total


def main(argv: List[str]) -> int:
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)
    cfg = load_config(project_root)

    chunks_dir = os.path.join(project_root, cfg.get("output_dir", "data/chunk"))
    if not os.path.isdir(chunks_dir):
        print(f"Chunks directory not found: {chunks_dir}")
        return 1

    chroma_dir = os.path.join(project_root, cfg.get("chroma", {}).get("persist_dir", "data/chroma_db"))
    collection_name = str(cfg.get("chroma", {}).get("collection_name", "schools"))
    batch_size = int(cfg.get("embedding", {}).get("batch_size", 64))

    model, device = build_embedder(cfg)
    print(f"Embedding model loaded on device: {device}")

    import chromadb  # type: ignore
    client = chromadb.PersistentClient(path=chroma_dir)
    coll = get_collection(client, collection_name)

    files = list_chunk_files(chunks_dir)
    if not files:
        print("No chunk JSON files found.")
        return 0

    total_added = 0
    for fp in files:
        try:
            meta, chunks = load_chunks(fp)
            school_name = str(meta.get("school_name") or os.path.splitext(os.path.basename(fp))[0])
            added = upsert_chunks(coll, school_name, chunks, model, batch_size)
            total_added += added
            print(f"Indexed {added} chunks from {school_name}")
        except Exception as e:
            print(f"Failed to index {fp}: {e}")

    print(f"Done. Total chunks indexed: {total_added}. Chroma path: {chroma_dir}, collection: {collection_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


