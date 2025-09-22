import json
import os
import re
import sys
from datetime import datetime
from typing import List, Dict, Optional, Any

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


def normalize_text(text: str, normalize_whitespace: bool) -> str:
    t = text
    if normalize_whitespace:
        t = re.sub(r"[\t\f\r ]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def chunk_paragraphs(
    paragraphs: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    max_chunk_chars: int,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    if not paragraphs:
        return chunks

    current_texts: List[str] = []
    current_pidxs: List[int] = []
    current_pages: List[int] = []
    current_mineru_idxs: List[int] = []
    current_len = 0

    def flush_chunk() -> Optional[Dict[str, Any]]:
        nonlocal current_texts, current_pidxs, current_pages, current_mineru_idxs, current_len
        if not current_texts:
            return None
        text = "\n\n".join(current_texts)
        if len(text) < min_chunk_chars and chunks:
            prev = chunks[-1]
            merged = prev["text"] + "\n\n" + text
            if len(merged) <= max_chunk_chars:
                prev["text"] = merged
                prev_meta = prev.setdefault("metadata", {})
                prev_meta["pidx_end"] = current_pidxs[-1]
                prev_meta["mineru_idx_end"] = current_mineru_idxs[-1]
                prev_meta["page_end"] = max(prev_meta.get("page_end", -1), max(current_pages))
                current_texts, current_pidxs, current_pages, current_mineru_idxs, current_len = [], [], [], [], 0
                return None

        meta = {
            "pidx_start": current_pidxs[0],
            "pidx_end": current_pidxs[-1],
            "mineru_idx_start": current_mineru_idxs[0],
            "mineru_idx_end": current_mineru_idxs[-1],
            "page_start": min(current_pages),
            "page_end": max(current_pages),
            "num_paragraphs": len(current_texts),
            "num_chars": len(text),
        }
        chunk = {"text": text, "metadata": meta}
        chunks.append(chunk)
        current_texts, current_pidxs, current_pages, current_mineru_idxs, current_len = [], [], [], [], 0
        return chunk

    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        body = p["text"]
        p_len = len(body) + (2 if current_texts else 0)

        if current_len + p_len <= chunk_size or not current_texts:
            current_texts.append(body)
            current_pidxs.append(p["pidx"])  # index within filtered paragraphs
            current_mineru_idxs.append(p.get("mineru_idx", p.get("idx", p["pidx"])) )
            current_pages.append(p.get("page_idx", -1))
            current_len += p_len
            i += 1
            continue

        last = flush_chunk()

        if chunk_overlap > 0 and last is not None:
            prev_num = last["metadata"]["num_paragraphs"]
            avg_chars = max(1, last["metadata"]["num_chars"] // prev_num)
            overlap_paras = max(1, min(prev_num - 1, chunk_overlap // max(1, avg_chars)))

            end_pidx = last["metadata"]["pidx_end"]
            start_pidx = max(last["metadata"]["pidx_start"], end_pidx - overlap_paras + 1)
            for oi in range(start_pidx, end_pidx + 1):
                src = paragraphs[oi]
                current_texts.append(src["text"])
                current_pidxs.append(src["pidx"])
                current_mineru_idxs.append(src.get("mineru_idx", src.get("idx", src["pidx"])) )
                current_pages.append(src.get("page_idx", -1))
                if current_len > 0:
                    current_len += 2
                current_len += len(src["text"])

        # After adding overlap, try to add the same paragraph p now
        p_len = len(body) + (2 if current_texts else 0)
        if current_len + p_len <= chunk_size:
            current_texts.append(body)
            current_pidxs.append(p["pidx"])
            current_mineru_idxs.append(p.get("mineru_idx", p.get("idx", p["pidx"])) )
            current_pages.append(p.get("page_idx", -1))
            current_len += p_len
            i += 1
            continue

        # If adding p still doesn't fit with overlap, handle two cases:
        # 1) ultra-long paragraph: hard split it (ignore overlap)
        if len(body) > max_chunk_chars:
            # drop overlap context for this special case
            current_texts.clear()
            current_pidxs.clear()
            current_mineru_idxs.clear()
            current_pages.clear()
            current_len = 0
            start = 0
            while start < len(body):
                end = min(len(body), start + chunk_size)
                window = body[start:end]
                chunks.append({
                    "text": window,
                    "metadata": {
                        "pidx_start": p["pidx"],
                        "pidx_end": p["pidx"],
                        "mineru_idx_start": p.get("mineru_idx", p.get("idx", p["pidx"])) ,
                        "mineru_idx_end": p.get("mineru_idx", p.get("idx", p["pidx"])) ,
                        "page_start": p.get("page_idx", -1),
                        "page_end": p.get("page_idx", -1),
                        "num_paragraphs": 1,
                        "num_chars": len(window),
                        "hard_split": True,
                        "paragraph_char_range": [start, end],
                    },
                })
                if end - start < chunk_size:
                    break
                start = max(0, end - chunk_overlap)
            i += 1
            continue

        # 2) normal paragraph but doesn't fit with overlap: drop overlap and start new chunk with p
        current_texts.clear()
        current_pidxs.clear()
        current_mineru_idxs.clear()
        current_pages.clear()
        current_len = 0
        current_texts.append(body)
        current_pidxs.append(p["pidx"])
        current_mineru_idxs.append(p.get("mineru_idx", p.get("idx", p["pidx"])) )
        current_pages.append(p.get("page_idx", -1))
        current_len = len(body)
        i += 1
        continue

    flush_chunk()
    return chunks


def collect_paragraphs_from_mineru(items: List[dict], normalize_whitespace: bool) -> List[Dict[str, Any]]:
    paragraphs: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        t = it.get("type")
        if t != "text":
            continue
        text = it.get("text", "")
        text = normalize_text(text, normalize_whitespace)
        if not text:
            continue
        pidx = len(paragraphs)
        paragraphs.append({
            "pidx": pidx,              # index within filtered paragraphs
            "mineru_idx": idx,         # original index in MinerU items
            "text": text,
            "page_idx": it.get("page_idx", -1),
            "text_level": it.get("text_level"),
        })
    return paragraphs


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def process_school(school_dir: str, auto_dir: str, config: dict, output_dir: str) -> Optional[str]:
    # locate single *_content_list.json in auto_dir
    target_file = None
    for name in os.listdir(auto_dir):
        if name.endswith("_content_list.json"):
            target_file = os.path.join(auto_dir, name)
            break
    if target_file is None:
        return None

    school_name = os.path.basename(school_dir)
    with open(target_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    paragraphs = collect_paragraphs_from_mineru(items, normalize_whitespace=bool(config.get("normalize_whitespace", True)))

    chunks = chunk_paragraphs(
        paragraphs,
        chunk_size=int(config.get("chunk_size", 800)),
        chunk_overlap=int(config.get("chunk_overlap", 200)),
        min_chunk_chars=int(config.get("min_chunk_chars", 200)),
        max_chunk_chars=int(config.get("max_chunk_chars", 1200)),
    )

    # Output format: JSON array, first element is metadata, following are chunks
    payload: List[Any] = [
        {
            "metadata": {
                "school_name": school_name,
                "source_file": os.path.relpath(target_file, os.path.dirname(os.path.dirname(output_dir))),
                "generated_time": datetime.now().isoformat(),
                "language": config.get("language", "zh"),
                "total_source_items": len(items),
                "total_chunks": len(chunks),
            }
        }
    ]
    for ch in chunks:
        payload.append({
            "text": ch["text"],
            "metadata": ch.get("metadata", {}),
        })

    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, f"{school_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def main(argv: List[str]) -> int:
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)  # go to repo root
    config = load_config(project_root)

    mineru_root = os.path.join(project_root, config.get("mineru_input_dir", "data/minerU_output"))
    output_dir = os.path.join(project_root, config.get("output_dir", "data/chunk"))

    if not os.path.isdir(mineru_root):
        print(f"MinerU root not found: {mineru_root}")
        return 1

    schools = [d for d in os.listdir(mineru_root) if os.path.isdir(os.path.join(mineru_root, d))]
    if not schools:
        print("No school directories found under minerU_output.")
        return 0

    written: List[str] = []
    for school in schools:
        school_dir = os.path.join(mineru_root, school)
        auto_dir = os.path.join(school_dir, "auto")
        if not os.path.isdir(auto_dir):
            continue
        try:
            out_path = process_school(school_dir, auto_dir, config, output_dir)
            if out_path:
                written.append(out_path)
                print(f"Processed {school} -> {out_path}")
        except Exception as e:
            print(f"Failed {school}: {e}")

    print(f"Done. {len(written)} files written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


