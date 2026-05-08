# Copyright 2026 Proyectos y Sistemas de Mantenimiento SL (eProsima).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SustainML MCP Server Implementation."""

import json
import numpy as np
import os
import pathlib
import sys
import traceback
import urllib.request

from typing import Any, Dict, List, Optional
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, hf_hub_download

# Silence banners/logspam
os.environ.setdefault("FASTMCP_NO_BANNER", "1")
os.environ.setdefault("FASTMCP_LOG_LEVEL", "ERROR")

mcp = FastMCP("SustainML Hugging Face Hub MCP")
api = HfApi()

# Load embedding model once (cached in HF_HOME / default cache)
_EMB_MODEL_NAME = os.environ.get("SUSTAINML_EMB_MODEL", "BAAI/bge-large-en")
_emb_model: Optional[SentenceTransformer] = None

_CACHE_DIR = os.path.join(pathlib.Path.home(), ".cache", "sustainml")
_CACHE_FILE = os.path.join(_CACHE_DIR, "hf_model_emb_cache.jsonl")

# In-memory cache
_emb_cache: Dict[str, Dict[str, Any]] = {}
_emb_cache_loaded = False

_ALLOWED_GOALS = ["summarization", "translation", "text-generation"]

_new_cache_entries: List[Dict[str, Any]] = []


# MCP stdio uses stdout for JSON-RPC messages not to print to stdout
def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# Gets a stable version key for a model card for embedding cache invalidation
def _cache_version_for_card(card: Dict[str, Any]) -> str:
    # Prefer sha; fallback to lastModified
    return (card.get("sha") or "").strip() or (card.get("lastModified") or "").strip() or "NA"


# Loads the embedding cache from disk into memory (only once per process)
def _load_emb_cache() -> None:
    global _emb_cache_loaded, _emb_cache
    if _emb_cache_loaded:
        return
    _emb_cache_loaded = True
    _emb_cache = {}

    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        if not os.path.isfile(_CACHE_FILE):
            _log(f"[cache] no cache file yet at {_CACHE_FILE}")
            return

        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    mid = rec.get("model_id")
                    ver = rec.get("ver")
                    emb = rec.get("emb")
                    if mid and ver and isinstance(emb, list):
                        _emb_cache[mid] = {"ver": ver, "emb": emb}
                except Exception:
                    continue

        _log(f"[cache] loaded {len(_emb_cache)} embeddings from {_CACHE_FILE}")
    except Exception as e:
        _emb_cache = {}
        _log(f"[cache][WARN] failed to load cache: {e}")


# Appends newly computed embeddings to the on-disk cache
def _save_emb_cache() -> None:
    global _new_cache_entries
    if not _new_cache_entries:
        return
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_CACHE_FILE, "a", encoding="utf-8") as f:
            for rec in _new_cache_entries:
                f.write(json.dumps(rec) + "\n")
        _log(f"[cache] appended {len(_new_cache_entries)} embeddings to {_CACHE_FILE}")
        _new_cache_entries = []
    except Exception as e:
        _log(f"[cache][WARN] failed to append cache: {e}")


# Returns a normalized embedding for a model card, reusing the cache when possible
def _get_or_compute_model_embedding(embedder: SentenceTransformer, card: Dict[str, Any]) -> np.ndarray:
    _load_emb_cache()

    mid = (card.get("model_id") or "").strip()
    ver = _cache_version_for_card(card)

    if mid:
        cached = _emb_cache.get(mid)
        if cached and cached.get("ver") == ver and isinstance(cached.get("emb"), list):
            return np.array(cached["emb"], dtype=np.float32)

    text = _text_for_embedding(card)
    vec = embedder.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    if mid:
        _emb_cache[mid] = {"ver": ver, "emb": vec.tolist()}
        _new_cache_entries.append({"model_id": mid, "ver": ver, "emb": vec.tolist()})

    return vec


# Loads and caches the sentence-transformers model used for reranking
def _get_embedder() -> Optional[SentenceTransformer]:
    global _emb_model
    if _emb_model is not None:
        return _emb_model
    try:
        _emb_model = SentenceTransformer(_EMB_MODEL_NAME)
        return _emb_model
    except Exception as e:
        _log(f"[hf_mcp][WARN] Embedder NOT available -> fallback mode. Reason: {e}")
        _log(traceback.format_exc())
        _emb_model = None
        return None


# Safely reads an attribute from HF objects that may have missing fields
def _safe_get(d: Any, key: str, default=None):
    try:
        return getattr(d, key, default)
    except Exception:
        return default


# Extracts the HF model id from either modelId or id fields
def _model_id(m) -> str:
    return _safe_get(m, "modelId", "") or _safe_get(m, "id", "") or ""


# Turns HF cardData into a short stable string for embeddings
def _carddata_snippet(cardData: Any) -> str:
    if not isinstance(cardData, dict):
        return ""
    keys = [
        "language",
        "license",
        "base_model",
        "model_name",
        "pipeline_tag",
        "tags",
        "datasets",
        "task_name",
        "metrics",
        "summary",
        "description",
    ]
    parts = []
    for k in keys:
        v = cardData.get(k, None)
        if not v:
            continue
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v[:20])
        elif isinstance(v, dict):
            v = " ".join(f"{kk}:{vv}" for kk, vv in list(v.items())[:10])
        parts.append(f"{k}={v}")
    return "\n".join(parts)


# Converts a HF Hub model object into a JSON-serializable dict for the UI
def _model_to_card(m) -> Dict[str, Any]:
    mid = _model_id(m)
    cd = _safe_get(m, "cardData", None)
    return {
        "model_id": mid,
        "author": _safe_get(m, "author", ""),
        "downloads": int(_safe_get(m, "downloads", 0) or 0),
        "likes": int(_safe_get(m, "likes", 0) or 0),
        "pipeline_tag": _safe_get(m, "pipeline_tag", "") or "",
        "library_name": _safe_get(m, "library_name", "") or "",
        "tags": list(_safe_get(m, "tags", []) or []),
        "cardData": cd,
        "card_text": _carddata_snippet(cd),
        "sha": _safe_get(m, "sha", ""),
        "lastModified": _safe_get(m, "lastModified", ""),
        "url": f"https://huggingface.co/{mid}",
    }


# Infers the best HF pipeline goal for a natural-language request using the Ollama model
def _infer_goal_with_ollama(description: str, model: str = "llama3") -> str:
    _log(f"[GOAL] Ollama inference START model={model} desc='{description[:1200]}'")

    goals_str = ", ".join(_ALLOWED_GOALS)
    prompt = (
        f"Choose the single best machine learning goal for the user's request.\n"
        f"Allowed goals: {goals_str}\n\n"
        f"User request: {description}\n\n"
        f"Rules:\n"
        f"- Reply with ONLY one goal from the allowed list.\n"
        f"- No quotes, no punctuation, no extra words.\n"
        f"- If unsure, reply with text-generation.\n"
    )

    url = "http://localhost:11434/api/chat"

    for attempt in range(3):
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            ans = (data.get("message") or {}).get("content", "") or ""
        except Exception as e:
            _log(f"[GOAL][ERR] Ollama HTTP call failed: {e}")
            ans = ""

        ans_n = (ans or "").strip().lower().replace("_", "-").replace(" ", "-")

        for g in _ALLOWED_GOALS:
            if g in ans_n:
                return g

        prompt = f"Your previous answer '{ans_n}' was invalid.\n\n" + prompt

    _log("[GOAL] Ollama failed to select a valid goal")
    return ""


# Builds the semantic query used for reranking by combining task + user description
def _build_task_query(description: str, task: str) -> str:
    base = (description or "").strip()
    if task == "summarization":
        return f"Task: summarization.\nUser description: {base}"
    if task == "text-generation":
        return f"Task: text generation.\nUser description: {base}"
    return f"Task: translation.\nUser description: {base}"


# Builds the text payload that is embedded for each candidate model card
def _text_for_embedding(card: Dict[str, Any]) -> str:
    mid = card.get("model_id", "")
    pt = card.get("pipeline_tag", "") or ""
    tags = card.get("tags", []) or []
    lib = card.get("library_name", "") or ""
    author = card.get("author", "") or ""
    card_text = card.get("card_text", "") or ""

    tags_txt = " ".join(str(t) for t in tags[:40])

    return (
        f"model_id: {mid}\n"
        f"pipeline_tag: {pt}\n"
        f"library: {lib}\n"
        f"author: {author}\n"
        f"tags: {tags_txt}\n"
        f"{card_text}"
    )


# Reranks candidate models using semantic similarity
def _semantic_rerank(query: str, cards: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:

    _load_emb_cache()

    if not cards:
        return [], 0

    embedder = _get_embedder()
    if embedder is None:
        _log("[rerank] FALLBACK active (no embedder). Using fallback ranking.")
        cards.sort(key=lambda c: (c.get("downloads",0), c.get("likes",0), c.get("lastModified","")), reverse=True)
        return cards[:top_k], 0

    qv = embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    # Compute/reuse each model embedding via cache
    mv = []
    miss = 0
    for c in cards:
        ver = _cache_version_for_card(c)
        mid = (c.get("model_id") or "").strip()
        cached = _emb_cache.get(mid) if mid else None
        if not (cached and cached.get("ver") == ver):
            miss += 1
        mv.append(_get_or_compute_model_embedding(embedder, c))

    mv = np.vstack(mv) if mv else np.zeros((0, qv.shape[0]), dtype=np.float32)

    _log(f"[cache] embed miss={miss} hit={len(cards)-miss} total={len(cards)}")

    scores = (mv @ qv).tolist()

    for c, s in zip(cards, scores):
        c["score"] = float(s)

    cards.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    return cards[:top_k], miss


# Normalizes a value into a list of strings for tooltip formatting
def _as_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x is not None]
    return [str(v)]


# Parses common structured information from HF tag strings
def _extract_from_tags(tags: List[str]) -> Dict[str, List[str]]:
    out = {"license": [], "datasets": [], "base_model": [], "languages": []}
    for t in (tags or []):
        s = str(t)
        sl = s.lower()

        if sl.startswith("license:"):
            out["license"].append(s.split(":", 1)[1])
        elif sl.startswith("dataset:"):
            out["datasets"].append(s.split(":", 1)[1])
        elif sl.startswith("base_model:"):
            out["base_model"].append(s.split(":", 1)[1])
        # Very common language tags are two-letter codes; a safe small hardcoded allowlist
        """
        else:
            if sl in {"en","fr","de","es","it","pt","ja","ko","zh","ar","ru","nl","pl","tr","uk","vi","fa","el","he","hi","id","cs","ro"}:
                out["languages"].append(sl)
        """

    # Dedupe while preserving order
    for k in out:
        seen = set()
        uniq = []
        for x in out[k]:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        out[k] = uniq
    return out


# Builds the compact multiline tooltip text shown in the UI
def _tooltip_from_details(details: Dict[str, Any]) -> str:
    cd = details.get("cardData") if isinstance(details.get("cardData"), dict) else {}

    task = details.get("pipeline_tag", "") or ""
    lib = details.get("library_name", "") or ""

    tags = details.get("tags", []) or []
    tx = _extract_from_tags(tags)

    base_model = cd.get("base_model", None) or tx["base_model"]
    license_ = cd.get("license", None) or tx["license"]
    language = cd.get("language", None) or tx["languages"]
    datasets = cd.get("datasets", None) or tx["datasets"]

    hint_tags = []
    for t in tags:
        tl = str(t).lower()
        if any(k in tl for k in [
            "instruct", "chat", "rlhf", "long-context", "quant", "gguf",
            "4bit", "8bit", "bnb", "lora", "adapter", "onnx", "tensorrt"
        ]):
            hint_tags.append(str(t))
    hint_tags = hint_tags[:8]

    lines = []
    if task:
        lines.append(f"Task: {task}")
    if lib:
        lines.append(f"Library: {lib}")

    bm = _as_list(base_model)
    if bm:
        lines.append(f"Base model: {', '.join(bm[:2])}")

    lg = _as_list(language)
    if lg:
        lines.append(f"Language: {', '.join(lg[:5])}")

    lc = _as_list(license_)
    if lc:
        lines.append(f"License: {', '.join(lc[:2])}")

    ds = _as_list(datasets)
    if ds:
        lines.append(f"Datasets: {', '.join(ds[:5])}")

    if hint_tags:
        lines.append(f"Tags: {', '.join(hint_tags)}")

    return "\n".join(lines).strip()


# Downloads a small JSON file from a HF repo and parses it
def _read_repo_json(repo_id: str, filename: str) -> Any:
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Builds tooltip text from config.json-only content (used for hover tooltip mode)
def _tooltip_from_config(cfg: Any) -> str:
    if not isinstance(cfg, dict):
        return ""

    def g(*keys):
        for k in keys:
            if k in cfg:
                return cfg[k]
        return None

    model_type = g("model_type")
    arch = g("architectures")
    if isinstance(arch, list):
        arch = ", ".join(str(x) for x in arch[:2])

    hidden = g("hidden_size", "d_model")
    layers = g("num_hidden_layers", "n_layer", "num_layers")
    heads = g("num_attention_heads", "n_head")
    vocab = g("vocab_size")
    max_pos = g("max_position_embeddings", "n_positions")
    dtype = g("torch_dtype")

    lines = []
    if model_type: lines.append(f"Config type: {model_type}")
    if arch:      lines.append(f"Architectures: {arch}")
    if hidden is not None: lines.append(f"Hidden size: {hidden}")
    if layers is not None: lines.append(f"Layers: {layers}")
    if heads is not None:  lines.append(f"Heads: {heads}")
    if max_pos is not None: lines.append(f"Max pos: {max_pos}")
    if vocab is not None: lines.append(f"Vocab: {vocab}")
    if dtype: lines.append(f"Dtype: {dtype}")

    return "\n".join(lines).strip()


# MCP tool: searches HF models for a user description and optionally fetch a single model's config tooltip
@mcp.tool()
def hf_search_models(
    description: str,
    limit: int = 10,
    candidate_limit: int = 200,
) -> Dict[str, Any]:
    description = (description or "").strip()

    # HOVER MODE: fetch only config.json for ONE model
    if description.startswith("__MODEL_CONFIG__:"):
        model_id = description.split(":", 1)[1].strip()
        _log(f"[hf_search_models][CONFIG] request model_id={model_id}")

        if not model_id:
            return {"model_id": "", "tooltip": ""}

        try:
            cfg = _read_repo_json(model_id, "config.json")
            tooltip = _tooltip_from_config(cfg)
            _log(f"[hf_search_models][CONFIG] OK model_id={model_id} tooltip_len={len(tooltip)}")
            return {"model_id": model_id, "tooltip": tooltip}
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if "gated repo" in low or "401" in low or "403" in low:
                tip = "Gated model (requires Hugging Face login / granted access)."
                return {"model_id": model_id, "tooltip": tip, "error": msg}
            return {"model_id": model_id, "tooltip": "Error fetching config.", "error": msg}

    # NORMAL SEARCH MODE
    if not description:
        return {"models": [], "error": "empty_description"}

    try:
        candidate_limit = max(3 * limit, candidate_limit)

        task = _infer_goal_with_ollama(description)
        if not task:
            return {"models": [], "error": "could_not_infer_goal"}

        pipeline_tag = task  # HF pipeline tags: "summarization" / "translation" / "text-generation"

        task_query = _build_task_query(description, task)

        _log(f"[hf_search_models] task='{task}' pipeline_tag='{pipeline_tag}' limit={limit} cand={candidate_limit}")

        # Candidate pool building (downloads + likes + lastModified)
        third = candidate_limit // 3

        models_downloads = list(api.list_models(
            pipeline_tag=pipeline_tag,
            sort="downloads",
            direction=-1,
            limit=third,
            full=True,
            cardData=True,
        ))

        models_likes = list(api.list_models(
            pipeline_tag=pipeline_tag,
            sort="likes",
            direction=-1,
            limit=third,
            full=True,
            cardData=True,
        ))

        try:
            models_fresh = list(api.list_models(
                pipeline_tag=pipeline_tag,
                sort="lastModified",
                direction=-1,
                limit=third,
                full=True,
                cardData=True,
            ))
        except Exception as e:
            _log(f"[hf_search_models][WARN] sort=lastModified not available: {e}")
            models_fresh = []

        # merge + dedupe
        seen = set()
        cards = []
        for m in (models_downloads + models_likes + models_fresh):
            c = _model_to_card(m)
            mid = c.get("model_id", "")
            if mid and mid not in seen:
                seen.add(mid)
                cards.append(c)

        _log(f"[hf_search_models] candidates fetched: {len(cards)} (downloads+likes+fresh merged), each={third}")

        top, miss = _semantic_rerank(task_query, cards, top_k=limit)

        # Enrich TOP results with practical tooltip (model card)
        for c in top:
            try:
                mid = (c.get("model_id") or "").strip()
                if not mid:
                    c["tooltip"] = ""
                    continue

                # Build "details" structure compatible with _tooltip_from_details()
                cd = c.get("cardData", {})
                if not isinstance(cd, dict):
                    cd = {}

                details = {
                    "model_id": mid,
                    "pipeline_tag": c.get("pipeline_tag", "") or "",
                    "library_name": c.get("library_name", "") or "",
                    "tags": c.get("tags", []) or [],
                    "cardData": cd,
                }
                c["tooltip"] = _tooltip_from_details(details)
            except Exception:
                c["tooltip"] = ""

        if miss > 0:
            _save_emb_cache()
        return {"models": top, "task": task, "pipeline_tag": pipeline_tag, "candidates": len(cards)}

    except Exception as e:
        _log(f"[hf_search_models][ERROR] {e}")
        return {"models": [], "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
