# Copyright 2025 Proyectos y Sistemas de Mantenimiento SL (eProsima).
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
from huggingface_hub import HfApi
# Semantic embedding
from sentence_transformers import SentenceTransformer

# Silence banners/logspam
os.environ.setdefault("FASTMCP_NO_BANNER", "1")
os.environ.setdefault("FASTMCP_DISABLE_BANNER", "1")
os.environ.setdefault("FASTMCP_LOG_LEVEL", "ERROR")
os.environ.setdefault("FASTMCP_BANNER", "0")
os.environ.setdefault("FASTMCP_SHOW_BANNER", "0")

mcp = FastMCP("SustainML Hugging Face Hub MCP")
api = HfApi()

# Load embedding model once (cached in HF_HOME / default cache)
# Good tradeoff model: small & fast
_EMB_MODEL_NAME = os.environ.get("SUSTAINML_EMB_MODEL", "BAAI/bge-large-en")
_emb_model: Optional[SentenceTransformer] = None

_CACHE_DIR = os.path.join(pathlib.Path.home(), ".cache", "sustainml")
_CACHE_FILE = os.path.join(_CACHE_DIR, "hf_model_emb_cache.jsonl")

# In-memory cache:
_emb_cache: Dict[str, Dict[str, Any]] = {}
_emb_cache_loaded = False

_ALLOWED_GOALS = ["summarization", "translation", "text-generation"]

_new_cache_entries: List[Dict[str, Any]] = []


# MCP stdio uses stdout for JSON-RPC messages → NEVER print to stdout
def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cache_version_for_card(card: Dict[str, Any]) -> str:
    # Prefer sha; fallback to lastModified
    return (card.get("sha") or "").strip() or (card.get("lastModified") or "").strip() or "NA"


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


def _get_or_compute_model_embedding(embedder: SentenceTransformer, card: Dict[str, Any]) -> np.ndarray:
    """
    Returns normalized embedding vector for this model card,
    using disk cache if available and up-to-date.
    """
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


def _get_embedder() -> Optional[SentenceTransformer]:
    global _emb_model
    if _emb_model is not None:
        return _emb_model
    try:
        _log(f"[hf_mcp] Loading embedder: {_EMB_MODEL_NAME}")
        _emb_model = SentenceTransformer(_EMB_MODEL_NAME)
        _log("[hf_mcp] Embedder loaded OK (semantic rerank enabled)")
        return _emb_model
    except Exception as e:
        _log(f"[hf_mcp][WARN] Embedder NOT available -> fallback mode. Reason: {e}")
        _log(traceback.format_exc())
        _emb_model = None
        return None


def _safe_get(d: Any, key: str, default=None):
    try:
        return getattr(d, key, default)
    except Exception:
        return default


def _model_id(m) -> str:
    return _safe_get(m, "modelId", "") or _safe_get(m, "id", "") or ""


def _carddata_snippet(cardData: Any) -> str:
    """
    Turn HF cardData into a short stable string for embeddings.
    cardData can be dict, None, or weird.
    """
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


def _normalize_goal(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    return s


def _infer_goal_with_ollama(description: str, model: str = "llama3") -> str:
    _log(f"[GOAL] Ollama inference START model={model} desc='{description[:120]}'")

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
            _log(f"[GOAL] Ollama HTTP attempt={attempt+1}")
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
            _log(f"[GOAL] Ollama raw='{ans[:120]}'")
        except Exception as e:
            _log(f"[GOAL][ERR] Ollama HTTP call failed: {e}")
            ans = ""

        ans_n = _normalize_goal(ans)
        _log(f"[GOAL] Ollama normalized='{ans_n}'")

        for g in _ALLOWED_GOALS:
            if g in ans_n:
                _log(f"[GOAL] Ollama selected='{g}'")
                return g

        prompt = f"Your previous answer '{ans_n}' was invalid.\n\n" + prompt

    _log("[GOAL] Ollama failed to select a valid goal")
    return ""


def _build_task_query(description: str, task: str) -> str:
    base = (description or "").strip()
    if task == "summarization":
        return f"Task: summarization.\nUser description: {base}"
    if task == "text-generation":
        return f"Task: text generation.\nUser description: {base}"
    return f"Task: translation.\nUser description: {base}"


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


def _semantic_rerank(query: str, cards: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:

    _load_emb_cache()

    if not cards:
        return []

    embedder = _get_embedder()
    if embedder is None:
        _log("[rerank] FALLBACK active (no embedder). Using fallback ranking.")
        cards.sort(key=lambda c: (c.get("downloads",0), c.get("likes",0), c.get("lastModified","")), reverse=True)
        _log("[rerank] fallback top:")
        for c in cards[:min(5, len(cards))]:
            _log(f"  downloads={c.get('downloads',0)} likes={c.get('likes',0)} id={c.get('model_id','')}")
        return cards[:top_k]

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

    _log("[rerank] semantic rerank active. top scores:")
    for c in cards[:min(5, len(cards))]:
        _log(f"  score={c.get('score', 0.0):.4f} downloads={c.get('downloads',0)} likes={c.get('likes',0)} id={c.get('model_id','')}")

    return cards[:top_k], miss


@mcp.tool()
def hf_search_models(
    description: str,
    limit: int = 10,
    candidate_limit: int = 1000,
    sort: str = "downloads",
) -> Dict[str, Any]:
    description = (description or "").strip()
    if not description:
        return {"models": [], "error": "empty_description"}

    try:
        limit = max(1, min(int(limit), 50))
        candidate_limit = max(limit, min(int(candidate_limit), 1000))

        task = _infer_goal_with_ollama(description)
        if not task:
            return {"models": [], "error": "could_not_infer_goal"}

        pipeline_tag = task  # HF pipeline tags: "summarization" / "translation" / "text-generation"

        task_query = _build_task_query(description, task)

        _log(f"[hf_search_models] task='{task}' pipeline_tag='{pipeline_tag}' limit={limit} cand={candidate_limit}")
        _log(f"[hf_search_models] task_query='{task_query[:120]}'...")

        # Candidate pool building (downloads + likes + lastModified)
        third = max(50, candidate_limit // 3)

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

        if cards:
            c0 = cards[0]
            _log(f"[hf_search_models] candidate sample: {c0.get('model_id')} downloads={c0.get('downloads')} likes={c0.get('likes')} tag={c0.get('pipeline_tag')}")


        top, miss = _semantic_rerank(task_query, cards, top_k=limit)
        if miss > 0:
            _save_emb_cache()
        return {"models": top, "task": task, "pipeline_tag": pipeline_tag, "candidates": len(cards)}

    except Exception as e:
        _log(f"[hf_search_models][ERROR] {e}")
        return {"models": [], "error": str(e)}


@mcp.tool()
def hf_get_model_details(model_id: str) -> Dict[str, Any]:
    model_id = (model_id or "").strip()
    if not model_id:
        return {"error": "empty_model_id"}

    try:
        info = api.model_info(repo_id=model_id, files_metadata=False)
        data = {
            "model_id": model_id,
            "author": _safe_get(info, "author", ""),
            "downloads": int(_safe_get(info, "downloads", 0) or 0),
            "likes": int(_safe_get(info, "likes", 0) or 0),
            "pipeline_tag": _safe_get(info, "pipeline_tag", "") or "",
            "library_name": _safe_get(info, "library_name", "") or "",
            "tags": list(_safe_get(info, "tags", []) or []),
            "cardData": _safe_get(info, "cardData", None),
            "sha": _safe_get(info, "sha", ""),
            "lastModified": _safe_get(info, "lastModified", ""),
            "siblings": [getattr(s, "rfilename", "") for s in (_safe_get(info, "siblings", []) or [])],
            "url": f"https://huggingface.co/{model_id}",
        }
        _log(f"[hf_get_model_details] {model_id}")
        return data
    except Exception as e:
        _log(f"[hf_get_model_details][ERROR] {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
