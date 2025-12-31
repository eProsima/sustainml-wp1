#!/usr/bin/env python3
# hf_mcp_server.py
#
# SustainML local MCP server (proxy) for Hugging Face Hub *metadata browsing*.
# IMPORTANT: This does NOT download model weights. It only queries the Hub API.

import os
import sys
from typing import Any, Dict, Optional

# MCP stdio uses stdout for JSON-RPC messages → NEVER print to stdout.
def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

# Silence banners/logspam
os.environ.setdefault("FASTMCP_NO_BANNER", "1")
os.environ.setdefault("FASTMCP_DISABLE_BANNER", "1")
os.environ.setdefault("FASTMCP_LOG_LEVEL", "ERROR")

from fastmcp import FastMCP
from huggingface_hub import HfApi

mcp = FastMCP("SustainML Hugging Face Hub MCP (metadata-only)")
api = HfApi()


def _safe_get(d: Any, key: str, default=None):
    try:
        return getattr(d, key, default)
    except Exception:
        return default


def _model_to_card(m) -> Dict[str, Any]:
    return {
        "model_id": _safe_get(m, "modelId", "") or _safe_get(m, "id", ""),
        "author": _safe_get(m, "author", ""),
        "downloads": int(_safe_get(m, "downloads", 0) or 0),
        "likes": int(_safe_get(m, "likes", 0) or 0),
        "pipeline_tag": _safe_get(m, "pipeline_tag", "") or "",
        "library_name": _safe_get(m, "library_name", "") or "",
        "tags": list(_safe_get(m, "tags", []) or []),
        "cardData": _safe_get(m, "cardData", None),
        "sha": _safe_get(m, "sha", ""),
        "lastModified": _safe_get(m, "lastModified", ""),
        "url": f"https://huggingface.co/{_safe_get(m, 'modelId', '') or _safe_get(m, 'id', '')}",
    }


@mcp.tool()
def hf_search_models(
    description: str,
    goal: Optional[str] = None,
    limit: int = 20,
    sort: str = "downloads",
) -> Dict[str, Any]:
    """
    Lexical search HF models by free text.
    No semantic embeddings.
    """
    description = (description or "").strip()
    goal = (goal or "").strip() if goal else None
    limit = int(limit)

    if not description:
        return {"models": [], "error": "empty_description"}

    try:
        models_iter = api.list_models(
            search=description,
            pipeline_tag=goal if goal else None,
            sort=sort,
            direction=-1,
            limit=limit,
            full=True,
            cardData=True,
        )
        models = list(models_iter)
        cards = [_model_to_card(m) for m in models]

        _log(f"[hf_search_models] desc='{description}' goal='{goal}' -> {len(cards)}")
        return {"models": cards}

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
