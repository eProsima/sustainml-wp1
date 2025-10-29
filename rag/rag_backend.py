import gdown
import json
import os
import pathlib
import random
import torch

from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANN_DIM = 1024
ANN_FILENAME = "models_index.ann"

GDRIVE_ID = "1TQvt1bSXares-I9l7Wki0Jge3oubRkOJ"

def _find_or_fetch_ann() -> str | None:
    local = os.path.join(BASE_DIR, ANN_FILENAME)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local

    cache_dir = os.path.join(pathlib.Path.home(), ".cache", "sustainml")
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, ANN_FILENAME)
    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        return cached

    dst = cached
    print(f"[RAG] Downloading ANN index from Google Drive id={GDRIVE_ID} -> {dst}", flush=True)
    gdown.download(id=GDRIVE_ID, output=dst, quiet=False)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst

    return None

# Load ANN
_ann_path = _find_or_fetch_ann()
if not _ann_path:
    raise FileNotFoundError(
        "[RAG] models_index.ann not found or downloadable.\n"
        "Place it next to this file, or ensure the fixed Google Drive id is reachable."
    )

annoy_index = AnnoyIndex(ANN_DIM, 'angular')
annoy_index.load(_ann_path)
print(f"[RAG] ANN index loaded from: {_ann_path}", flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sentence_model = SentenceTransformer('BAAI/bge-large-en', device=device)

"""
# When there is no allowed_models list. Additional condition needed in generate_natural_answer().
# Take it from the previous version/commit. Include Client and GraphDatabase.
# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

MODEL = "mistral-small"

# Connect to Neo4j
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

client = Client(host='http://localhost:11434')
"""

# Load metadata & encoder
with open(os.path.join(BASE_DIR, 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)


def search_semantic(query, allowed_models=None, top_k=20, sample_k=10):
    """Semantic search. If allowed_models is provided, search ONLY among those models."""
    print("Search Semantic")
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()[0]

    # If we have an allowed list, build a tiny sub-index and query ONLY that
    if allowed_models:
        # Map: normalized name -> global id in full index
        name_to_global = {
            (m.get("name", "").strip().casefold()): i
            for i, m in enumerate(metadata) if m.get("name")
        }

        pairs = []
        for n in allowed_models:
            gid = name_to_global.get((n or "").strip().casefold())
            if gid is not None:
                pairs.append((gid, n))
        if not pairs:
            print("No allowed models matched metadata names.")
            return []

        # Build small in-memory Annoy with ONLY allowed vectors
        sub = AnnoyIndex(ANN_DIM, 'angular')
        id2name = []
        for new_id, (gid, name) in enumerate(pairs):
            sub.add_item(new_id, annoy_index.get_item_vector(gid))
            id2name.append(name)
        sub.build(10)

        k = min(top_k, sub.get_n_items()) or 1
        nn = sub.get_nns_by_vector(query_embedding, k)
        if not nn:
            print("No neighbors within allowed subset.")
            return []

        # Map back by names (names originate from allowed list)
        name_to_data = {m["name"]: m for m in metadata if m.get("name")}
        results = [
            name_to_data[id2name[i]]
            for i in nn
            if 0 <= i < len(id2name) and id2name[i] in name_to_data
        ]

        if not results:
            return []

        sampled = random.sample(results, min(sample_k, len(results)))
        print("Allowed-only neighbors:", [x.get("name") for x in sampled])

        if allowed_models:
            allowed_set = {a.strip() for a in allowed_models if a}
            offenders = [m.get("name") for m in sampled if m.get("name") not in allowed_set]
            if offenders:
                print("[ERR] sampled names not in allowed set:", offenders)
        return sampled

    # No whitelist: use the global index
    nn = annoy_index.get_nns_by_vector(query_embedding, top_k)
    if not nn:
        print("No neighbors in global index.")
        return []
    results = [metadata[i] for i in nn if 0 <= i < len(metadata)]
    sampled = random.sample(results, min(sample_k, len(results)))
    print("Global neighbors:", [x.get("name") for x in sampled])
    return sampled


def generate_natural_answer(knowledge, user_question, allowed_models=None):
    """
    Return {"model_name": "..."} as JSON string.
    If a whitelist is provided, pick deterministically from retrieved items (no LLM).
    Otherwise, query Ollama (required).
    """
    print("Generate natural answer")

    allowed_set = {str(a).strip() for a in allowed_models if a}
    candidates = [m.get("name") for m in knowledge if m.get("name") in allowed_set]
    picked = candidates[0] if candidates else "None"
    return json.dumps({"model_name": picked})


def answer_question(user_question, allowed_models=None):
    """
    Pick a model strictly from the provided allowed_models list (goal-scoped).
    """
    if not allowed_models:
        print("[WARN] answer_question called without allowed_models; returning 'None'")
        return "None"

    knowledge = search_semantic(user_question, allowed_models=allowed_models)
    raw = generate_natural_answer(knowledge, user_question, allowed_models=allowed_models)

    clean = raw.strip().removeprefix("```json").removesuffix("```").strip().replace("'", '"')
    try:
        data = json.loads(clean)
        picked = data.get("model_name", "None")
    except Exception:
        picked = "None"
    return picked
