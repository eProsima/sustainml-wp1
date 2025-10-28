import os
import json
import torch
import accelerate.utils.memory as _mem
if not hasattr(_mem, "clear_device_cache"):
    _mem.clear_device_cache = torch.cuda.empty_cache
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from neo4j import GraphDatabase
from ollama import Client
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load precomputed embeddings and metadata
with open(os.path.join(BASE_DIR, 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)

annoy_index = AnnoyIndex(1024, 'angular')
annoy_index.load(os.path.join(BASE_DIR, 'models_index.ann'))

# Load sentence transformer for semantic search
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sentence_model = SentenceTransformer('BAAI/bge-large-en', device=device)

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

MODEL = "mistral-small"

# Connect to Neo4j
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

client = Client(host='http://localhost:11434')


def search_semantic(query, allowed_models=None, top_k=20, sample_k=10):
    """Semantic search. If allowed_models is provided, search ONLY among those models."""
    print("Search Semantic")
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()[0]

    # 1) If we have an allowed list, build a tiny sub-index and query ONLY that
    if allowed_models:
        # Normalize names to map reliably
        name_to_global = {
            (m.get("name","").strip().casefold()): i
            for i, m in enumerate(metadata) if m.get("name")
        }
        # Map allowed names -> global ids; keep original names for mapping back
        pairs = []
        for n in allowed_models:
            gid = name_to_global.get((n or "").strip().casefold())
            if gid is not None:
                pairs.append((gid, n))
        if not pairs:
            print("No allowed models matched metadata names.")
            return []

        # Build a small in-memory Annoy index with ONLY allowed vectors
        sub = AnnoyIndex(1024, 'angular')
        id2name = []
        for new_id, (gid, name) in enumerate(pairs):
            sub.add_item(new_id, annoy_index.get_item_vector(gid))
            id2name.append(name)
        sub.build(10)

        # Query sub-index; cap K by available items
        k = min(top_k, sub.get_n_items()) or 1
        nn = sub.get_nns_by_vector(query_embedding, k)
        if not nn:
            print("No neighbors within allowed subset.")
            return []

        # Map back to original metadata using names (names are from allowed list)
        name_to_data = {m["name"]: m for m in metadata if m.get("name")}
        results = [name_to_data[id2name[i]] for i in nn if 0 <= i < len(id2name) and id2name[i] in name_to_data]

        # Sample for downstream prompt size
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

    # 2) No whitelist: fall back to global index (original behavior)
    nn = annoy_index.get_nns_by_vector(query_embedding, top_k)
    if not nn:
        print("No neighbors in global index.")
        return []
    results = [metadata[i] for i in nn if 0 <= i < len(metadata)]
    sampled = random.sample(results, min(sample_k, len(results)))
    print("Global neighbors:", [x.get("name") for x in sampled])
    return sampled


def generate_natural_answer(knowledge, user_question, allowed_models=None):
    """Return a model_name. If a whitelist is provided, pick directly from knowledge (no LLM)."""
    print("Generate natural answer")

    if allowed_models:
        allowed_set = {str(a).strip() for a in allowed_models if a}
        # Keep only names that actually came back from retrieval
        candidates = [m.get("name") for m in knowledge if m.get("name") in allowed_set]
        picked = candidates[0] if candidates else "None"
        return json.dumps({"model_name": picked})

    template = '{"model_name":"The name of the best-fitting model."}'
    final_prompt = f"""
    Based on the retrieved knowledge:
    {knowledge}

    Answer the following question: {user_question}.
    Output strictly in JSON following {template}.
    """
    system_msg = ('Use only the provided context; if no answer, respond with {"model_name":"None"}.')

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": final_prompt}
    ]

    return client.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": 0.0}
    )["message"]["content"].strip()


def answer_question(user_question, allowed_models=None):
    """Pick a model strictly from the provided allowed_models list (goal-scoped)."""
    if not allowed_models:
        print("[WARN] answer_question called without allowed_models; returning 'None'")
        return "None"

    # Retrieval restricted to allowed_models via sub-index
    knowledge = search_semantic(user_question, allowed_models=allowed_models)

    # Deterministic pick from retrieved knowledge (no LLM wandering)
    raw = generate_natural_answer(knowledge, user_question, allowed_models=allowed_models)

    clean = raw.strip().removeprefix("```json").removesuffix("```").strip().replace("'", '"')
    try:
        data = json.loads(clean)
        picked = data.get("model_name", "None")
    except Exception:
        picked = "None"

    return picked
