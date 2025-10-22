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

def search_semantic(query, top_k=200, sample_k=30):
    """Performs semantic search on the Annoy index."""
    print("Search Semantic")
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()[0]
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, top_k)
    sampled = random.sample(nearest_neighbors, min(sample_k, len(nearest_neighbors)))

    results = []
    for idx in sampled:
        results.append(metadata[idx])
    print("Neighbors results: ", results)  # debug
    return results

def generate_natural_answer(knowledge, user_question, allowed_models=None):
    """Generates a natural language response using the LLM."""
    print("Generate natural answer")

    if allowed_models:
        template = '{"model_name":"<one item from allowed_models>"}'
        allowed_str = ", ".join(allowed_models)
        final_prompt = f"""
        Based on the retrieved knowledge:
        {knowledge}

        Consider ONLY these allowed model names (choose exactly one):
        {allowed_str}

        Answer the following question: {user_question}
        Output strictly in JSON following {template}.
        """
        system_msg = 'Only pick from the provided allowed list. If none, say: {"model_name":"none"}.'
    else:
        template = '{"model_name":"The name of the best-fitting model."}'
        final_prompt = f"""
        Based on the retrieved knowledge:
        {knowledge}

        Answer the following question: {user_question}.
        Output strictly in JSON following {template}.
        """
        system_msg = ('Use only the provided context; if no answer, respond with {"model_name":"none"}.')

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
    """Handles user questions by retrieving search results."""

    knowledge = search_semantic(user_question)
    raw = generate_natural_answer(knowledge, user_question, allowed_models=allowed_models)

    clean = raw.strip().removeprefix("```json").removesuffix("```").strip().replace("'", '"')
    try:
        data = json.loads(clean)
        picked = data.get("model_name", "none")
    except Exception:
        picked = "none"

    # Safety net: keep choice on-list
    if allowed_models and picked not in allowed_models:
        picked = allowed_models[0] if allowed_models else "none"
    return picked


def get_allowed_models_for_problem(problem_name: str):
    """
    Filter models by:
      - For text-classification: status must be OK.
      - For other problems: status not in (FAIL, OOM)
        AND library must be transformers (accept namespaced ':transformers').
    """
    cypher = """
    MATCH (m:Model)-[:HAS_PROBLEM]->(p:Problem {name: $problem_name})
    OPTIONAL MATCH (m)-[:HAS_HEALTH_STATUS]->(hs:HealthStatus)
    OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
    WITH m, toLower(coalesce(hs.status, m.health_status)) AS status, l, $problem_name AS prob
    WHERE
      (
        prob = 'text-classification' AND status = 'ok'
      )
      OR
      (
        prob <> 'text-classification'
        AND (NOT status IN ['fail','oom'])
        AND (l.name = 'transformers' OR l.name ENDS WITH ':transformers')
      )
    RETURN m.name AS name
    ORDER BY m.downloads DESC, name
    """
    with neo4j_driver.session() as s:
        return [r["name"] for r in s.run(cypher, problem_name=problem_name)]
