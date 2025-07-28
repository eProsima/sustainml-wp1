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

MODEL = "llama3"

# Connect to Neo4j
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

client = Client(host='http://localhost:11434')


def search_semantic(query, top_k=100):
    """Performs semantic search on the Annoy index."""
    print("Search Semantic")
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()[0]
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, top_k)

    results = []
    for idx in nearest_neighbors:
        results.append(metadata[idx])
    print("Neighbors results: ", results)  # debug
    return results


def generate_cypher_query(semantic_results):
    """Generates a Cypher query based on retrieved metadata."""
    print("generate Cypher")
    if not semantic_results:
        return None

    relevant_models = [res["name"] for res in semantic_results]

    cypher_query = f"""
    WITH {relevant_models} AS model_names
    MATCH (m:Model)
      WHERE m.name IN model_names
    OPTIONAL MATCH (m)-[:HAS_PROBLEM]->(p:Problem)
    OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
    OPTIONAL MATCH (m)-[:HAS_TAG]->(t:Tag)
    OPTIONAL MATCH (m)-[:HAS_COVER_TAG]->(ct:CoverTag)
    RETURN
      m.name         AS name,
      m.id           AS modelId,
      m.downloads    AS downloads,
      m.likes        AS likes,
      m.lastModified AS lastModified,
      p.name         AS problem,
      l.name         AS library,
      collect(DISTINCT t.name)  AS tags,
      collect(DISTINCT ct.name) AS coverTags
    LIMIT 100
    """
    return cypher_query


def execute_cypher_query(cypher_query):
    """Executes the Cypher query on the Neo4j database."""
    print("Execute cypher")
    with neo4j_driver.session() as session:
        results = session.run(cypher_query)
        data = [dict(record) for record in results]
        print(f"Data retrieve from graph with neighbors: {data}")  # debug
        return data


def generate_natural_answer(knowledge, user_question):
    """Generates a natural language response using the LLM."""
    print("Generate natural answer")
    print(f"Full Knowledge: {knowledge}")  # debug
    final_prompt = f"""
    Based on the retrieved knowledge:
    {knowledge}

    Answer the following question: {user_question}
    """
    messages = [
        {"role": "system", "content": "You are an extremely concise assistant. If the user asks for just the model name, output exactly the name and nothing else. In other cases, give a one‐sentence answer."},
        {"role": "user", "content": final_prompt}
    ]

    return client.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": 0.0}
    )["message"]["content"].strip()


def answer_question(user_question):
    """Handles user questions by retrieving search results."""

    # TBD category
    semantic_results = search_semantic(user_question)
    cypher_query = generate_cypher_query(semantic_results)

    if cypher_query:
        graph_results = execute_cypher_query(cypher_query)
        knowledge = semantic_results + graph_results
    else:
        knowledge = semantic_results

    answer = generate_natural_answer(knowledge, user_question)
    print(f"Response: {answer}")  # debug

    return answer
