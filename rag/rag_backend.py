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

def generate_natural_answer(knowledge, user_question):
    """Generates a natural language response using the LLM."""
    print("Generate natural answer")
    template = """
    {
        "model_name": "The name of the model that best fits the user's question with the given knowledge."
    }
    """
    print(f"Full Knowledge: {knowledge}")  # debug
    final_prompt = f"""
    Based on the retrieved knowledge:
    {knowledge}

    Answer the following question: {user_question}.
    The format of the Hugging Face name of the model must be like this one that follows: 'openai-community/gpt2-large'.
    Only output one model name in JSON format.
    For the json use the following template {template}. 
    Do not add any sentence before and after.
    """
    messages = [
        {"role": "system", "content": "You are an extremely concise assistant. Use only the provided context information to form your response. If an answer can not be found within the provided context information respond with 'The answer could not be found in the provided context."},
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
    knowledge = search_semantic(user_question)

    answer = generate_natural_answer(knowledge, user_question)
    clean = answer.strip().removeprefix("```json").removesuffix("```").strip().replace("'", '"')
    data = json.loads(clean)
    answer = data["model_name"]
    print(f"Response: {answer}")  # debug

    return answer
