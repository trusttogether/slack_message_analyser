import openai
import numpy as np
from config.config import OPENAI_API_KEY, EMBEDDING_MODEL

openai.api_key = OPENAI_API_KEY

def get_embedding(text, model=EMBEDDING_MODEL):
    response = openai.Embedding.create(
        input=[text],
        model=model
    )
    return response['data'][0]['embedding']

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) 