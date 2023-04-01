from typing import *
from sentence_transformers import SentenceTransformer


def sbert_embedding(data_per_domain: List[List[str]]) -> List[List[List[float]]]:
    """Generate a random ordering of domains."""
    model = SentenceTransformer('paraphrase-distilroberta-base-v2')
    embeddings_per_domain = [model.encode(sentences) for sentences in data_per_domain]
    return embeddings_per_domain


EMBEDDING = {
    "sbert": sbert_embedding
}
