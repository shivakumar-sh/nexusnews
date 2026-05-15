from sentence_transformers import SentenceTransformer
import numpy as np

model=SentenceTransformer('all-MiniLM-L6-v2')

sentences=["Zomato delivers food to  your door","Swiggy is a food delivery platform","Python is a programming language",]

embeddings=model.encode(sentences)

print(f"shape: {embeddings.shape}")

print(f"First Embedding (first 5 value: {embeddings[0][:5]})")
