import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

model=SentenceTransformer('all-MiniLM-L6-v2')

docs=["How to reset your password",
      "Track your zomoto delivery live",
      "Cancel a Zomato Order before pickup",
      "Zomato Gold membership benefits",
      "Contact Zomato Customer Support"]

doc_embeddings=model.encode(docs).astype('float32')

index=faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

query="I want to cancel my food order"
q_emb=model.encode([query]).astype('float32')

distanmces, indices = index.search(q_emb, k=2)
for i,idx in enumerate(indices[0]):
    print(f"Result {i+1}: {docs[idx]} (score: {distanmces[0][i]:.2f})")