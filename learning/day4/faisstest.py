import faiss
import numpy as np
 
from sentence_transformers import SentenceTransformer

model=SentenceTransformer('all-MiniLM-L6-v2')

docs=["How to reset your Zomato password",
      "Track your Zomato delivery live",
      "Cancel a zomato  order before pinkup",
      "Zomato Gold membership benifits",
      "Contact Zomato customer support",]

doc_embeddings=model.encode(docs).astype('float32')

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

query ="I want to cancle my food order"
q_emb = model.encode([query]).astype('float32')

distances,indices= index.search(q_emb,k=2)
for i,idx in enumerate(indices[0]):
    print(f"Result {i+1}: {docs[idx]} (score: {distances[0][i]:.2f})")