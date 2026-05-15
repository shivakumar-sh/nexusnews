from sentence_transformers import SentenceTransformer,util

model=SentenceTransformer('all-MiniLM-L6-v2')

s1="Zomato delivers food"
s2="Swiggy delivers food"
s3="Python programming"

e1=model.encode(s1)
e2=model.encode(s2)
e3=model.encode(s3)

sim1=util.cos_sim(e1,e2).item()
sim2=util.cos_sim(e1,e3).item()

print(f"Zomato vs Swiggy: {sim1:.3f}")
print(f"Zomato vs Python: {sim2:.3f}")