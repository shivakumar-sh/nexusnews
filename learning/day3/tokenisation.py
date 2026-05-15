from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Zomato."
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(tokens)
print("Tokens:", tokens)
print("IDs:", ids)

decoded = tokenizer.decode(ids)
print("Decoded:", decoded)
