from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
result = generator("The future of AI in India is", max_new_tokens=50, num_return_sequences=1)
print(result[0]['generated_text'])
creative = generator("Once upon a time in Channapatna", max_new_tokens=60, temperature=1.2, do_sample=True)
focused=generator("Once upon a time in Channapatna", max_new_tokens=60, temperature=0.3, do_sample=True)
print("Creative:", creative[0]['generated_text'])
print("Focused:", focused[0]['generated_text'])