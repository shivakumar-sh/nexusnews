from transformers import pipeline
sa=pipeline('sentiment-analysis')
print(sa("The food quality is excellent!"))

ner=pipeline('ner', aggregation_strategy='simple')
print(ner("Infosys is based in Bangalore, ,Karnataka, India."))

qa=pipeline('question-answering')
context = "Python was created by Guido van Rossum and released in 1991. It is widely used in AI."
summ=pipeline('summarization')
long_text="Generative AI refers to AI System that can generate new context..." * 5
print(summ(long_text, max_length=60, min_length=20))