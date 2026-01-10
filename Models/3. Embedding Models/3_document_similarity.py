from langchain_openai import OpenAIEmbeddings   
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embedding = OpenAIEmbeddings(model="text-embedding-3-large",
                 dimensions=300)

documents = [
        "Virat Kohli is a famous Indian cricketer.",
        "MS Dhoni is known for his calm demeanor on the field.",
        "Sachin Tendulkar is regarded as one of the greatest batsmen in cricket history",
        "Rohit Sharma is the captain of the Indian cricket team.",
        "Shreyas Iyer is a talented opening batsman."
]

query = "Tell me about Virat Kohli."

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
similarities = cosine_similarity(
    [query_embedding], doc_embeddings
)[0]
index,score = sorted(list(enumerate(similarities)),key=lambda x: x[1])[-1]
print(documents[index])
print("Similarity Score:",score)    
