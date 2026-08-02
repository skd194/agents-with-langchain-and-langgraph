from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# single text embedding
text = "This is a test document."
embedding = embeddings.embed_query(text)
print(f"Embedding for '{text}': {embedding}")

print("\nEmbedding vector length:", len(embedding)) 


# multiple text embeddings
texts = [
    "This is the first test document.",
    "This is the second test document.",
    "This is the third test document."
]   

embeddings_list = embeddings.embed_documents(texts)
print(f"\nEmbeddings for '{texts}': {embeddings_list}")

print("\nEmbedding vector lengths:", [len(e) for e in embeddings_list])

