from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain.schema import Document
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize embedding model (uses OpenAI API key from .env)
embedder = OpenAIEmbeddings(model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY"))

def generate_embeddings(chunks: List[Document]) -> List[dict]:
    texts = [chunk.page_content for chunk in chunks]
    vectors = embedder.embed_documents(texts)

    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "id": f"{chunk.metadata.get('source', 'unknown')}_{i}",
            "values": vectors[i],
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "")
            }
        })

    return results
