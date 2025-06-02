import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. 'us-west4-gcp-free'
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Get index object
index = pc.Index(PINECONE_INDEX_NAME)

def upload_to_pinecone(vectors: list, batch_size: int = 100):
    print(f"🚀 Uploading {len(vectors)} vectors to Pinecone index: {PINECONE_INDEX_NAME}...")

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"✅ Uploaded batch {i // batch_size + 1}")

    print("🎉 All chunks uploaded to Pinecone.")

