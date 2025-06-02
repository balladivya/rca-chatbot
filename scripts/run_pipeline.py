
import sys
import os
from dotenv import load_dotenv
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.services.chunker import load_and_chunk_pdfs
from api.services.embedder import generate_embeddings
from api.services.indexer import upload_to_pinecone


# Load environment variables
load_dotenv()

print("🔑 OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
# Define your input folder for PDFs
PDF_DIR = "data"

def main():
    print("📄 Loading and chunking RCA PDFs...")
    chunks = load_and_chunk_pdfs(PDF_DIR)
    print(f"✅ {len(chunks)} chunks created.")

    print("🔍 Generating embeddings...")
    vectors = generate_embeddings(chunks)

    print("🚀 Uploading vectors to Pinecone...")
    upload_to_pinecone(vectors)

    print("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()