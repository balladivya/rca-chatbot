import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def load_and_chunk_pdfs(pdf_dir: str) -> List[Document]:
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            loader = PyMuPDFLoader(path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file

            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

    return all_chunks

