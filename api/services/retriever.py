import os
import streamlit as st
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# Load secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", st.secrets.get("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", st.secrets.get("PINECONE_INDEX_NAME"))

# Initialize models
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Setup Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_conversational_chain():
    vectorstore = LangchainPinecone(index, embedding_model, text_key="page_content")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert semiconductor manufacturing assistant. Use the context below to answer the question.
If you don't know the answer, say "I don't know".

Context: {context}

Question: {question}

Answer:""")

    document_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        document_variable_name="context"
    )

    qa_chain = RetrievalQAWithSourcesChain(
        combine_documents_chain=document_chain,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

def answer_question_conversationally(question: str) -> dict:
    chain = get_conversational_chain()
    result = chain({"question": question})
    answer = result["answer"]
    sources = result.get("source_documents", [])

    extracted_sources = [
        {
            "file": doc.metadata.get("source", "Unknown"),
            "excerpt": doc.page_content[:500]
        }
        for doc in sources
    ]

    return {"answer": answer, "sources": extracted_sources}
