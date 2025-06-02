import os
import streamlit as st
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load API keys from Streamlit Cloud or local .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", st.secrets.get("PINECONE_INDEX_NAME"))

# Initialize components
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_conversational_chain():
    vectorstore = LangchainPinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_model
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

def answer_question_conversationally(question: str) -> dict:
    chain = get_conversational_chain()
    result = chain({"question": question})
    answer = result["answer"]
    sources = result.get("source_documents", [])

    extracted_sources = []
    for doc in sources:
        extracted_sources.append({
            "file": doc.metadata.get("source", "Unknown"),
            "excerpt": doc.page_content[:500]
        })

    return {"answer": answer, "sources": extracted_sources}
