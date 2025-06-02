
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from pinecone import Pinecone  # SDK client
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Initialize embedding and LLM
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  # 👈 explicitly tell it which key to track
)


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
        return_source_documents=True
    )

def answer_question_conversationally(question: str) -> dict:
    chain = get_conversational_chain()
    result = chain({"question": question})
    return {
        "answer": result["answer"],
        "sources": [
            {
                "file": doc.metadata.get("source", "unknown"),
                "excerpt": doc.page_content.strip()[:600]
            }
            for doc in result["source_documents"]
        ]
    }
