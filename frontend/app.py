
import os
import sys
import streamlit as st

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.services.retriever import answer_question_conversationally

st.set_page_config(page_title="RCA Chatbot", page_icon="🤖", layout="wide")
st.title("💬 RCA Chatbot")
st.caption("Ask anything about Root Cause Analysis documents in semiconductor manufacturing.")

# Inject CSS to support chat bubbles and wider layout
st.markdown("""
    <style>
    .bubble-container {
        max-width: 80%;
        margin: 0 auto;
    }
    .user-bubble {
        background-color: #1e90ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        text-align: right;
        color: white;
        font-size: 1rem;
    }
    .bot-bubble {
        background-color: #e2e2e2;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        text-align: left;
        color: black;
        font-size: 1rem;
    }
    .source-list {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.25rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""
if "next_query" not in st.session_state:
    st.session_state.next_query = None
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# Reset answer flags
if "answered" in st.session_state:
    del st.session_state.answered
if st.session_state.clear_input:
    st.session_state.pending_query = ""
    st.session_state.clear_input = False

# Sidebar reset button
if st.sidebar.button("🧹 Start New Chat"):
    st.session_state.chat_history = []
    st.session_state.pending_query = ""
    st.session_state.next_query = None
    st.rerun()

# Follow-up suggestions
suggestions = [
    "What was the corrective action?",
    "Was the yield restored?",
    "Was this issue recurring?",
    "Which tool failed most often?"
]

# Display conversation with chat bubble styling
with st.container():
    for i, pair in enumerate(st.session_state.chat_history):
        st.markdown(f'<div class="bubble-container"><div class="user-bubble">🧑‍💼 {pair["question"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble-container"><div class="bot-bubble">🤖 {pair["answer"]}</div></div>', unsafe_allow_html=True)
        if pair["sources"]:
            st.markdown('<div class="bubble-container"><div class="source-list">📁 Sources:<br>' + "<br>".join(f"`{src['file']}`" for src in pair["sources"]) + '</div></div>', unsafe_allow_html=True)

        if i == len(st.session_state.chat_history) - 1:
            st.markdown('<div class="bubble-container"><strong>💡 Try a follow-up:</strong></div>', unsafe_allow_html=True)
            cols = st.columns(len(suggestions))
            for j, suggestion in enumerate(suggestions):
                if cols[j].button(suggestion, key=f"suggestion_{j}"):
                    st.session_state.next_query = suggestion
                    st.session_state.answered = False
                    st.rerun()

# Handle follow-up trigger
if st.session_state.next_query:
    st.session_state.pending_query = st.session_state.next_query
    st.session_state.next_query = None
    st.rerun()

# Input field
query = st.text_input("💬 Type your question here and press Enter:", key="pending_query")

# Handle user input
if query and not st.session_state.get("answered"):
    with st.spinner("Thinking..."):
        result = answer_question_conversationally(query)
        st.session_state.chat_history.append({
            "question": query,
            "answer": result["answer"],
            "sources": result["sources"]
        })
        st.session_state.clear_input = True
        st.session_state.answered = True
        st.rerun()
