import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# -----------------------------
# STREAMLIT UI SETUP
# -----------------------------
st.set_page_config(page_title="Nova AI Pro", layout="centered")
st.title("🤖 AI - Chatbot")
st.sidebar.title("AI Chatbot")

if st.sidebar.button("Clear Chat"):
    st.session_state.chat = []
    st.rerun()
# -----------------------------
# LOAD LLM (PRETRAINED MODEL)
# -----------------------------
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="gpt2")

llm = load_llm()

# -----------------------------
# EMBEDDING MODEL (NLP)
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# -----------------------------
# LOAD KNOWLEDGE BASE
# -----------------------------
try:
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        docs = f.readlines()
except FileNotFoundError:
    st.error("Knowledge base file not found. Please ensure 'knowledge_base.txt' exists.")
    st.stop()
doc_embeddings = embedder.encode(docs)
# -----------------------------
# FAISS INDEX (FAST SEARCH)
# -----------------------------
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# -----------------------------
# RETRIEVAL FUNCTION (RAG)
# -----------------------------
def retrieve_context(query):
    q_emb = embedder.encode([query])
    _, idx = index.search(np.array(q_emb), k=2)
    return " ".join([docs[i] for i in idx[0]])

# -----------------------------
# CHAT FUNCTION
# -----------------------------
def generate_response(user_input):

    context = retrieve_context(user_input)

    prompt = f"""
You are Nova AI, a helpful assistant.

Context:
{context}

User: {user_input}
AI:
"""

    response = llm(
        prompt,
        max_length=120,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )[0]["generated_text"]

    return response.split("AI:")[-1].strip()
# -----------------------------
# SESSION MEMORY
# -----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.text_input("Ask something:")

if user_input and user_input.strip():

    with st.spinner("Nova is thinking..."):
        reply = generate_response(user_input)

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Nova", reply))

# -----------------------------
# CHAT HISTORY
# -----------------------------
for role, msg in st.session_state.chat[::-1]:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Nova:** {msg}")

