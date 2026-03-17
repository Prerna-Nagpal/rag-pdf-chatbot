import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
import os

load_dotenv()

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize models
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

# Title
st.title("📄 RAG Chatbot for PDFs")

# ================= SIDEBAR UPLOAD =================
st.sidebar.title("📂 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

# Process PDFs
if uploaded_files:

    st.sidebar.write("Processing documents...")

    all_chunks = []

    for uploaded_file in uploaded_files:

        reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text()

            if text:

                chunk_size = 500
                overlap = 100
                start = 0

                while start < len(text):

                    chunk = text[start:start + chunk_size]

                    all_chunks.append({
                        "text": chunk,
                        "page": page_number + 1,
                        "source": uploaded_file.name
                    })

                    start += chunk_size - overlap

    texts = [c["text"] for c in all_chunks]

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    st.session_state.index = index
    st.session_state.chunks = all_chunks

    st.sidebar.success("✅ Documents indexed successfully!")

# ================= CHAT UI =================

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
question = st.chat_input("Ask a question about your PDFs")

if question:

    if st.session_state.index is None:
        st.warning("⚠️ Please upload a PDF first!")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Convert query to embedding
    query_embedding = model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = st.session_state.index.search(query_embedding, 3)

    # Retrieve chunks
    retrieved_chunks = [
        st.session_state.chunks[i]["text"] for i in indices[0]
    ]

    context = "\n".join(retrieved_chunks)

    # Prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the provided context.
If the answer is not present in the context, say:
"I could not find the answer in the document."

Give a clear, concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

    # LLM call
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

        st.markdown("**Sources:**")
        for i in indices[0]:
            chunk = st.session_state.chunks[i]
            st.write(f"{chunk['source']} | page {chunk['page']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})  