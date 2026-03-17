# 📄 RAG Chatbot for PDFs

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDFs and ask questions about them.

## 🚀 Features
- Upload multiple PDFs
- Semantic search using embeddings
- Vector search using FAISS
- LLM-based answers (Llama 3 via Groq)
- Source citations with page numbers
- Streamlit chat interface

## 🛠 Tech Stack
- Python
- SentenceTransformers
- FAISS
- Groq API (Llama 3)
- Streamlit

## ▶️ Run Locally

```bash
git clone <repo-link>
cd rag-pdf-chatbot
pip install -r requirements.txt
streamlit run app.py
