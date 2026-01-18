# 💬 Chat with PDF using LLaMA-3.1 (Groq + LangChain)

This project allows users to chat with their PDF files using natural language queries.
It is built using LangChain, LLaMA-3.1 (hosted on Groq) for large language model inference, FAISS for vector-based document retrieval, and Streamlit for an interactive web interface.

The application follows a Retrieval-Augmented Generation (RAG) approach, enabling accurate and context-aware answers strictly based on the content of the uploaded PDF documents.
---
🚀 Features

📄 Upload and process multiple PDF files

🔍 Ask questions and receive detailed, context-aware answers from the uploaded PDFs

🤖 Powered by LLaMA-3.1 (hosted on Groq) for fast and high-quality LLM responses

🔗 Uses LangChain + FAISS for efficient vector-based document retrieval

🧠 Semantic chunking and embeddings using HuggingFace sentence transformers for accurate context matching

☁️ Fully deployable on Streamlit Cloud (no local models required)

## 🛠️ Tech Stack

Python

Streamlit

LangChain

Groq API (LLaMA-3.1)

FAISS (vector database)

HuggingFace Sentence Transformers (embeddings)

PyPDF

python-dotenv

---

## 📂 Project Structure

```bash
chat_with_pdf/
├── chat.py                # Main Streamlit application
├── faiss_index/           # FAISS vector store (auto-created after processing PDFs)
├── .env                   # Stores Groq API key (local use only, not committed)
├── .gitignore             # Excludes .env and other sensitive files
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
