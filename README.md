# 💬 Chat with PDF using Gemini (Google Generative AI + LangChain)

This project allows users to **chat with their PDF files** using natural language. It uses **LangChain**, **Google Gemini (Generative AI)**, **FAISS** for vector search, and **Streamlit** for the frontend interface.

---
🚀 Features

📄 Upload and process multiple PDF files

🔍 Ask questions and receive detailed, context-aware answers from the uploaded PDFs

🤖 Powered by LLaMA-3.1 (hosted on Groq) for fast and high-quality LLM responses

🔗 Uses LangChain + FAISS for efficient vector-based document retrieval

🧠 Semantic chunking and embeddings using HuggingFace sentence transformers for accurate context matching

☁️ Fully deployable on Streamlit Cloud (no local models required)

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Generative AI (Gemini API)
- FAISS (vector database)
- PyPDF2
- dotenv

---

## 📂 Project Structure

```bash
chat_with_pdf/
├── chat.py                # Main Streamlit app
├── faiss_index/           # Saved vector store (auto-created)
├── .env                   # Stores Google API key
├── README.md              # Project description
└── requirements.txt       # List of dependencies
