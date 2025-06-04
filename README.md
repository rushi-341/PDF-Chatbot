# 💬 Chat with PDF using Gemini (Google Generative AI + LangChain)

This project allows users to **chat with their PDF files** using natural language. It uses **LangChain**, **Google Gemini (Generative AI)**, **FAISS** for vector search, and **Streamlit** for the frontend interface.

---

## 🚀 Features

- 📄 Upload and process multiple PDF files
- 🔍 Ask questions and get detailed answers based on PDF content
- 🤖 Uses Google's Gemini (`gemini-pro`) for LLM responses
- 🔗 Powered by LangChain and FAISS for efficient vector-based document search
- 🧠 Chunking and semantic embedding for accurate context

---

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
