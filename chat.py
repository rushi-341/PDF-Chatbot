import streamlit as st
import os

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --------------------------------------------------
# Config
# --------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# --------------------------------------------------
# PDF Processing
# --------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    docs = [Document(page_content=c) for c in chunks]
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# --------------------------------------------------
# LLM Chain (LCEL – Modern LangChain)
# --------------------------------------------------
def get_chain():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present, say:
"Answer is not available in the provided context."

Context:
{context}

Question:
{question}

Answer (detailed and structured):
""")

    return prompt | llm

# --------------------------------------------------
# Question Answering
# --------------------------------------------------
def answer_question(question):
    db = load_vector_store()
    docs = db.similarity_search(question, k=4)

    context = "\n\n".join(doc.page_content for doc in docs)

    chain = get_chain()
    response = chain.invoke({
        "context": context,
        "question": question
    })

    st.write("### Answer")
    st.write(response.content)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
def main():
    st.set_page_config(page_title="Chat with PDF using LLaMA-3 (Groq)")
    st.header("📄 Chat with PDF using LLaMA-3 (Groq)")

    user_question = st.text_input("Ask a question from the uploaded PDFs")

    if user_question:
        answer_question(user_question)

    with st.sidebar:
        st.title("📂 Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in the PDFs.")
                    return

                chunks = get_text_chunks(raw_text)
                create_vector_store(chunks)

            st.success("PDFs processed successfully!")

# --------------------------------------------------
if __name__ == "__main__":
    main()
