import streamlit as st
import os
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error(
        "❌ GROQ_API_KEY not found.\n\n"
        "• Local: add it to a `.env` file\n"
        "• Streamlit Cloud: add it in App → Settings → Secrets"
    )
    st.stop()

# --------------------------------------------------
# PDF text extraction
# --------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# --------------------------------------------------
# Text chunking
# --------------------------------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )
    return splitter.split_text(text)

# --------------------------------------------------
# Create vector store (FREE embeddings)
# --------------------------------------------------
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("faiss_index")

# --------------------------------------------------
# Groq QA Chain (VERBOSE OUTPUT)
# --------------------------------------------------
def get_qa_chain():
    prompt_template = """
    You are an expert assistant.

    Using ONLY the information from the context below,
    answer the question in a clear, detailed, and well-structured manner.

    Guidelines:
    - Explain concepts step by step when applicable
    - Use multiple paragraphs if needed
    - Use bullet points or numbered lists where helpful
    - Do NOT add information outside the context
    - If the answer is not present in the context, say:
      "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.4,
        max_tokens=1024
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# --------------------------------------------------
# Answer user question
# --------------------------------------------------
def answer_question(question):
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Please upload and process PDFs first.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve more context for richer answers
    docs = db.similarity_search(question, k=8)

    chain = get_qa_chain()

    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )

    st.subheader("📌 Answer")
    st.write(response["output_text"])

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="Chat with PDF (Groq LLaMA-3.1)",
        layout="wide"
    )

    st.title("📄 Chat with PDF using LLaMA-3.1 (Groq)")

    user_question = st.text_input("Ask a question from the uploaded PDFs")

    if user_question:
        answer_question(user_question)

    with st.sidebar:
        st.header("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload one or more PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No readable text found in the PDFs.")
                        return
                    chunks = get_text_chunks(raw_text)
                    create_vector_store(chunks)
                    st.success("✅ PDFs processed successfully!")

# --------------------------------------------------
if __name__ == "__main__":
    main()
