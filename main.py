import streamlit as st
import os
from utils.pdf_loader import load_and_chunk_pdf
from utils.vector_store import create_vector_store
from utils.rag_pipeline import get_rag_response
from utils.create_faiss import create_and_save_faiss_index

# Ensure the directory exists
uploaded_pdfs_dir = "uploaded_pdfs"
os.makedirs(uploaded_pdfs_dir, exist_ok=True)

st.title("📄 AI-Powered Document Q&A")

# Upload PDF Section
st.subheader("Upload a PDF Document")
pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])

if pdf_file is not None:
    # Save the uploaded file
    pdf_path = os.path.join(uploaded_pdfs_dir, pdf_file.name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    st.success("PDF uploaded successfully!")

    # Process PDF
    with st.spinner("Processing document..."):
        chunks = load_and_chunk_pdf(pdf_path)
        create_vector_store(chunks)

    st.success("Document processed and stored in vector database!")

# Question Answering Section
st.subheader("Ask a Question")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Fetching answer..."):
        response = get_rag_response(query)

    st.write("### Answer:")
    st.write(response)
