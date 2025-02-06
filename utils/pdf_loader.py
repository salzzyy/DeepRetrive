# load and process the vector DB
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


def load_and_chunk_pdf(pdf_path):
    """Extracts text from a PDF and chunks it using SemanticChunker."""
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()

    # Initialize semantic chunker
    chunker = SemanticChunker(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    chunks = chunker.split_documents(documents)
    return chunks
