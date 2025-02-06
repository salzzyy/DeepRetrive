# this module create and manage the vector DB


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os


def create_vector_store(chunks, db_path="faiss_index"):
    """Creates and saves a FAISS vector database."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save vector store
    vector_store.save_local(db_path)
    print("Vector store saved.")


def load_vector_store():
    # Set the correct path to your FAISS DB folder
    db_path = "D:/DeepRetrive/faiss_index"  # Replace with your actual FAISS DB folder path

    # Ensure the index.faiss file exists in this folder
    index_file_path = os.path.join(db_path, "index.faiss")
    if not os.path.exists(index_file_path):
        raise ValueError(f"FAISS index file not found at {index_file_path}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the FAISS index from the specified path
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

