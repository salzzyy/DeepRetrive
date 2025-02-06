from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_and_save_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    faiss_index = FAISS.from_documents(chunks, embeddings)

    # Specify the path to save the FAISS index
    index_path = "D:/DeepRetrive/faiss_index"  # Replace with your desired path
    faiss_index.save_local(index_path)
    print(f"FAISS index saved at {index_path}")
