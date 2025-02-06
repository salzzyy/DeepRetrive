# RAG logic(retrival +LLM)
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from utils.vector_store import load_vector_store


def get_rag_response(query):
    """Retrieves relevant chunks and generates an answer."""

    vector_store = load_vector_store()  # Load FAISS vector DB
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 chunks

    llm = Ollama(model="deepseek-r1")  # DeepSeek-R1 model

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    response = qa_chain.run(query)
    return response
