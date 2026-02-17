from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma


def get_ensembled_retriever(collection: Chroma):
    try:
        collection_data = collection.get()
        documents = collection_data['documents']
        metadatas = collection_data['metadatas']

        bm24_retriever = BM25Retriever.from_documents(documents, metadatas=metadatas)
        chroma_retriever = collection.as_retriever()

        ensemble_retriever = EnsembleRetriever(retrievers=[bm24_retriever, chroma_retriever], weights=[0.5, 0.5])
        return ensemble_retriever
    except Exception as e:
        print(f"Error creating retriever: {e}")
        return None
