try:
    from langchain.retrievers import EnsembleRetriever # type: ignore
except ImportError:
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever # type: ignore
    except ImportError:
        try:
            from langchain_community.retrievers import EnsembleRetriever
        except ImportError:
            # Fallback for older or weird installations
            from langchain_classic.retrievers import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma


def get_ensembled_retriever(collection: Chroma,  score_threshold: float = 0.1):
    try:
        collection_data = collection.get()
        doc_contents = collection_data['documents']
        metadatas = collection_data['metadatas']
        
        # collection.get() returns documents as list of strings, but BM25Retriever.from_documents expects Document objects
        if doc_contents:
            documents = [Document(page_content=content, metadata=meta) for content, meta in zip(doc_contents, metadatas)]
            bm25_retriever = BM25Retriever.from_documents(documents)
            chroma_retriever = collection.as_retriever()

            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5])
            return ensemble_retriever
        else:
             print("Warning: Collection is empty. Returning None for retriever.")
             return None
    except Exception as e:
        print(f"Error creating retriever: {e}")
        return None
