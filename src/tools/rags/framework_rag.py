from ...services.proprietary_framework_etl_service import ingest_proprietary_framework
from ...services.retrieval_service import get_ensembled_retriever
from langchain_core.tools import tool #type: ignore

_rag_instance: "FrameworkRAG | None" = None


class FrameworkRAG:
    def __init__(self):
        self.vector_store = ingest_proprietary_framework()
        self.retriever = get_ensembled_retriever(self.vector_store)

    def query(self, query_text: str, top_k: int = 5):
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        results = self.retriever.invoke(query_text, top_k=top_k)
        if not results:
            return "No framework information was found related to this query."
        return results


def get_framework_rag() -> FrameworkRAG:
    """Return the singleton FrameworkRAG, initializing it on first call."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = FrameworkRAG()
    return _rag_instance


def reset_instance():
    """Invalidate the cached RAG instance so the next query rebuilds it with updated docs."""
    global _rag_instance
    _rag_instance = None
    from ...services.proprietary_framework_etl_service import vectorStoreService
    vectorStoreService._collections.pop("proprietary_framework", None)


@tool
def search_framework(query: str, top_k: int = 5):
    """Search for information in the proprietary framework based on a query.

    Args:
        query: The search query string describing the topic to look for in the framework.
        top_k: The number of top results to return from the search.
    """
    return get_framework_rag().query(query, top_k)


if __name__ == "__main__":
    query = "Como é classificado o risco para sistemas de IA usado no processo de contratação?"
    results = search_framework.invoke({"query": query, "top_k": 5})
    for i, doc in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"Result {i}")
        print(f"{'='*60}")
        print(doc.page_content)
        print(f"\n  [source: {doc.metadata.get('source', 'N/A')} | owner: {doc.metadata.get('data_owner', 'N/A')}]")
