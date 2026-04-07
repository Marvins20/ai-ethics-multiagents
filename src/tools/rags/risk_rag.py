from ...services.ai_risk_etl_service import ingest_ai_risk_csv
from ...services.retrieval_service import get_ensembled_retriever
from langchain_core.tools import tool

_rag_instance: "RiskRAG | None" = None


class RiskRAG:
    def __init__(self):
        self.vector_store = ingest_ai_risk_csv()
        self.retriever = get_ensembled_retriever(self.vector_store)

    def query(self, query_text: str, top_k: int = 5):
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        results = self.retriever.invoke(query_text, top_k=top_k)
        if not results:
            return "No risks were found related to this type of query."
        return results


def get_risk_rag() -> RiskRAG:
    """Return the singleton RiskRAG, initializing it on first call."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RiskRAG()
    return _rag_instance


@tool
def search_risks(query: str, top_k: int = 5):
    """Search for AI risks in the database based on a query.

    Args:
        query: The search query string describing the risk or topic to look for.
        top_k: The number of top results to return from the search.
    """
    return get_risk_rag().query(query, top_k)
