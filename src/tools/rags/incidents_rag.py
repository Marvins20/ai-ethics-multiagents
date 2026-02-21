from ...services.incidents_etl_service import ingest_incidents_csv
from ...services.retrieval_service import get_ensembled_retriever
from langchain_core.tools import tool

class IncidentsRAG:
    def __init__(self):
        self.vector_store = ingest_incidents_csv()
        self.retriever = get_ensembled_retriever(self.vector_store)
    
    def query(self, query_text: str, top_k: int = 5):
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        results = self.retriever.invoke(query_text, top_k=top_k)
        if not results:
            return "No incidents were found related to this type of query."
        return results

_rag_instance = IncidentsRAG()

@tool
def search_incidents(query: str, top_k: int = 5):
    """Search for AI incidents in the database based on a query.
    
    Args:
        query: The search query string describing the incident or topic to look for.
        top_k: The number of top results to return from the search.
    """
    return _rag_instance.query(query, top_k)