from ...services.incidents_etl_service import ingest_incidents_csv
from ...services.incidents_reports_etl_service import get_reports_by_ids
from ...services.retrieval_service import get_ensembled_retriever
from langchain_core.tools import tool
import json
import ast

_rag_instance: "IncidentsRAG | None" = None


class IncidentsRAG:
    def __init__(self):
        self.vector_store_service = ingest_incidents_csv()
        self.retriever = get_ensembled_retriever(self.vector_store_service)

    def query(self, query_text: str, top_k: int = 5):
        if not self.retriever:
            self.vector_store_service = ingest_incidents_csv()
            self.retriever = get_ensembled_retriever(self.vector_store_service)
            if not self.retriever:
                return "Error: Retriever could not be initialized."

        results = self.retriever.invoke(query_text, top_k=top_k)
        if not results:
            return "No incidents were found related to this type of query."

        for doc in results:
            reports_meta = doc.metadata.get("reports")
            if reports_meta and isinstance(reports_meta, str):
                try:
                    parsed = None
                    try:
                        parsed = json.loads(reports_meta)
                    except json.JSONDecodeError:
                        try:
                            parsed = ast.literal_eval(reports_meta)
                        except Exception:
                            pass

                    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                        continue

                    if isinstance(parsed, list):
                        data_indices = [int(x) - 2 for x in parsed if str(x).isdigit()]
                        if data_indices:
                            fetched_reports = get_reports_by_ids(data_indices)
                            doc.metadata["reports_details"] = json.dumps(fetched_reports, default=str)
                except Exception as e:
                    print(f"Error enriching report metadata: {e}")

        return results


def get_incidents_rag() -> IncidentsRAG:
    """Return the singleton IncidentsRAG, initializing it on first call."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = IncidentsRAG()
    return _rag_instance


@tool
def search_incidents(project_description: str, action: str, top_k: int = 5):
    """Search for AI incidents in the database based on the project description and specific action.
    This search considers relevant reports linked to the incident.

    Args:
        project_description: The description of the AI project.
        action: The specific action being analyzed for risks.
        top_k: The number of top results to return from the search.
    """
    query = "\n".join(filter(None, [project_description, action]))
    return get_incidents_rag().query(query, top_k)
