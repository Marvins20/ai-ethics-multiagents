from ...services.incidents_etl_service import ingest_incidents_csv
from ...services.incidents_reports_etl_service import get_reports_by_ids
from ...services.retrieval_service import get_ensembled_retriever
from langchain_core.tools import tool
import json
import ast

class IncidentsRAG:
    def __init__(self):
        self.vector_store_service = ingest_incidents_csv()
        self.retriever = get_ensembled_retriever(self.vector_store_service)
    
    def query(self, query_text: str, top_k: int = 5):
        if not self.retriever:
            # Re-initialize if retriever is None (e.g. if vector store was empty initially)
            self.vector_store_service = ingest_incidents_csv()
            self.retriever = get_ensembled_retriever(self.vector_store_service)
            if not self.retriever:
                return "Error: Retriever could not be initialized."
        
        results = self.retriever.invoke(query_text, top_k=top_k)
        if not results:
            return "No incidents were found related to this type of query."
        
        # Enrich results with report details if explicitly needed or missing
        # ingest_incidents_csv attempts to populate metadata['reports']
        # But this backup logic ensures we can get them if we only have the IDs string in metadata
        for doc in results:
            reports_meta = doc.metadata.get('reports')
            if reports_meta and isinstance(reports_meta, str):
                try:
                    # Check if it is a formatted list string
                    parsed = None
                    try:
                         # Try parsing as JSON first (might be serialized objects or list of ints)
                        parsed = json.loads(reports_meta)
                    except json.JSONDecodeError:
                        # Try literal eval if JSON fails (e.g. single quotes)
                        try:
                            parsed = ast.literal_eval(reports_meta)
                        except:
                            pass
                            
                    # Case 1: Already has report objects (list of dicts)
                    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                        continue # Already fully populated
                    
                    # Case 2: List of integers (incident report IDs)
                    # We need to fetch the content using our helper function
                    if isinstance(parsed, list):
                         # Convert 1-based CSV row number to 0-based data index: index = val - 2
                         # Filter out non-digits just in case
                         data_indices = []
                         for x in parsed:
                             if str(x).isdigit():
                                 data_indices.append(int(x) - 2)
                         
                         if data_indices:
                            fetched_reports = get_reports_by_ids(data_indices)
                            # Update metadata with full details
                            doc.metadata['reports_details'] = json.dumps(fetched_reports, default=str)
                except Exception as e:
                    print(f"Error enriching report metadata: {e}")

        return results

_rag_instance = IncidentsRAG()

@tool
def search_incidents(project_description: str, action: str, top_k: int = 5):
    """Search for AI incidents in the database based on the project description and specific action.
    This search considers relevant reports linked to the incident.
    
    Args:
        project_description: The description of the AI project.
        action: The specific action being analyzed for risks.
        top_k: The number of top results to return from the search.
    """
    # Create a semantic query combining project context and action
    query = f"Project context: {project_description}. Action: {action}. Find relevant AI incidents and failures."
    return _rag_instance.query(query, top_k)
