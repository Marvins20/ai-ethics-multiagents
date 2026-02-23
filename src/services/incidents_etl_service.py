from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .vector_store_service import VectorStoreService
from .incidents_reports_etl_service import get_reports_by_ids
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from datetime import datetime
import os
import csv
import json
import ast

INCIDENTS_DATA_DIR = os.getenv("INCIDENTS_DATA_DIR", "data/raw/incidents.csv")

vectorStoreService = VectorStoreService()


def ingest_incidents_csv(chunk_size: int = 1000, chunk_overlap: int = 200):
    processed_docs = []

    with open(INCIDENTS_DATA_DIR, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reports_data = []
            reports_str = row.get('reports', '')
            if reports_str:
                try:
                    if reports_str.startswith('['):
                        report_indices = ast.literal_eval(reports_str)
                    else:
                        report_indices = [int(r.strip()) for r in reports_str.split(',') if r.strip()]
                    
                    if isinstance(report_indices, (list, tuple)):
                        # Convert 1-based CSV row number (accounting for header) to 0-based data index
                        data_indices = [int(idx) - 2 for idx in report_indices if str(idx).isdigit()]
                        if data_indices:
                            reports_data = get_reports_by_ids(data_indices)
                except Exception as e:
                    print(f"Error parsing reports for incident {row.get('incident_id')}: {e}")

            metadata = {
                'source': 'incidents.csv',
                'ingestion_date': datetime.now().strftime('%Y-%m-%d'),
                'data_owner': 'AIID',
                'id': row.get('_id', ''),
                'incident_id': row.get('incident_id', ''),
                'incident_date': row.get('date', ''),
                'deployer': row.get('Alleged deployer of AI system', ''),
                'developer': row.get('Alleged developer of AI system', ''),
                'harmed_parties': row.get('Alleged harmed or nearly harmed parties', ''),
                'title': row.get('title', ''),
                'reports': json.dumps(reports_data)
            }

            content_parts = []

            if row.get('title'):
                content_parts.append(f"Title: {row['title']}")
            if row.get('description'):
                content_parts.append(f"Description: {row['description']}")
            if row.get('Alleged deployer of AI system'):
                content_parts.append(f"Deployer: {row['Alleged deployer of AI system']}")
            if row.get('Alleged developer of AI system'):
                content_parts.append(f"Developer: {row['Alleged developer of AI system']}")
            if row.get('Alleged harmed or nearly harmed parties'):
                content_parts.append(f"Harmed Parties: {row['Alleged harmed or nearly harmed parties']}")

            page_content = "\n".join(content_parts)

            if page_content:
                processed_docs.append(Document(page_content=page_content, metadata=metadata))
    
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_spliter.split_documents(processed_docs)

    vector_db = vectorStoreService.ingest_documents(split_docs, collection_name="incidents_database")
    return vector_db