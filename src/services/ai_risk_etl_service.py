from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .vector_store_service import VectorStoreService
from datetime import datetime
import os
import csv
import logging

logger = logging.getLogger(__name__)

AI_RISK_DATA_DIR = os.getenv("AI_RISK_DATA_DIR", "data/raw/ai_risk_database_v3.csv")

vectorStoreService = VectorStoreService()


def ingest_ai_risk_csv(chunk_size: int = 1000, chunk_overlap: int = 200):
    logger.info("Starting AI Risk CSV ingestion from %s", AI_RISK_DATA_DIR)
    processed_docs = []

    with open(AI_RISK_DATA_DIR, "r", encoding="utf-8") as csvfile: 
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata = {
                'source': 'ai_risk_database_v3.csv',
                'ingestion_date': datetime.now().strftime('%Y-%m-%d'),
                'data_owner': 'AI Ethics Team',
                'title': row.get('Title', ''),
                'risk_category': row.get('Risk category', ''),
                'risk_subcategory': row.get('Risk subcategory', ''),
                'entity': row.get('Entity', ''),
                'intent': row.get('Intent', ''),
                'timing': row.get('Timing', ''),
                'domain': row.get('Domain', ''),
                'sub_domain': row.get('Sub-domain', ''),
                'quick_ref': row.get('QuickRef', ''),
                'ev_id': row.get('Ev_ID', '')
            }

            content_parts = []

            if row.get('Title'):
                content_parts.append(f"Title: {row['Title']}")
            if row.get('Risk category'):
                content_parts.append(f"Risk category: {row['Risk category']}")
            if row.get('Risk subcategory'):
                content_parts.append(f"Risk subcategory: {row['Risk subcategory']}")
            if row.get('Description'):
                content_parts.append(f"Description: {row['Description']}")
            if row.get('Additional ev.'):
                content_parts.append(f"Additional ev.: {row['Additional ev.']}")

            page_content = "\n".join(content_parts) 

            if page_content:
                processed_docs.append(Document(page_content=page_content, metadata=metadata))
    
    logger.info("Parsed %d documents from CSV", len(processed_docs))

    text_spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_spliter.split_documents(processed_docs)
    logger.info("Split into %d chunks (chunk_size=%d, overlap=%d)", len(split_docs), chunk_size, chunk_overlap)

    vector_db = vectorStoreService.ingest_documents(split_docs, collection_name="ai_risk_database_v3")
    logger.info("AI Risk ingestion complete")
    return vector_db