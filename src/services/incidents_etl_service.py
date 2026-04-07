import os
import csv
import json
import ast
import logging
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

INCIDENTS_DATA_DIR = os.getenv("INCIDENTS_DATA_DIR", "data/raw/incidents.csv")

vectorStoreService = VectorStoreService()


def ingest_incidents_csv(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    reset: bool = False,
):
    logger.info("Starting Incidents CSV ingestion from %s", INCIDENTS_DATA_DIR)
    processed_docs = []

    with open(INCIDENTS_DATA_DIR, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Parse the raw report number IDs from the CSV and store as-is
            report_ids: list[int] = []
            reports_str = row.get("reports", "")
            if reports_str:
                try:
                    if reports_str.startswith("["):
                        parsed = ast.literal_eval(reports_str)
                    else:
                        parsed = [int(r.strip()) for r in reports_str.split(",") if r.strip()]
                    report_ids = [int(x) for x in parsed if str(x).isdigit()]
                except Exception as exc:
                    logger.warning("Error parsing report IDs for incident %s: %s", row.get("incident_id"), exc)

            metadata = {
                "source": "incidents.csv",
                "ingestion_date": datetime.now().strftime("%Y-%m-%d"),
                "data_owner": "AIID",
                "id": row.get("_id", ""),
                "incident_id": row.get("incident_id", ""),
                "incident_date": row.get("date", ""),
                "deployer": row.get("Alleged deployer of AI system", ""),
                "developer": row.get("Alleged developer of AI system", ""),
                "harmed_parties": row.get("Alleged harmed or nearly harmed parties", ""),
                "title": row.get("title", ""),
                # Raw report_number IDs — looked up at query time via /api/v1/reports
                "report_ids": json.dumps(report_ids),
            }

            content_parts = []
            for field, label in [
                ("title", "Title"),
                ("description", "Description"),
                ("Alleged deployer of AI system", "Deployer"),
                ("Alleged developer of AI system", "Developer"),
                ("Alleged harmed or nearly harmed parties", "Harmed Parties"),
            ]:
                if row.get(field):
                    content_parts.append(f"{label}: {row[field]}")

            if content_parts:
                processed_docs.append(Document(page_content="\n".join(content_parts), metadata=metadata))

    logger.info("Parsed %d documents from CSV", len(processed_docs))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(processed_docs)
    logger.info("Split into %d chunks (chunk_size=%d, overlap=%d)", len(split_docs), chunk_size, chunk_overlap)

    vector_db = vectorStoreService.ingest_documents(split_docs, collection_name="incidents_database", reset=reset)
    logger.info("Incidents ingestion complete")
    return vector_db
