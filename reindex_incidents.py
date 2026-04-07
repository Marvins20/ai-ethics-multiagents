"""Run once to wipe and re-ingest the incidents Chroma collection with correct report_ids metadata."""
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from src.services.incidents_reports_etl_service import create_reports_table
from src.services.incidents_etl_service import ingest_incidents_csv

print("Initializing reports DuckDB table...")
create_reports_table()

print("Re-indexing incidents (this may take a few minutes)...")
ingest_incidents_csv(reset=True)
print("Done — incidents_database rebuilt with report_ids metadata.")
