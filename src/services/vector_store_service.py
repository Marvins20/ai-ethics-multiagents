import logging
from langchain_core.documents import Document
from langchain_chroma import Chroma
from pathlib import Path
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from typing import Dict, Optional

from ..config import settings

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parents[2] / "data" / "chroma")
EMBEDDING_MODEL_NAME = settings.embedding_model_name

class VectorStoreService:
    _instance = None
    _collections: Dict[str, Chroma] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            cls._instance._collections = {}
        return cls._instance
    
    def ingest_documents(self, documents: list[Document], collection_name: str, reset: bool = False) -> Chroma:
        if reset:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
                client.delete_collection(collection_name)
                logger.info("Deleted Chroma collection '%s' for reset", collection_name)
            except Exception as exc:
                logger.warning("Could not delete collection '%s': %s", collection_name, exc)
            self._collections.pop(collection_name, None)

        if not documents:
            # No documents provided — just open (or create) the collection and return it
            if collection_name not in self._collections:
                self._collections[collection_name] = Chroma(
                    collection_name=collection_name,
                    embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=settings.google_api_key),
                    persist_directory=CHROMA_PERSIST_DIR
                )
            return self._collections[collection_name]

        if collection_name not in self._collections:
            self._collections[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=settings.google_api_key),
                persist_directory=CHROMA_PERSIST_DIR
            )
        
        # Check if collection is empty before adding documents to avoid duplicates and re-embedding cost
        # Using internal collection checking or a simple query
        try:
             # Check if we have any documents
             existing_count = self._collections[collection_name]._collection.count() # Access internal chroma collection
             if existing_count == 0 and documents:
                 print(f"Ingesting {len(documents)} documents into {collection_name}...")
                 self._collections[collection_name].add_documents(documents)
             elif documents:
                 print(f"Collection {collection_name} already has {existing_count} documents. Skipping ingestion.")
        except Exception as e:
            # Fallback if count check fails
            print(f"Warning: Could not check collection size: {e}. Proceeding with ingestion.")
            if documents:
                self._collections[collection_name].add_documents(documents)
            
        return self._collections[collection_name]

    def get_collection(self, collection_name: str) -> Optional[Chroma]: 
        if collection_name in self._collections:
            return self._collections[collection_name]
        return None
    
    def add_documents_to_collection(self, documents: list[Document], collection_name: str) -> Chroma:
        """Add documents to a collection unconditionally (no skip-if-exists check)."""
        if collection_name not in self._collections:
            self._collections[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=settings.google_api_key),
                persist_directory=CHROMA_PERSIST_DIR
            )
        if documents:
            self._collections[collection_name].add_documents(documents)
        return self._collections[collection_name]

    def clear_cache(self):
        self._collections.clear()