from langchain_chroma import Chroma
from pathlib import Path
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from typing import Dict, Optional
import os

CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parents[5] / "data" / "chroma")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001") 

class VectorStoreService:
    _instance = None
    _collections: Dict[str, Chroma] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            cls._instance._collections = {}
        return cls._instance
    
    def ingest_documents(self, documents, collection_name: str) -> Chroma:
        if collection_name not in self._collections:
            self._collections[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME),
                persist_directory=CHROMA_PERSIST_DIR
            )
        self._collections[collection_name].add_documents(documents)
        return self._collections[collection_name]

    def get_collection(self, collection_name: str) -> Optional[Chroma]: 
        if collection_name not in self._collections:
            return self._collections[collection_name]
        return None
    
    def clear_cache(self):
        self._collections.clear()