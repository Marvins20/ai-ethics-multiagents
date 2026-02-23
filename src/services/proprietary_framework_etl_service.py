from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .vector_store_service import VectorStoreService
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from datetime import datetime
import os
import csv

PROPRIETARY_FRAMEWORK_DATA_DIR = os.getenv("PROPRIETARY_FRAMEWORK_DATA_DIR", "data/raw/PL_2338-2023.csv")

vectorStoreService = VectorStoreService()


def ingest_proprietary_framework(chunk_size: int = 1000, chunk_overlap: int = 200):
    
    loader = UnstructuredPDFLoader(PROPRIETARY_FRAMEWORK_DATA_DIR, mode="elements")
    docs_unstructured = loader.load()
    structured_docs = []
    for doc in docs_unstructured:
        doc.metadata['source'] = 'proprietary_framework.pdf'
        doc.metadata['ingestion_date'] = datetime.now().strftime('%Y-%m-%d')
        doc.metadata['data_owner'] = 'PL 2338/2023'
        structured_docs.append(doc)

    text_spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_spliter.split_documents(structured_docs)

    vector_db = vectorStoreService.ingest_documents(split_docs, collection_name="reports_database")
    return vector_db

    



