from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .vector_store_service import VectorStoreService
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from datetime import datetime
import os
import ssl
import nltk
import logging

logger = logging.getLogger(__name__)

# Fix NLTK SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)

PROPRIETARY_FRAMEWORK_DATA_DIR = os.getenv("PROPRIETARY_FRAMEWORK_DATA_DIR", "data/raw/PL_2338-2023.pdf")

vectorStoreService = VectorStoreService()


def ingest_proprietary_framework(chunk_size: int = 1000, chunk_overlap: int = 200):
    collection_name = "proprietary_framework"
    logger.info("Starting Proprietary Framework ingestion (collection: %s)", collection_name)
    
    vector_db = vectorStoreService.ingest_documents([], collection_name=collection_name)
    existing_data = vector_db.get()
    
    if existing_data and existing_data.get("ids") and len(existing_data["ids"]) > 0:
        logger.info("Collection '%s' already exists with %d documents. Skipping ingestion.", collection_name, len(existing_data['ids']))
        return vector_db

    logger.info("Collection '%s' is empty. Ingesting from %s...", collection_name, PROPRIETARY_FRAMEWORK_DATA_DIR)
    
    file_extension = os.path.splitext(PROPRIETARY_FRAMEWORK_DATA_DIR)[1].lower()
    
    if file_extension == '.pdf':
        loader = UnstructuredPDFLoader(PROPRIETARY_FRAMEWORK_DATA_DIR, mode="elements", strategy="fast", languages=["por"]) # strategy="fast" avoids OCR/image processing
    elif file_extension in ['.docx', '.doc']:
        loader = UnstructuredWordDocumentLoader(PROPRIETARY_FRAMEWORK_DATA_DIR)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only PDF and Word documents are supported.")

    docs_unstructured = loader.load()
    logger.info("Loaded %d raw elements from document", len(docs_unstructured))
    structured_docs = []
    for doc in docs_unstructured:
        doc.metadata['source'] = 'proprietary_framework.pdf'
        doc.metadata['ingestion_date'] = datetime.now().strftime('%Y-%m-%d')
        doc.metadata['data_owner'] = 'PL 2338/2023'
        structured_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(structured_docs)
    split_docs = filter_complex_metadata(split_docs)
    logger.info("Split into %d chunks (chunk_size=%d, overlap=%d)", len(split_docs), chunk_size, chunk_overlap)

    vector_db = vectorStoreService.ingest_documents(split_docs, collection_name=collection_name)
    logger.info("Proprietary Framework ingestion complete")
    return vector_db


def ingest_norm_document(file_path: str, norm_id: str, norm_title: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Ingest a single norm file into the proprietary_framework collection."""
    collection_name = "proprietary_framework"
    logger.info("Ingesting norm '%s' (id=%s) from %s", norm_title, norm_id, file_path)

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="fast", languages=["por"])
    elif file_extension in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    docs_raw = loader.load()
    for doc in docs_raw:
        doc.metadata["source"] = norm_title
        doc.metadata["norm_id"] = norm_id
        doc.metadata["ingestion_date"] = datetime.now().strftime("%Y-%m-%d")
        doc.metadata["data_owner"] = norm_title

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs_raw)
    split_docs = filter_complex_metadata(split_docs)

    vectorStoreService.add_documents_to_collection(split_docs, collection_name=collection_name)
    logger.info("Norm '%s' ingested: %d chunks added", norm_title, len(split_docs))

    # Invalidate the framework RAG singleton so next query rebuilds with new docs
    from ..tools.rags.framework_rag import reset_instance as reset_framework_rag
    reset_framework_rag()

    



