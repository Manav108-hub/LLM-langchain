"""Document processing using LangChain components."""

import logging
from pathlib import Path
from typing import List, Optional
import hashlib

from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from core.ollama_llm import create_ollama_embeddings
from config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, processing, and vector store management using LangChain."""

    def __init__(self):
        """Initialize the document processor with embeddings and text splitter."""
        # Choose embedding provider
        if config.get("embedding.provider") == "ollama":
            self.embeddings = create_ollama_embeddings()
        else:
            # Default to HuggingFace
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.get("embedding.model", "sentence-transformers/all-MiniLM-L6-v2"),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': config.get("embedding.normalize", True)}
            )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("document_processing.chunk_size", 1000),
            chunk_overlap=config.get("document_processing.chunk_overlap", 200),
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )

        self.vector_store: Optional[FAISS] = None
        self.processed_files = set()

        # Try to load existing vector store
        self._load_existing_vector_store()

    def _load_existing_vector_store(self) -> None:
        """Load existing vector store if available."""
        try:
            vector_store_path = config.get_paths()["data"] / config.get("vector_store.path", "vector_store")
            if vector_store_path.exists():
                self.vector_store = FAISS.load_local(
                    str(vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing vector store")
            else:
                logger.info("No existing vector store found")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.vector_store = None

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to track processing status."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_document(self, file_path: Path) -> List[Document]:
        """Load document using appropriate LangChain loader."""
        file_extension = file_path.suffix.lower()

        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_extension == '.docx':
                loader = Docx2txtLoader(str(file_path))
            elif file_extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()

            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'filename': file_path.name,
                    'file_type': file_extension
                })

            return documents

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise

    def process_document(self, file_path: Path) -> bool:
        """Process a single document and add to vector store."""
        try:
            # Check if file is supported
            allowed_extensions = set(config.get("document_processing.allowed_extensions", [".pdf", ".docx", ".txt"]))
            if file_path.suffix.lower() not in allowed_extensions:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            # Check if already processed
            file_hash = self._get_file_hash(file_path)
            if file_hash in self.processed_files:
                logger.info(f"Document {file_path.name} already processed")
                return True

            # Load document
            documents = self._load_document(file_path)
            if not documents:
                logger.warning(f"No content found in {file_path.name}")
                return False

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            if not chunks:
                logger.warning(f"No chunks created from {file_path.name}")
                return False

            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")

            # Add to vector store
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                # Add to existing vector store
                self.vector_store.add_documents(chunks)

            # Save vector store
            self._save_vector_store()

            # Mark as processed
            self.processed_files.add(file_hash)

            logger.info(f"Successfully processed {file_path.name}")
            return True

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return False

    def process_multiple_documents(self, file_paths: List[Path]) -> dict:
        """Process multiple documents and return results."""
        results = {
            'successful': [],
            'failed': [],
            'total_chunks': 0
        }

        for file_path in file_paths:
            try:
                if self.process_document(file_path):
                    results['successful'].append(str(file_path))
                else:
                    results['failed'].append(str(file_path))
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results['failed'].append(str(file_path))

        # Count total chunks in vector store
        if self.vector_store:
            results['total_chunks'] = self.vector_store.index.ntotal

        return results

    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents using similarity search."""
        if not self.vector_store:
            logger.warning("No vector store available for search")
            return []

        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} relevant chunks for query")
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Search for relevant documents with similarity scores."""
        if not self.vector_store:
            logger.warning("No vector store available for search")
            return []

        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} relevant chunks with scores")
            return results

        except Exception as e:
            logger.error(f"Error during search with scores: {e}")
            return []

    def _save_vector_store(self) -> None:
        """Save vector store to disk."""
        try:
            if self.vector_store:
                vector_store_path = config.get_paths()["data"] / config.get("vector_store.path", "vector_store")
                vector_store_path.parent.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(str(vector_store_path))
                logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    def get_stats(self) -> dict:
        """Get statistics about processed documents."""
        stats = {
            'total_documents': len(self.processed_files),
            'vector_store_size': 0,
            'embedding_model': getattr(self.embeddings, 'model_name', 'Unknown'),
            'vector_store_type': config.VECTOR_STORE_TYPE
        }

        if self.vector_store:
            stats['vector_store_size'] = self.vector_store.index.ntotal

        return stats

    def clear_vector_store(self) -> bool:
        """Clear all processed documents and vector store."""
        try:
            self.vector_store = None
            self.processed_files.clear()

            # Remove saved vector store
            vector_store_path = config.get_paths()["data"] / config.get("vector_store.path", "vector_store")
            if vector_store_path.exists():
                import shutil
                shutil.rmtree(vector_store_path)

            logger.info("Vector store cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
