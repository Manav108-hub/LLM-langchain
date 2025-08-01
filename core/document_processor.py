"""Document processing using LangChain components."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import config
from core.ollama_llm import create_ollama_embeddings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, processing, and vector store management using LangChain."""
    
    def __init__(self):  # Fixed: double underscores
        """Initialize the document processor with embeddings and text splitter."""
        try:
            # Choose embedding provider
            if config.get("embedding.provider") == "ollama":
                logger.info("Using Ollama embeddings...")
                self.embeddings = create_ollama_embeddings()
            else:
                # Default to HuggingFace
                logger.info("Using HuggingFace embeddings...")
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
            
            # Initialize vector store and processed files tracking
            self.vector_store: Optional[FAISS] = None
            self.processed_files = set()
            
            # Try to load existing vector store
            self._load_existing_vector_store()
            
            logger.info("✅ DocumentProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing DocumentProcessor: {e}")
            # Initialize with None values to prevent crashes
            self.vector_store = None
            self.processed_files = set()
            raise
    
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
                logger.info("✅ Loaded existing vector store")
            else:
                logger.info("ℹ No existing vector store found - will create new one when documents are uploaded")
        except Exception as e:
            logger.error(f"⚠ Error loading vector store: {e}")
            self.vector_store = None
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to track processing status."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash for {file_path}: {e}")
            return str(file_path)  # Fallback to filename
    
    def _load_document(self, file_path: Path) -> List[Document]:
        """Load document using appropriate LangChain loader."""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                # Try different PDF loading strategies
                try:
                    loader = PyPDFLoader(str(file_path))
                    documents = loader.load()
                except Exception as pdf_error:
                    logger.warning(f"PyPDFLoader failed for {file_path.name}: {pdf_error}")
                    # Fallback: try reading as text
                    try:
                        import PyPDF2
                        documents = []
                        with open(file_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                            if pdf_reader.is_encrypted:
                                raise ValueError("PDF is encrypted/password protected")
                            
                            for page_num, page in enumerate(pdf_reader.pages):
                                text = page.extract_text()
                                if text.strip():  # Only add pages with content
                                    doc = Document(
                                        page_content=text,
                                        metadata={'page': page_num + 1}
                                    )
                                    documents.append(doc)
                    except Exception as fallback_error:
                        logger.error(f"All PDF loading methods failed: {fallback_error}")
                        raise ValueError(f"Cannot read PDF file: {pdf_error}")
                        
            elif file_extension == '.docx':
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()
            elif file_extension in ['.txt', '.md', '.csv']:
                # Try different encodings for text files
                encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
                documents = None
                
                for encoding in encodings:
                    try:
                        loader = TextLoader(str(file_path), encoding=encoding)
                        documents = loader.load()
                        logger.info(f"Successfully loaded {file_path.name} with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if documents is None:
                    raise ValueError(f"Could not decode text file with any supported encoding")
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Validate documents
            if not documents:
                raise ValueError("No content could be extracted from the file")
            
            # Add metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source': str(file_path),
                    'filename': file_path.name,
                    'file_type': file_extension,
                    'chunk_id': i
                })
                
                # Validate content
                if not doc.page_content or not doc.page_content.strip():
                    logger.warning(f"Empty content in chunk {i} of {file_path.name}")
            
            # Filter out empty documents
            valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
            
            if not valid_documents:
                raise ValueError("All extracted content was empty")
            
            logger.info(f"✅ Loaded {len(valid_documents)} valid document(s) from {file_path.name}")
            return valid_documents
        
        except Exception as e:
            logger.error(f"❌ Error loading document {file_path}: {e}")
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
                logger.info(f"ℹ Document {file_path.name} already processed")
                return True
            
            # Load document
            documents = self._load_document(file_path)
            if not documents:
                logger.warning(f"⚠ No content found in {file_path.name}")
                return False
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            if not chunks:
                logger.warning(f"⚠ No chunks created from {file_path.name}")
                return False
            
            logger.info(f"📄 Created {len(chunks)} chunks from {file_path.name}")
            
            # Add to vector store
            if self.vector_store is None:
                # Create new vector store
                logger.info("📊 Creating new vector store...")
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                # Add to existing vector store
                logger.info("📊 Adding to existing vector store...")
                self.vector_store.add_documents(chunks)
            
            # Save vector store
            self._save_vector_store()
            
            # Mark as processed
            self.processed_files.add(file_hash)
            
            logger.info(f"✅ Successfully processed {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing document {file_path}: {e}")
            return False
    
    def process_multiple_documents(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Process multiple documents and return results."""
        results = {
            'successful': [],
            'failed': [],
            'total_chunks': 0,
            'errors': []
        }
        
        for file_path in file_paths:
            try:
                if self.process_document(file_path):
                    results['successful'].append(str(file_path))
                else:
                    results['failed'].append(str(file_path))
                    results['errors'].append(f"Processing failed for {file_path.name}")
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")
                results['failed'].append(str(file_path))
                results['errors'].append(f"Error processing {file_path.name}: {str(e)}")
        
        # Count total chunks in vector store
        if self.vector_store:
            results['total_chunks'] = self.vector_store.index.ntotal
        
        logger.info(f"📊 Processing complete: {len(results['successful'])} successful, {len(results['failed'])} failed")
        return results
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents using similarity search."""
        if not self.vector_store:
            logger.warning("⚠ No vector store available for search - please upload documents first")
            return []
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"🔍 Found {len(results)} relevant chunks for query")
            return results
        
        except Exception as e:
            logger.error(f"❌ Error during search: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Search for relevant documents with similarity scores."""
        if not self.vector_store:
            logger.warning("⚠ No vector store available for search - please upload documents first")
            return []
        
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"🔍 Found {len(results)} relevant chunks with scores")
            return results
        
        except Exception as e:
            logger.error(f"❌ Error during search with scores: {e}")
            return []
    
    def _save_vector_store(self) -> None:
        """Save vector store to disk."""
        try:
            if self.vector_store:
                vector_store_path = config.get_paths()["data"] / config.get("vector_store.path", "vector_store")
                vector_store_path.parent.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(str(vector_store_path))
                logger.info("💾 Vector store saved successfully")
        except Exception as e:
            logger.error(f"❌ Error saving vector store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        stats = {
            'total_documents': len(self.processed_files),
            'vector_store_size': 0,
            'embedding_model': getattr(self.embeddings, 'model_name', 'Unknown'),
            'vector_store_type': config.get("vector_store.type", "faiss"),
            'has_vector_store': self.vector_store is not None
        }
        
        if self.vector_store:
            try:
                stats['vector_store_size'] = self.vector_store.index.ntotal
            except:
                stats['vector_store_size'] = 0
        
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
            
            logger.info("🧹 Vector store cleared successfully")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error clearing vector store: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if the document processor is ready to handle requests."""
        return self.embeddings is not None and self.text_splitter is not None
    
    def get_document_count(self) -> int:
        """Get the number of processed documents."""
        return len(self.processed_files)
    
    def has_documents(self) -> bool:
        """Check if any documents have been processed."""
        return self.vector_store is not None and len(self.processed_files) > 0