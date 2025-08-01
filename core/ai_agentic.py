"""AI Agent for question answering using Ollama integration."""

import logging
from typing import Dict, List, Optional, Tuple
import time

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

from core.document_processor import DocumentProcessor
from core.ollama_llm import create_ollama_llm
from config import config

logger = logging.getLogger(__name__)

class DocumentAIAgent:
    """Main AI Agent for document-based question answering."""
    
    def __init__(self):  # Fixed: double underscores
        """Initialize the AI agent with document processor and language model."""
        # Initialize attributes first to prevent AttributeError
        self.document_processor = None
        self.llm = None
        
        try:
            logger.info("🔄 Initializing DocumentAIAgent...")
            
            # Initialize document processor
            logger.info("📄 Creating document processor...")
            self.document_processor = DocumentProcessor()
            
            # Initialize LLM
            logger.info("🤖 Creating Ollama LLM...")
            self.llm = create_ollama_llm()
            
            # Setup prompts
            logger.info("📝 Setting up prompts...")
            self._setup_qa_prompts()
            
            logger.info("✅ DocumentAIAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing DocumentAIAgent: {e}")
            # Don't raise - let the agent work in degraded mode
            logger.warning("⚠ Agent running in degraded mode")
    
    def is_ready(self) -> bool:
        """Check if the AI agent is ready to handle requests."""
        try:
            return (
                self.document_processor is not None and
                self.document_processor.is_ready() and
                self.llm is not None
            )
        except Exception as e:
            logger.error(f"Error checking readiness: {e}")
            return False
    
    def _setup_qa_prompts(self):
        """Set up the QA prompt templates."""
        # Custom prompt template for better responses
        self.prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: Let me help you with that based on the documents. """,
            input_variables=["context", "question"]
        )
        
        # Summary prompt for document analysis
        self.summary_prompt = PromptTemplate(
            template="""Please provide a comprehensive summary of the following documents:
            
{context}

Summary:""",
            input_variables=["context"]
        )
    
    def process_documents(self, file_paths: List[str]) -> Dict:
        """Process multiple documents."""
        from pathlib import Path
        
        # Check if components are ready
        if not self.is_ready():
            logger.error("❌ DocumentAIAgent not ready - missing components")
            return {
                'successful': [],
                'failed': file_paths,
                'total_chunks': 0,
                'error': 'Agent not properly initialized'
            }
        
        try:
            path_objects = [Path(fp) for fp in file_paths]
            result = self.document_processor.process_multiple_documents(path_objects)
            logger.info(f"📊 Processed {len(result.get('successful', []))} documents successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Error processing documents: {e}")
            return {
                'successful': [],
                'failed': file_paths,
                'total_chunks': 0,
                'error': str(e)
            }
    
    def _calculate_confidence(self, retrieved_docs: List[Tuple[Document, float]]) -> float:
        """Calculate confidence score based on retrieval scores."""
        if not retrieved_docs:
            return 0.0
        
        # Use the best similarity score as base confidence
        best_score = min(score for _, score in retrieved_docs)  # Lower is better for FAISS
        
        # Convert distance to confidence (approximate)
        confidence = max(0.0, 1.0 - (best_score / 2.0))
        return min(confidence, 1.0)
    
    def _format_sources(self, retrieved_docs: List[Tuple[Document, float]]) -> List[Dict]:
        """Format source information from retrieved documents."""
        sources = []
        seen_sources = set()
        
        for doc, score in retrieved_docs:
            source_info = {
                'filename': doc.metadata.get('filename', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'score': round(1.0 - (score / 2.0), 3),  # Convert to confidence-like score
                'snippet': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            
            # Avoid duplicate sources
            source_key = f"{source_info['filename']}_{source_info['page']}"
            if source_key not in seen_sources:
                sources.append(source_info)
                seen_sources.add(source_key)
        
        return sources
    
    def ask_question(self, question: str) -> Dict:
        """Answer a question based on processed documents."""
        # Check if components are ready
        if not self.is_ready():
            return {
                'answer': "The AI agent is not properly initialized. Please check the system status.",
                'confidence': 0.0,
                'sources': [],
                'error': "Agent not ready"
            }
        
        try:
            # Validate input
            if not question or not question.strip():
                return {
                    'answer': "Please provide a valid question.",
                    'confidence': 0.0,
                    'sources': [],
                    'error': "Empty question"
                }
            
            # Search for relevant documents
            retrieved_docs = self.document_processor.search_with_scores(
                question, 
                k=config.get("document_processing.top_k_retrieval", 5)
            )
            
            if not retrieved_docs:
                return {
                    'answer': "I don't have any relevant documents to answer your question. Please upload some documents first.",
                    'confidence': 0.0,
                    'sources': [],
                    'error': "No documents found"
                }
            
            # Calculate confidence
            confidence = self._calculate_confidence(retrieved_docs)
            
            # Check confidence threshold
            confidence_threshold = config.get("document_processing.confidence_threshold", 0.7)
            if confidence < confidence_threshold:
                return {
                    'answer': "I don't have enough relevant information to provide a confident answer to your question.",
                    'confidence': confidence,
                    'sources': self._format_sources(retrieved_docs),
                    'error': "Low confidence"
                }
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc, _ in retrieved_docs])
            
            # Generate answer using the language model
            prompt = self.prompt.format(context=context, question=question)
            answer = self.llm._call(prompt)
            
            # Clean up the answer
            answer = answer.strip()
            if not answer or answer.lower() in ['', 'i don\'t know', 'i don\'t know.']:
                answer = "Based on the available documents, I cannot provide a specific answer to your question."
            
            return {
                'answer': answer,
                'confidence': confidence,
                'sources': self._format_sources(retrieved_docs),
                'error': None
            }
        
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}")
            return {
                'answer': "I encountered an error while processing your question. Please try again.",
                'confidence': 0.0,
                'sources': [],
                'error': str(e)
            }
    
    def ask_with_context(self, question: str, custom_config: Dict = None) -> Dict:
        """Ask a question with optional custom configuration."""
        try:
            # Temporarily update LLM config if provided
            if custom_config and hasattr(self.llm, 'update_config'):
                original_config = {
                    'temperature': getattr(self.llm, 'temperature', 0.7),
                    'max_tokens': getattr(self.llm, 'max_tokens', 1024)
                }
                
                # Apply custom config
                self.llm.update_config(custom_config)
                
                try:
                    # Get answer with custom config
                    result = self.ask_question(question)
                finally:
                    # Restore original config
                    self.llm.update_config(original_config)
                
                return result
            else:
                # Use default configuration
                return self.ask_question(question)
        
        except Exception as e:
            logger.error(f"Error in ask_with_context: {e}")
            return {
                'answer': "I encountered an error while processing your question with custom configuration.",
                'confidence': 0.0,
                'sources': [],
                'error': str(e)
            }
    
    def summarize_documents(self, query: str = None, max_docs: int = 10) -> Dict:
        """Generate a summary of uploaded documents."""
        try:
            if not self.document_processor.has_documents():
                return {
                    'summary': "No documents have been uploaded yet. Please upload some documents first.",
                    'confidence': 0.0,
                    'sources': [],
                    'error': "No documents available"
                }
            
            # Get relevant documents
            if query:
                docs = self.document_processor.search_documents(query, k=max_docs)
            else:
                # Get a sample of documents from the vector store
                docs = self.document_processor.search_documents("", k=max_docs)
            
            if not docs:
                return {
                    'summary': "No documents found to summarize.",
                    'confidence': 0.0,
                    'sources': []
                }
            
            # Combine document content
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate summary
            summary_prompt = self.summary_prompt.format(context=context)
            summary = self.llm._call(summary_prompt)
            
            # Format sources
            sources = []
            for doc in docs:
                sources.append({
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'snippet': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })
            
            return {
                'summary': summary,
                'confidence': 0.8,  # Summaries generally have good confidence
                'sources': sources,
                'document_count': len(docs)
            }
        
        except Exception as e:
            logger.error(f"Error in summarize_documents: {e}")
            return {
                'summary': "I encountered an error while generating the summary.",
                'confidence': 0.0,
                'sources': [],
                'error': str(e)
            }
    
    def get_similar_questions(self, question: str, limit: int = 3) -> List[str]:
        """Get similar questions based on document content."""
        try:
            retrieved_docs = self.document_processor.search_documents(question, k=limit)
            
            similar_questions = []
            for doc in retrieved_docs:
                # Extract potential questions from document content
                content = doc.page_content
                sentences = content.split('.')
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if '?' in sentence and len(sentence) > 10:
                        similar_questions.append(sentence)
                        if len(similar_questions) >= limit:
                            break
                
                if len(similar_questions) >= limit:
                    break
            
            return similar_questions[:limit]
        
        except Exception as e:
            logger.error(f"Error getting similar questions: {e}")
            return []
    
    def analyze_document(self, file_path: str) -> Dict:
        """Analyze a specific document."""
        try:
            from pathlib import Path
            
            # Process the document if not already processed
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {'error': f'File not found: {file_path}'}
            
            # Get document content
            docs = self.document_processor._load_document(file_path_obj)
            if not docs:
                return {'error': 'Could not load document'}
            
            content = "\n".join([doc.page_content for doc in docs])
            
            # Generate analysis
            analysis_prompt = f"""
            Analyze the following document and provide insights:
            
            Document: {file_path_obj.name}
            Content:
            {content[:2000]}...
            
            Please provide:
            1. Document type and purpose
            2. Main topics covered
            3. Key insights
            4. Structure analysis
            """
            
            analysis = self.llm._call(analysis_prompt)
            
            return {
                'filename': file_path_obj.name,
                'analysis': analysis,
                'word_count': len(content.split()),
                'char_count': len(content)
            }
        
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {'error': str(e)}
    
    def get_stats(self) -> Dict:
        """Get statistics about the AI agent."""
        try:
            if not self.document_processor:
                return {
                    'error': 'Document processor not initialized',
                    'agent_ready': False,
                    'has_documents': False
                }
            
            doc_stats = self.document_processor.get_stats()
            
            return {
                **doc_stats,
                'generation_model': getattr(self.llm, 'model', config.get('ollama.model', 'gemma:latest')),
                'top_k_retrieval': config.get("document_processing.top_k_retrieval", 5),
                'confidence_threshold': config.get("document_processing.confidence_threshold", 0.7),
                'agent_ready': self.is_ready(),
                'has_documents': self.document_processor.has_documents()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'error': str(e),
                'agent_ready': False,
                'has_documents': False
            }
    
    def clear_knowledge_base(self) -> bool:
        """Clear all processed documents."""
        try:
            if not self.document_processor:
                return False
            return self.document_processor.clear_vector_store()
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False
    
    def reload_config(self) -> bool:
        """Reload configuration and reinitialize components."""
        try:
            # Reinitialize LLM with new config
            self.llm = create_ollama_llm()
            logger.info("✅ Configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error reloading configuration: {e}")
            return False
    
    def health_check(self) -> Dict:
        """Perform health check on all components."""
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': time.time()
        }
        
        try:
            # Check document processor
            if self.document_processor:
                health['components']['document_processor'] = {
                    'status': 'healthy' if self.document_processor.is_ready() else 'unhealthy',
                    'has_documents': self.document_processor.has_documents(),
                    'document_count': self.document_processor.get_document_count()
                }
            else:
                health['components']['document_processor'] = {
                    'status': 'unhealthy',
                    'error': 'Document processor not initialized'
                }
            
            # Check LLM
            try:
                if self.llm:
                    test_response = self.llm._call("Hello")
                    health['components']['llm'] = {
                        'status': 'healthy' if test_response else 'unhealthy',
                        'model': getattr(self.llm, 'model', 'Unknown')
                    }
                else:
                    health['components']['llm'] = {
                        'status': 'unhealthy',
                        'error': 'LLM not initialized'
                    }
            except Exception as e:
                health['components']['llm'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Overall status
            unhealthy_components = [
                comp for comp in health['components'].values() 
                if comp.get('status') != 'healthy'
            ]
            
            if unhealthy_components:
                health['status'] = 'degraded' if len(unhealthy_components) < len(health['components']) else 'unhealthy'
        
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health