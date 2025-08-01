"""LangChain service layer for advanced document processing and retrieval."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from core.document_processor import DocumentProcessor
from core.ai_agent import DialoGPTLLM
from utils.file_handlers import FileValidator, DocumentMetadata
from config import EMBEDDING_MODEL, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

class AdvancedRetrievalService:
    """Advanced retrieval service using LangChain components."""

    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor
        self.llm = DialoGPTLLM()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Keep last 5 exchanges
        )
        self._setup_chains()

    def _setup_chains(self):
        """Set up various LangChain chains for different use cases."""

        # Standard QA Chain
        self.qa_prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}

            Helpful Answer:""",
            input_variables=["context", "question"]
        )

        # Conversational QA Chain
        self.conversational_prompt = PromptTemplate(
            template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

            Chat History:
            {chat_history}

            Follow Up Input: {question}

            Standalone question:""",
            input_variables=["chat_history", "question"]
        )

        # Summary Chain
        self.summary_prompt = PromptTemplate(
            template="""Please provide a comprehensive summary of the following documents:

            {context}

            Summary:""",
            input_variables=["context"]
        )

    def get_qa_chain(self) -> Optional[RetrievalQA]:
        """Get a standard QA chain."""
        if not self.document_processor.vector_store:
            return None

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.document_processor.vector_store.as_retriever(
                search_kwargs={"k": TOP_K_RETRIEVAL}
            ),
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )

    def get_conversational_chain(self) -> Optional[ConversationalRetrievalChain]:
        """Get a conversational QA chain with memory."""
        if not self.document_processor.vector_store:
            return None

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.document_processor.vector_store.as_retriever(
                search_kwargs={"k": TOP_K_RETRIEVAL}
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )

    def ask_with_context(self, question: str, use_conversation: bool = False) -> Dict[str, Any]:
        """Ask a question with full context and chain selection."""
        try:
            if use_conversation:
                chain = self.get_conversational_chain()
                if not chain:
                    return self._no_documents_response()

                result = chain({"question": question})
            else:
                chain = self.get_qa_chain()
                if not chain:
                    return self._no_documents_response()

                result = chain({"query": question})

            # Process the result
            return self._process_chain_result(result, question)

        except Exception as e:
            logger.error(f"Error in ask_with_context: {e}")
            return {
                'answer': "I encountered an error while processing your question.",
                'confidence': 0.0,
                'sources': [],
                'error': str(e),
                'chain_type': 'conversational' if use_conversation else 'standard'
            }

    def _process_chain_result(self, result: Dict, question: str) -> Dict[str, Any]:
        """Process the result from a LangChain chain."""
        answer = result.get('result') or result.get('answer', '')
        source_docs = result.get('source_documents', [])

        # Calculate confidence based on source relevance
        confidence = self._calculate_answer_confidence(answer, source_docs, question)

        # Format sources
        sources = self._format_chain_sources(source_docs)

        return {
            'answer': answer,
            'confidence': confidence,
            'sources': sources,
            'error': None,
            'source_count': len(source_docs)
        }

    def _calculate_answer_confidence(self, answer: str, source_docs: List[Document], question: str) -> float:
        """Calculate confidence score for the answer."""
        if not answer or not source_docs:
            return 0.0

        # Basic confidence calculation
        confidence = 0.5  # Base confidence

        # Increase confidence based on number of sources
        if len(source_docs) >= 3:
            confidence += 0.2
        elif len(source_docs) >= 2:
            confidence += 0.1

        # Increase confidence if answer is not generic
        generic_responses = [
            "i don't know", "i don't have", "not sure", "cannot determine",
            "unable to", "no information", "not available"
        ]

        if not any(generic in answer.lower() for generic in generic_responses):
            confidence += 0.2

        # Decrease confidence for very short answers
        if len(answer.split()) < 5:
            confidence -= 0.1

        return min(max(confidence, 0.0), 1.0)

    def _format_chain_sources(self, source_docs: List[Document]) -> List[Dict]:
        """Format source documents for API response."""
        sources = []
        for i, doc in enumerate(source_docs):
            source = {
                'id': i,
                'filename': doc.metadata.get('filename', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'snippet': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'metadata': doc.metadata
            }
            sources.append(source)

        return sources

    def _no_documents_response(self) -> Dict[str, Any]:
        """Standard response when no documents are available."""
        return {
            'answer': "I don't have any documents to search through. Please upload some documents first.",
            'confidence': 0.0,
            'sources': [],
            'error': "No vector store available"
        }

    def summarize_documents(self, query: str = None, max_docs: int = 10) -> Dict[str, Any]:
        """Generate a summary of uploaded documents."""
        try:
            if not self.document_processor.vector_store:
                return self._no_documents_response()

            # Get relevant documents
            if query:
                docs = self.document_processor.search_documents(query, k=max_docs)
            else:
                # Get a sample of documents from the vector store
                docs = self.document_processor.vector_store.similarity_search("", k=max_docs)

            if not docs:
                return {
                    'summary': "No documents found to summarize.",
                    'confidence': 0.0,
                    'sources': []
                }

            # Combine document content
            context = "\n\n".join([doc.page_content for doc in docs])

            # Generate summary
            summary_chain = self.qa_prompt.format(context=context, question="Please provide a comprehensive summary of these documents.")
            summary = self.llm._call(summary_chain)

            return {
                'summary': summary,
                'confidence': 0.8,  # Summaries generally have good confidence
                'sources': self._format_chain_sources(docs),
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

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        try:
            history = []
            if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
                for msg in self.memory.chat_memory.messages:
                    history.append({
                        'type': msg.type,
                        'content': msg.content,
                        'timestamp': getattr(msg, 'timestamp', None)
                    })
            return history
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    def clear_conversation_history(self) -> bool:
        """Clear the conversation history."""
        try:
            self.memory.clear()
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            return False

class DocumentAnalysisService:
    """Service for advanced document analysis using LangChain."""

    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor
        self.llm = DialoGPTLLM()

    def analyze_document_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze the structure and content of a document."""
        try:
            # Validate file first
            validation = FileValidator.validate_file(file_path)
            if not validation['valid']:
                return {'error': validation['error']}

            # Load document
            documents = self.document_processor._load_document(file_path)
            if not documents:
                return {'error': 'Could not load document'}

            # Analyze content
            total_content = "\n".join([doc.page_content for doc in documents])
            metadata = DocumentMetadata.extract_metadata(file_path, total_content)

            # Generate analysis using LLM
            analysis_prompt = f"""
            Analyze the following document and provide insights about its structure and content:

            Document: {file_path.name}
            Content length: {len(total_content)} characters

            Content preview:
            {total_content[:1000]}...

            Please provide:
            1. Document type and purpose
            2. Main topics covered
            3. Structure analysis
            4. Key insights
            """

            analysis = self.llm._call(analysis_prompt)

            return {
                'filename': file_path.name,
                'metadata': metadata,
                'analysis': analysis,
                'document_count': len(documents),
                'total_length': len(total_content)
            }

        except Exception as e:
            logger.error(f"Error analyzing document {file_path}: {e}")
            return {'error': str(e)}

    def find_similar_documents(self, query_doc: Path, top_k: int = 5) -> List[Dict]:
        """Find documents similar to the query document."""
        try:
            if not self.document_processor.vector_store:
                return []

            # Load and process query document
            query_docs = self.document_processor._load_document(query_doc)
            if not query_docs:
                return []

            query_content = "\n".join([doc.page_content for doc in query_docs])

            # Search for similar documents
            similar_docs = self.document_processor.search_with_scores(query_content, k=top_k)

            results = []
            for doc, score in similar_docs:
                results.append({
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'similarity_score': 1.0 - (score / 2.0),  # Convert to similarity
                    'snippet': doc.page_content[:200] + "...",
                    'metadata': doc.metadata
                })

            return results

        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []

    def extract_key_information(self, topic: str, max_results: int = 5) -> Dict[str, Any]:
        """Extract key information about a specific topic from all documents."""
        try:
            if not self.document_processor.vector_store:
                return {'error': 'No documents available'}

            # Search for relevant content
            relevant_docs = self.document_processor.search_documents(topic, k=max_results)

            if not relevant_docs:
                return {'message': f'No information found about "{topic}"'}

            # Extract key information using LLM
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            extraction_prompt = f"""
            Extract and summarize key information about "{topic}" from the following documents:

            {context}

            Please provide:
            1. Key facts and figures
            2. Important concepts
            3. Relevant details
            4. Any conclusions or insights

            Focus on information specifically related to "{topic}".
            """

            extracted_info = self.llm._call(extraction_prompt)

            return {
                'topic': topic,
                'key_information': extracted_info,
                'sources': [doc.metadata.get('filename', 'Unknown') for doc in relevant_docs],
                'source_count': len(relevant_docs)
            }

        except Exception as e:
            logger.error(f"Error extracting key information for topic '{topic}': {e}")
            return {'error': str(e)}
