"""AI Agent for question answering using LangChain."""

import logging
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

from core.document_processor import DocumentProcessor
from core.ollama_llm import create_ollama_llm
from config import config

logger = logging.getLogger(__name__)

class DialoGPTLLM(LLM):
    """Custom LangChain LLM wrapper for DialoGPT."""

    def __init__(self):
        super().__init__()
        self.model_name = GENERATION_MODEL
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the DialoGPT model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "dialogpt"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Generate response using DialoGPT."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=1000, truncation=True)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the new part (remove the input prompt)
            if prompt in response:
                response = response.replace(prompt, "").strip()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response."

class DocumentAIAgent:
    """Main AI Agent for document-based question answering."""

    def __init__(self):
        """Initialize the AI agent with document processor and language model."""
        self.document_processor = DocumentProcessor()
        self.llm = create_ollama_llm()
        self.qa_chain = None
        self._setup_qa_chain()

    def _setup_qa_chain(self):
        """Set up the QA chain with custom prompt template."""
        # Custom prompt template for better responses
        prompt_template = """
        Use the following pieces of context to answer the question. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Answer: Let me help you with that based on the documents. """

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

    def process_documents(self, file_paths: List[str]) -> Dict:
        """Process multiple documents."""
        from pathlib import Path

        path_objects = [Path(fp) for fp in file_paths]
        return self.document_processor.process_multiple_documents(path_objects)

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
            logger.error(f"Error answering question: {e}")
            return {
                'answer': "I encountered an error while processing your question. Please try again.",
                'confidence': 0.0,
                'sources': [],
                'error': str(e)
            }

    def get_similar_questions(self, question: str, limit: int = 3) -> List[str]:
        """Get similar questions based on document content (placeholder implementation)."""
        # This is a simplified implementation
        # In a production system, you might want to maintain a question history
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

    def get_stats(self) -> Dict:
        """Get statistics about the AI agent."""
        doc_stats = self.document_processor.get_stats()
        return {
            **doc_stats,
            'generation_model': getattr(self.llm, 'model_name', config.GENERATION_MODEL),
            'top_k_retrieval': TOP_K_RETRIEVAL,
            'confidence_threshold': CONFIDENCE_THRESHOLD
        }

    def clear_knowledge_base(self) -> bool:
        """Clear all processed documents."""
        return self.document_processor.clear_vector_store()
