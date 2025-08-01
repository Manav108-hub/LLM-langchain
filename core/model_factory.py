"""Model factory for dynamic model loading based on configuration."""

import logging
import os
from typing import Optional, Union
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

import config

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating AI models based on configuration."""

    @staticmethod
    def create_embeddings() -> Embeddings:
        """Create embeddings model based on configuration."""
        try:
            # Set Hugging Face cache directory if specified
            if config.HUGGINGFACE_CACHE_DIR:
                os.environ['TRANSFORMERS_CACHE'] = config.HUGGINGFACE_CACHE_DIR

            # Set Hugging Face token if provided
            if config.HUGGINGFACE_TOKEN:
                os.environ['HUGGINGFACE_HUB_TOKEN'] = config.HUGGINGFACE_TOKEN

            # Check if using OpenAI embeddings
            if config.USE_OPENAI and config.OPENAI_API_KEY:
                from langchain_openai import OpenAIEmbeddings
                logger.info(f"Using OpenAI embeddings: {config.OPENAI_EMBEDDING_MODEL}")
                return OpenAIEmbeddings(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    openai_api_key=config.OPENAI_API_KEY
                )

            # Check if using Azure OpenAI embeddings
            if config.USE_AZURE_OPENAI and config.AZURE_OPENAI_API_KEY:
                from langchain_openai import AzureOpenAIEmbeddings
                logger.info("Using Azure OpenAI embeddings")
                return AzureOpenAIEmbeddings(
                    azure_deployment=config.AZURE_DEPLOYMENT_NAME,
                    openai_api_version=config.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                    api_key=config.AZURE_OPENAI_API_KEY
                )

            # Default to Hugging Face embeddings
            logger.info(f"Using Hugging Face embeddings: {config.EMBEDDING_MODEL}")
            return HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            # Fallback to a lightweight model
            logger.info("Falling back to sentence-transformers/all-MiniLM-L6-v2")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

    @staticmethod
    def create_llm() -> LLM:
        """Create language model based on configuration."""
        try:
            # Check if using OpenAI
            if config.USE_OPENAI and config.OPENAI_API_KEY:
                from langchain_openai import ChatOpenAI
                logger.info(f"Using OpenAI model: {config.OPENAI_MODEL}")
                return ChatOpenAI(
                    model=config.OPENAI_MODEL,
                    openai_api_key=config.OPENAI_API_KEY,
                    temperature=0.7,
                    max_tokens=500
                )

            # Check if using Anthropic
            if config.USE_ANTHROPIC and config.ANTHROPIC_API_KEY:
                from langchain_anthropic import ChatAnthropic
                logger.info(f"Using Anthropic model: {config.ANTHROPIC_MODEL}")
                return ChatAnthropic(
                    model=config.ANTHROPIC_MODEL,
                    anthropic_api_key=config.ANTHROPIC_API_KEY,
                    temperature=0.7,
                    max_tokens=500
                )

            # Check if using Google
            if config.USE_GOOGLE and config.GOOGLE_API_KEY:
                from langchain_google_genai import ChatGoogleGenerativeAI
                logger.info(f"Using Google model: {config.GOOGLE_MODEL}")
                return ChatGoogleGenerativeAI(
                    model=config.GOOGLE_MODEL,
                    google_api_key=config.GOOGLE_API_KEY,
                    temperature=0.7,
                    max_output_tokens=500
                )

            # Check if using Azure OpenAI
            if config.USE_AZURE_OPENAI and config.AZURE_OPENAI_API_KEY:
                from langchain_openai import AzureChatOpenAI
                logger.info("Using Azure OpenAI model")
                return AzureChatOpenAI(
                    azure_deployment=config.AZURE_DEPLOYMENT_NAME,
                    openai_api_version=config.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                    api_key=config.AZURE_OPENAI_API_KEY,
                    temperature=0.7,
                    max_tokens=500
                )

            # Default to local Hugging Face model
            from core.ai_agent import DialoGPTLLM
            logger.info(f"Using local Hugging Face model: {config.GENERATION_MODEL}")
            return DialoGPTLLM()

        except Exception as e:
            logger.error(f"Error creating LLM: {e}")
            # Fallback to DialoGPT
            from core.ai_agent import DialoGPTLLM
            logger.info("Falling back to DialoGPT-large")
            return DialoGPTLLM()

    @staticmethod
    def create_vector_store(embeddings: Embeddings, documents=None):
        """Create vector store based on configuration."""
        try:
            if config.VECTOR_STORE_TYPE.lower() == 'pinecone':
                return ModelFactory._create_pinecone_store(embeddings, documents)
            elif config.VECTOR_STORE_TYPE.lower() == 'weaviate':
                return ModelFactory._create_weaviate_store(embeddings, documents)
            elif config.VECTOR_STORE_TYPE.lower() == 'chroma':
                return ModelFactory._create_chroma_store(embeddings, documents)
            else:
                # Default to FAISS
                return ModelFactory._create_faiss_store(embeddings, documents)

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            # Fallback to FAISS
            return ModelFactory._create_faiss_store(embeddings, documents)

    @staticmethod
    def _create_faiss_store(embeddings: Embeddings, documents=None):
        """Create FAISS vector store."""
        from langchain_community.vectorstores import FAISS

        if documents:
            return FAISS.from_documents(documents, embeddings)
        elif config.VECTOR_STORE_PATH.exists():
            return FAISS.load_local(
                str(config.VECTOR_STORE_PATH),
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            return None

    @staticmethod
    def _create_pinecone_store(embeddings: Embeddings, documents=None):
        """Create Pinecone vector store."""
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required for Pinecone vector store")

        import pinecone
        from langchain_community.vectorstores import Pinecone

        pinecone.init(
            api_key=config.PINECONE_API_KEY,
            environment=config.PINECONE_ENVIRONMENT
        )

        if documents:
            return Pinecone.from_documents(
                documents,
                embeddings,
                index_name=config.PINECONE_INDEX_NAME
            )
        else:
            return Pinecone.from_existing_index(
                config.PINECONE_INDEX_NAME,
                embeddings
            )

    @staticmethod
    def _create_weaviate_store(embeddings: Embeddings, documents=None):
        """Create Weaviate vector store."""
        import weaviate
        from langchain_community.vectorstores import Weaviate

        auth_config = None
        if config.WEAVIATE_API_KEY:
            auth_config = weaviate.AuthApiKey(api_key=config.WEAVIATE_API_KEY)

        client = weaviate.Client(
            url=config.WEAVIATE_URL,
            auth_client_secret=auth_config
        )

        if documents:
            return Weaviate.from_documents(
                documents,
                embeddings,
                client=client,
                index_name=config.VECTOR_STORE_INDEX_NAME
            )
        else:
            return Weaviate(
                client=client,
                index_name=config.VECTOR_STORE_INDEX_NAME,
                text_key="text"
            )

    @staticmethod
    def _create_chroma_store(embeddings: Embeddings, documents=None):
        """Create Chroma vector store."""
        from langchain_community.vectorstores import Chroma

        persist_directory = str(config.VECTOR_STORE_PATH)

        if documents:
            return Chroma.from_documents(
                documents,
                embeddings,
                persist_directory=persist_directory,
                collection_name=config.VECTOR_STORE_INDEX_NAME
            )
        else:
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=config.VECTOR_STORE_INDEX_NAME
            )

class CacheManager:
    """Manages caching for the application."""

    def __init__(self):
        self.redis_client = None
        if config.USE_REDIS:
            self._setup_redis()

    def _setup_redis(self):
        """Setup Redis connection."""
        try:
            import redis
            self.redis_client = redis.from_url(
                config.REDIS_URL,
                password=config.REDIS_PASSWORD,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if not self.redis_client:
            return None

        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.redis_client:
            return False

        try:
            return self.redis_client.set(
                key,
                value,
                ex=ttl or config.CACHE_TTL
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.redis_client:
            return False

        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache."""
        if not self.redis_client:
            return False

        try:
            return self.redis_client.flushdb()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

# Global instances
cache_manager = CacheManager()
