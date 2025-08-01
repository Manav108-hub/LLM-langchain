"""Ollama LLM integration for LangChain."""

import json
import logging
import requests
from typing import Any, Dict, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings.base import Embeddings

from config import config

logger = logging.getLogger(__name__)

class OllamaLLM(LLM):
    """Custom LangChain LLM for Ollama integration."""

    def __init__(self):
        super().__init__()
        self.ollama_config = config.get_ollama_config()
        self.base_url = self.ollama_config.get("base_url", "http://localhost:11434")
        self.model = self.ollama_config.get("model", "gemma:8b")
        self.timeout = self.ollama_config.get("timeout", 300)
        self.temperature = self.ollama_config.get("temperature", 0.7)
        self.max_tokens = self.ollama_config.get("max_tokens", 1024)
        self.stream = self.ollama_config.get("stream", False)

        # Validate connection on initialization
        if not self._validate_connection():
            logger.warning("Ollama connection validation failed")

    def _validate_connection(self) -> bool:
        """Validate Ollama connection."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Ollama API to generate response."""
        try:
            # Prepare the payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": self.stream,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }

            # Add stop tokens if provided
            if stop:
                payload["options"]["stop"] = stop

            # Make the request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "I apologize, but I encountered an error while generating a response."

            # Parse response
            if self.stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                full_response += chunk['response']
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                return full_response.strip()
            else:
                # Handle non-streaming response
                result = response.json()
                return result.get('response', '').strip()

        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout")
            return "I'm sorry, but the request timed out. Please try again."

        except requests.exceptions.ConnectionError:
            logger.error("Ollama connection error")
            return "I'm unable to connect to the Ollama service. Please ensure Ollama is running."

        except Exception as e:
            logger.error(f"Unexpected error in Ollama call: {e}")
            return "I encountered an unexpected error. Please try again."

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update Ollama configuration from payload."""
        self.ollama_config.update(new_config)

        # Update instance variables
        self.base_url = self.ollama_config.get("base_url", self.base_url)
        self.model = self.ollama_config.get("model", self.model)
        self.timeout = self.ollama_config.get("timeout", self.timeout)
        self.temperature = self.ollama_config.get("temperature", self.temperature)
        self.max_tokens = self.ollama_config.get("max_tokens", self.max_tokens)
        self.stream = self.ollama_config.get("stream", self.stream)

        logger.info(f"Ollama configuration updated: {self.model}")

class OllamaEmbeddings(Embeddings):
    """Ollama embeddings for LangChain."""

    def __init__(self):
        self.ollama_config = config.get_ollama_config()
        self.base_url = self.ollama_config.get("base_url", "http://localhost:11434")
        self.model = self.ollama_config.get("embedding_model", "nomic-embed-text")
        self.timeout = self.ollama_config.get("timeout", 300)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []

        for text in texts:
            try:
                payload = {
                    "model": self.model,
                    "prompt": text
                }

                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])
                    embeddings.append(embedding)
                else:
                    logger.error(f"Embedding error for text: {response.status_code}")
                    # Return zero vector as fallback
                    embeddings.append([0.0] * 768)  # Default dimension

            except Exception as e:
                logger.error(f"Error embedding text: {e}")
                embeddings.append([0.0] * 768)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]

class ModelManager:
    """Manages Ollama models and configurations."""

    def __init__(self):
        self.ollama_config = config.get_ollama_config()
        self.base_url = self.ollama_config.get("base_url", "http://localhost:11434")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull/download a model."""
        try:
            payload = {"name": model_name}
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=600  # 10 minutes for model download
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        try:
            payload = {"name": model_name}
            response = requests.delete(
                f"{self.base_url}/api/delete",
                json=payload,
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False

    def check_model_exists(self, model_name: str) -> bool:
        """Check if model exists locally."""
        models = self.list_models()
        return any(model["name"] == model_name for model in models)

    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure model is available, pull if necessary."""
        if self.check_model_exists(model_name):
            logger.info(f"Model {model_name} is already available")
            return True

        logger.info(f"Pulling model {model_name}...")
        return self.pull_model(model_name)

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        models = self.list_models()
        for model in models:
            if model["name"] == model_name:
                return model
        return None

    def update_model_config(self, payload: Dict[str, Any]) -> None:
        """Update model configuration from payload."""
        # Update global config
        config.update_from_payload({"ollama": payload})

        # Update local config
        self.ollama_config.update(payload)

        logger.info("Model configuration updated")

# Global model manager instance
model_manager = ModelManager()

def create_ollama_llm() -> OllamaLLM:
    """Create and return Ollama LLM instance."""
    return OllamaLLM()

def create_ollama_embeddings() -> OllamaEmbeddings:
    """Create and return Ollama embeddings instance."""
    return OllamaEmbeddings()

def validate_ollama_setup() -> Dict[str, Any]:
    """Validate complete Ollama setup."""
    results = {
        "status": "success",
        "ollama_running": False,
        "model_available": False,
        "embedding_model_available": False,
        "errors": []
    }

    try:
        # Check if Ollama is running
        response = requests.get(f"{config.get('ollama.base_url')}/api/tags", timeout=5)
        if response.status_code == 200:
            results["ollama_running"] = True

            # Check models
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            # Check main model
            main_model = config.get("ollama.model", "gemma:8b")
            if main_model in model_names:
                results["model_available"] = True
            else:
                results["errors"].append(f"Model {main_model} not found")

            # Check embedding model
            embed_model = config.get("ollama.embedding_model", "nomic-embed-text")
            if embed_model in model_names:
                results["embedding_model_available"] = True
            else:
                results["errors"].append(f"Embedding model {embed_model} not found")
        else:
            results["errors"].append("Ollama service not responding")

    except Exception as e:
        results["status"] = "error"
        results["errors"].append(f"Connection error: {str(e)}")

    if results["errors"]:
        results["status"] = "error" if not results["ollama_running"] else "warning"

    return results
