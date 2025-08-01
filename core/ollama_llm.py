"""Ollama LLM integration for LangChain."""

import json
import logging
import requests
from typing import Any, Dict, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings.base import Embeddings
from pydantic import Field

from config import config

logger = logging.getLogger(__name__)

class OllamaLLM(LLM):
    """Custom LangChain LLM for Ollama integration."""
    
    # Define Pydantic fields with defaults
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="gemma:latest")
    timeout: int = Field(default=300)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1024)
    streaming: bool = Field(default=False, alias="stream")  # Use alias to avoid conflict
    
    def __init__(self, **kwargs):  # Fixed: double underscores
        # Get config values
        ollama_config = config.get_ollama_config()
        
        # Set defaults from config
        defaults = {
            "base_url": ollama_config.get("base_url", "http://localhost:11434"),
            "model": ollama_config.get("model", "gemma:latest"),
            "timeout": ollama_config.get("timeout", 300),
            "temperature": ollama_config.get("temperature", 0.7),
            "max_tokens": ollama_config.get("max_tokens", 1024),
            "streaming": ollama_config.get("stream", False)
        }
        
        # Merge with any provided kwargs
        final_kwargs = {**defaults, **kwargs}
        
        # Initialize parent class
        super().__init__(**final_kwargs)
        
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
                "stream": self.streaming,
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
            if self.streaming:
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
        # Update instance variables
        if "base_url" in new_config:
            self.base_url = new_config["base_url"]
        if "model" in new_config:
            self.model = new_config["model"]
        if "timeout" in new_config:
            self.timeout = new_config["timeout"]
        if "temperature" in new_config:
            self.temperature = new_config["temperature"]
        if "max_tokens" in new_config:
            self.max_tokens = new_config["max_tokens"]
        if "stream" in new_config:
            self.streaming = new_config["stream"]
        
        logger.info(f"Ollama configuration updated: {self.model}")

class OllamaEmbeddings(Embeddings):
    """Ollama embeddings for LangChain."""
    
    def __init__(self):  # Fixed: double underscores
        super().__init__()
        ollama_config = config.get_ollama_config()
        self.base_url = ollama_config.get("base_url", "http://localhost:11434")
        self.model = ollama_config.get("embedding_model", "nomic-embed-text")
        self.timeout = ollama_config.get("timeout", 300)
    
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
    
    def __init__(self):  # Fixed: double underscores
        ollama_config = config.get_ollama_config()
        self.base_url = ollama_config.get("base_url", "http://localhost:11434")
    
    @staticmethod
    def list_models() -> List[Dict[str, Any]]:
        """List available Ollama models."""
        try:
            ollama_config = config.get_ollama_config()
            base_url = ollama_config.get("base_url", "http://localhost:11434")
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    @staticmethod
    def pull_model(model_name: str) -> bool:
        """Pull/download a model."""
        try:
            ollama_config = config.get_ollama_config()
            base_url = ollama_config.get("base_url", "http://localhost:11434")
            payload = {"name": model_name}
            response = requests.post(
                f"{base_url}/api/pull",
                json=payload,
                timeout=600  # 10 minutes for model download
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    @staticmethod
    def delete_model(model_name: str) -> bool:
        """Delete a model."""
        try:
            ollama_config = config.get_ollama_config()
            base_url = ollama_config.get("base_url", "http://localhost:11434")
            payload = {"name": model_name}
            response = requests.delete(
                f"{base_url}/api/delete",
                json=payload,
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    @staticmethod
    def check_model_exists(model_name: str) -> bool:
        """Check if model exists locally."""
        models = ModelManager.list_models()
        return any(model["name"] == model_name for model in models)
    
    @staticmethod
    def ensure_model_available(model_name: str) -> bool:
        """Ensure model is available, pull if necessary."""
        if ModelManager.check_model_exists(model_name):
            logger.info(f"Model {model_name} is already available")
            return True
        
        logger.info(f"Pulling model {model_name}...")
        return ModelManager.pull_model(model_name)
    
    @staticmethod
    def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        models = ModelManager.list_models()
        for model in models:
            if model["name"] == model_name:
                return model
        return None
    
    @staticmethod
    def update_model_config(payload: Dict[str, Any]) -> None:
        """Update model configuration from payload."""
        # Update global config
        config.update_from_payload({"ollama": payload})
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
        base_url = config.get('ollama.base_url', 'http://localhost:11434')
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            results["ollama_running"] = True
            
            # Check models
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            # Check main model
            main_model = config.get("ollama.model", "gemma:latest")
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