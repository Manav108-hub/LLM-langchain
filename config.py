"""Configuration settings for Document AI Agent with Ollama."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration manager with JSON payload support."""

    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self._config = self._load_config()
        self._setup_directories()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"✅ Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                print(f"⚠️ Error loading config file: {e}")
                print("Using default configuration...")

        # Default configuration
        default_config = self._get_default_config()
        self.save_config(default_config)
        return default_config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Ollama + Gemma 8B."""
        return {
            "app": {
                "name": "Document AI Agent",
                "version": "1.0.0",
                "description": "Intelligent document processing with Ollama & LangChain",
                "debug": True,
                "log_level": "INFO"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "reload": True
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "gemma:8b",
                "embedding_model": "nomic-embed-text",
                "timeout": 300,
                "keep_alive": "5m",
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": False
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "provider": "huggingface",  # Options: huggingface, ollama
                "dimension": 384,
                "normalize": True,
                "batch_size": 32
            },
            "document_processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "top_k_retrieval": 5,
                "confidence_threshold": 0.7,
                "max_file_size": 52428800,
                "allowed_extensions": [".pdf", ".docx", ".txt", ".md", ".csv"],
                "max_files_per_request": 10
            },
            "vector_store": {
                "type": "faiss",
                "path": "data/vector_store",
                "index_name": "documents",
                "save_every": 10
            },
            "directories": {
                "data": "data",
                "uploads": "uploads",
                "templates": "templates",
                "cache": "cache",
                "logs": "logs"
            },
            "features": {
                "conversation_memory": True,
                "document_analysis": True,
                "similarity_search": True,
                "batch_processing": True,
                "caching": False
            },
            "security": {
                "enable_auth": False,
                "secret_key": "change-this-in-production",
                "cors_origins": ["http://localhost:3000", "http://localhost:8080"],
                "cors_allow_credentials": True
            },
            "performance": {
                "max_concurrent_requests": 10,
                "request_timeout": 300,
                "cache_ttl": 3600,
                "enable_profiling": False
            }
        }

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        base_dir = Path.cwd()
        for dir_name in self._config["directories"].values():
            dir_path = base_dir / dir_name
            dir_path.mkdir(exist_ok=True)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config

        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update configuration from JSON payload."""
        def update_nested(target: Dict, source: Dict):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_nested(target[key], value)
                else:
                    target[key] = value

        update_nested(self._config, payload)
        self.save_config()
        print("✅ Configuration updated from payload")

    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to JSON file."""
        try:
            config_to_save = config or self._config
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")

    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama-specific configuration."""
        return self._config.get("ollama", {})

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return self._config.get("embedding", {})

    def validate_ollama_connection(self) -> bool:
        """Validate Ollama connection and model availability."""
        try:
            import requests

            ollama_config = self.get_ollama_config()
            base_url = ollama_config.get("base_url", "http://localhost:11434")

            # Check if Ollama is running
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if model is available
            models = response.json().get("models", [])
            model_name = ollama_config.get("model", "gemma:8b")

            available_models = [model["name"] for model in models]
            if model_name not in available_models:
                print(f"⚠️ Model {model_name} not found. Available models: {available_models}")
                return False

            print(f"✅ Ollama connection validated. Model {model_name} is available.")
            return True

        except Exception as e:
            print(f"❌ Ollama validation failed: {e}")
            return False

    def get_paths(self) -> Dict[str, Path]:
        """Get all directory paths."""
        base_dir = Path.cwd()
        return {
            name: base_dir / path
            for name, path in self._config["directories"].items()
        }

    @property
    def app_name(self) -> str:
        return self.get("app.name", "Document AI Agent")

    @property
    def debug(self) -> bool:
        return self.get("app.debug", True)

    @property
    def host(self) -> str:
        return self.get("server.host", "0.0.0.0")

    @property
    def port(self) -> int:
        return self.get("server.port", 8000)

# Global configuration instance
config = Config()

# Convenience functions for backward compatibility
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value."""
    return config.get(key, default)

def set_config(key: str, value: Any) -> None:
    """Set configuration value."""
    config.set(key, value)

def update_config_from_payload(payload: Dict[str, Any]) -> None:
    """Update configuration from payload."""
    config.update_from_payload(payload)

# Export commonly used values
APP_NAME = config.app_name
DEBUG = config.debug
HOST = config.host
PORT = config.port

# Paths
PATHS = config.get_paths()
DATA_DIR = PATHS["data"]
UPLOADS_DIR = PATHS["uploads"]
TEMPLATES_DIR = PATHS["templates"]
CACHE_DIR = PATHS["cache"]
LOGS_DIR = PATHS["logs"]

# Document processing settings
CHUNK_SIZE = config.get("document_processing.chunk_size", 1000)
CHUNK_OVERLAP = config.get("document_processing.chunk_overlap", 200)
TOP_K_RETRIEVAL = config.get("document_processing.top_k_retrieval", 5)
CONFIDENCE_THRESHOLD = config.get("document_processing.confidence_threshold", 0.7)
MAX_FILE_SIZE = config.get("document_processing.max_file_size", 52428800)
ALLOWED_EXTENSIONS = set(config.get("document_processing.allowed_extensions", [".pdf", ".docx", ".txt"]))

# Vector store settings
VECTOR_STORE_PATH = PATHS["data"] / config.get("vector_store.path", "vector_store")
VECTOR_STORE_TYPE = config.get("vector_store.type", "faiss")

print(f"📊 Configuration loaded - Ollama: {config.get('ollama.model', 'gemma:8b')}")
