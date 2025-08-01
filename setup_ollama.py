#!/usr/bin/env python3
"""
Ollama setup script for Document AI Agent.
Handles Ollama installation, model management, and configuration.
"""

import json
import os
import sys
import subprocess
import requests
import time
from pathlib import Path
from typing import Dict, List, Any

def print_banner():
    """Print setup banner."""
    print("\n" + "="*70)
    print("🦙 Document AI Agent - Ollama Setup")
    print("="*70)
    print("Setting up Ollama with Gemma 8B for local AI processing")
    print("="*70 + "\n")

def check_ollama_installed() -> bool:
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(['ollama', '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def install_ollama():
    """Install Ollama if not present."""
    print("📥 Installing Ollama...")

    # Detect OS and install accordingly
    import platform
    system = platform.system().lower()

    try:
        if system == "darwin":  # macOS
            print("Installing Ollama for macOS...")
            subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'],
                         capture_output=True, check=True)

        elif system == "linux":
            print("Installing Ollama for Linux...")
            subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh', '|', 'sh'],
                         shell=True, check=True)

        elif system == "windows":
            print("For Windows, please download Ollama from: https://ollama.ai/download")
            print("Then run this script again.")
            return False

        else:
            print(f"❌ Unsupported OS: {system}")
            return False

        # Wait a bit for installation to complete
        time.sleep(3)

        if check_ollama_installed():
            print("✅ Ollama installed successfully!")
            return True
        else:
            print("❌ Ollama installation may have failed")
            return False

    except Exception as e:
        print(f"❌ Error installing Ollama: {e}")
        return False

def start_ollama_service():
    """Start Ollama service."""
    print("🚀 Starting Ollama service...")

    try:
        # Start Ollama in background
        if sys.platform == "win32":
            subprocess.Popen(['ollama', 'serve'], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait for service to start
        print("⏳ Waiting for Ollama service to start...")
        for _ in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print("✅ Ollama service is running!")
                    return True
            except:
                pass
            time.sleep(1)

        print("❌ Ollama service failed to start")
        return False

    except Exception as e:
        print(f"❌ Error starting Ollama: {e}")
        return False

def check_ollama_running() -> bool:
    """Check if Ollama service is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def list_available_models() -> List[Dict[str, Any]]:
    """List models available in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

def pull_model(model_name: str) -> bool:
    """Pull/download a model."""
    print(f"📥 Pulling model: {model_name}")
    print("This may take several minutes depending on model size...")

    try:
        # Use subprocess to show progress
        result = subprocess.run(['ollama', 'pull', model_name],
                              capture_output=True, text=True, timeout=1800)  # 30 minutes

        if result.returncode == 0:
            print(f"✅ Model {model_name} pulled successfully!")
            return True
        else:
            print(f"❌ Failed to pull model {model_name}")
            print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"❌ Timeout while pulling model {model_name}")
        return False
    except Exception as e:
        print(f"❌ Error pulling model {model_name}: {e}")
        return False

def setup_models() -> Dict[str, bool]:
    """Setup required models."""
    print("\n📚 Setting up AI models...")

    required_models = {
        "gemma:8b": "Main language model (4.8GB)",
        "nomic-embed-text": "Embedding model (274MB)"
    }

    results = {}
    existing_models = [model["name"] for model in list_available_models()]

    for model_name, description in required_models.items():
        print(f"\n🔍 Checking {model_name} - {description}")

        if model_name in existing_models:
            print(f"✅ {model_name} is already available")
            results[model_name] = True
        else:
            print(f"📥 {model_name} not found, downloading...")
            results[model_name] = pull_model(model_name)

    return results

def create_configuration(model_results: Dict[str, bool]) -> bool:
    """Create configuration file."""
    print("\n⚙️ Creating configuration...")

    config_data = {
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
            "model": "gemma:8b" if model_results.get("gemma:8b", False) else "gemma:2b",
            "embedding_model": "nomic-embed-text" if model_results.get("nomic-embed-text", False) else None,
            "timeout": 300,
            "keep_alive": "5m",
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        },
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "provider": "ollama" if model_results.get("nomic-embed-text", False) else "huggingface",
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

    try:
        with open("config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        print("✅ Configuration file created: config.json")
        return True
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return False

def create_sample_payloads():
    """Create sample JSON payloads for testing."""
    print("\n📄 Creating sample payloads...")

    # Sample configuration update payload
    config_update_payload = {
        "ollama": {
            "temperature": 0.8,
            "max_tokens": 512,
            "model": "gemma:8b"
        },
        "document_processing": {
            "chunk_size": 800,
            "confidence_threshold": 0.6
        }
    }

    # Sample question payload
    question_payload = {
        "question": "What are the main topics discussed in the uploaded documents?",
        "config": {
            "ollama": {
                "temperature": 0.5,
                "max_tokens": 300
            },
            "document_processing": {
                "top_k_retrieval": 3
            }
        }
    }

    # Sample model management payload
    model_management_payload = {
        "model_name": "llama2:7b",
        "action": "pull"
    }

    try:
        # Create payloads directory
        payloads_dir = Path("sample_payloads")
        payloads_dir.mkdir(exist_ok=True)

        # Save sample payloads
        with open(payloads_dir / "config_update.json", "w") as f:
            json.dump(config_update_payload, f, indent=2)

        with open(payloads_dir / "question_with_config.json", "w") as f:
            json.dump(question_payload, f, indent=2)

        with open(payloads_dir / "model_management.json", "w") as f:
            json.dump(model_management_payload, f, indent=2)

        print("✅ Sample payloads created in sample_payloads/ directory")
        return True

    except Exception as e:
        print(f"❌ Error creating sample payloads: {e}")
        return False

def run_validation_test():
    """Run a quick validation test."""
    print("\n🧪 Running validation test...")

    try:
        # Test Ollama connection
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            print(f"✅ Ollama is running with {len(models)} models")

            if "gemma:8b" in model_names:
                print("✅ Gemma 8B model is available")
            else:
                print("⚠️ Gemma 8B model not found")

            return True
        else:
            print("❌ Ollama is not responding correctly")
            return False

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def print_usage_examples():
    """Print usage examples and next steps."""
    print("\n" + "="*70)
    print("🎉 Ollama Setup Complete!")
    print("="*70)

    print("\n📋 What was installed:")
    print("   • Ollama service running on http://localhost:11434")
    print("   • Gemma 8B model for text generation")
    print("   • Configuration file: config.json")
    print("   • Sample payloads in sample_payloads/")

    print("\n🚀 Next Steps:")
    print("1. Install Python dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Start the Document AI Agent:")
    print("   python main.py")
    print("\n3. Open your browser:")
    print("   http://localhost:8000")

    print("\n🔧 Configuration Management:")
    print("• Edit config.json to modify settings")
    print("• Use /config/update endpoint with JSON payloads")
    print("• Sample payloads available in sample_payloads/")

    print("\n💡 API Examples:")
    print("• Ask question: POST /ask")
    print('  {"question": "What is this document about?"}')
    print("\n• Ask with custom config: POST /ask-with-payload")
    print('  {"question": "...", "config": {"ollama": {"temperature": 0.8}}}')
    print("\n• Update config: POST /config/update")
    print('  {"config": {"ollama": {"model": "gemma:2b"}}}')
    print("\n• Manage models: POST /models/manage")
    print('  {"model_name": "llama2:7b", "action": "pull"}')

    print("\n📚 Resources:")
    print("   • Ollama models: https://ollama.ai/library")
    print("   • API docs: http://localhost:8000/docs")
    print("   • Health check: http://localhost:8000/health")

    print("\n" + "="*70)

def main():
    """Main setup function."""
    print_banner()

    # Check if Ollama is already installed and running
    if check_ollama_running():
        print("✅ Ollama is already running!")
    else:
        # Check if Ollama is installed
        if not check_ollama_installed():
            print("❌ Ollama not found. Installing...")
            if not install_ollama():
                print("Setup failed. Please install Ollama manually from https://ollama.ai")
                sys.exit(1)

        # Start Ollama service
        if not start_ollama_service():
            print("❌ Failed to start Ollama service")
            print("Please start Ollama manually: ollama serve")
            sys.exit(1)

    # Setup models
    model_results = setup_models()

    # Check if essential models are available
    if not model_results.get("gemma:8b", False):
        print("⚠️ Warning: Gemma 8B model not available")
        print("You can still use the system, but responses may be limited")

    # Create configuration
    if not create_configuration(model_results):
        print("❌ Failed to create configuration")
        sys.exit(1)

    # Create sample payloads
    create_sample_payloads()

    # Run validation test
    run_validation_test()

    # Print usage information
    print_usage_examples()

if __name__ == "__main__":
    main()
