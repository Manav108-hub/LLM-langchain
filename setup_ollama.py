import subprocess
import json
import sys
import os

def check_ollama_installed():
    try:
        subprocess.run(["ollama", "--version"], check=True, stdout=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_model_downloaded(model_name):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        return model_name in result.stdout
    except subprocess.CalledProcessError:
        return False

def pull_model(model_name):
    try:
        print(f"📥 Pulling model: {model_name}...")
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ Pulled {model_name} successfully.\n")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to pull model: {model_name}\n")
        return False

def setup_models():
    required_models = {
        "gemma": "Main language model (Gemma 7B)",
        "nomic-embed-text": "Embedding model (274MB)"
    }

    print("🔍 Checking for required models...\n")
    model_results = {}
    for model, desc in required_models.items():
        if check_model_downloaded(model):
            print(f"✅ {model} ({desc}) is already downloaded.")
            model_results[model] = True
        else:
            print(f"⬇️  {model} ({desc}) is not present. Pulling...")
            success = pull_model(model)
            model_results[model] = success

    return model_results

def create_configuration(model_results):
    config = {
        "model": "gemma" if model_results.get("gemma", False) else "gemma:2b",
        "embedding_model": "nomic-embed-text" if model_results.get("nomic-embed-text", False) else "default"
    }

    with open("ollama_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\n🛠️  Created config file: `ollama_config.json`\n")
    print(json.dumps(config, indent=4))

def run_validation_test():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        print("\n📃 Installed Ollama Models:\n")
        print(result.stdout)
        model_names = result.stdout

        if "gemma" in model_names or "gemma:7b" in model_names:
            print("✅ Gemma model is available")
        else:
            print("⚠️ Gemma model not found")

        if "nomic-embed-text" in model_names:
            print("✅ Embedding model is available")
        else:
            print("⚠️ Embedding model not found")

    except subprocess.CalledProcessError as e:
        print("❌ Could not validate installed models.")
        print(e)

def create_sample_payloads():
    sample_payload = {
        "model": "gemma",
        "prompt": "What are Large Language Models and how are they used?",
        "stream": False
    }

    with open("sample_payload.json", "w") as f:
        json.dump(sample_payload, f, indent=4)

    print("\n📝 Created `sample_payload.json` for testing.")
    print(json.dumps(sample_payload, indent=4))

def main():
    print("🚀 Ollama Model Setup Utility\n")

    if not check_ollama_installed():
        print("❌ Ollama is not installed or not in PATH.")
        print("👉 Install it from https://ollama.com/download\n")
        sys.exit(1)

    model_results = setup_models()
    create_configuration(model_results)
    run_validation_test()
    create_sample_payloads()

    print("\n✅ Setup complete. You can now use Ollama in your LangChain project.")

if __name__ == "__main__":
    main()
