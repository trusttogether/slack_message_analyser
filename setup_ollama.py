#!/usr/bin/env python3
"""
Ollama Setup Script for Local Gemma Models
Helps install Ollama and pull Gemma models
"""

import subprocess
import sys
import time
import requests

def run_command(command, description):
    """Run a shell command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def check_ollama_installed():
    """Check if Ollama is already installed"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama is already installed: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def install_ollama():
    """Install Ollama on Linux"""
    print("🚀 Installing Ollama...")
    
    # Download and install Ollama
    install_command = "curl -fsSL https://ollama.ai/install.sh | sh"
    if run_command(install_command, "Installing Ollama"):
        print("✅ Ollama installation completed!")
        return True
    else:
        print("❌ Ollama installation failed!")
        return False

def start_ollama_service():
    """Start Ollama service"""
    print("🚀 Starting Ollama service...")
    
    # Start Ollama in background
    start_command = "ollama serve &"
    if run_command(start_command, "Starting Ollama service"):
        # Wait a moment for service to start
        time.sleep(3)
        return True
    else:
        return False

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running!")
            return True
        else:
            print("❌ Ollama is not responding properly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def pull_gemma_models():
    """Pull Gemma models from Ollama"""
    print("📥 Pulling Gemma models...")
    
    gemma_models = [
        "gemma:2b",
        "gemma:7b", 
        "gemma:2b-instruct",
        "gemma:7b-instruct"
    ]
    
    success_count = 0
    for model in gemma_models:
        print(f"📥 Pulling {model}...")
        pull_command = f"ollama pull {model}"
        if run_command(pull_command, f"Pulling {model}"):
            success_count += 1
        else:
            print(f"⚠️  Failed to pull {model}, continuing with others...")
    
    print(f"✅ Successfully pulled {success_count}/{len(gemma_models)} models")
    return success_count > 0

def list_available_models():
    """List available models in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("📋 Available models in Ollama:")
            for model in models:
                print(f"  - {model['name']} ({model.get('size', 'Unknown size')})")
            return True
        else:
            print("❌ Could not list models")
            return False
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False

def main():
    """Main setup function"""
    print("🎯 OLLAMA SETUP FOR LOCAL GEMMA MODELS")
    print("=" * 45)
    
    # Check if Ollama is already installed
    if not check_ollama_installed():
        print("📦 Ollama not found. Installing...")
        if not install_ollama():
            print("❌ Failed to install Ollama. Please install manually:")
            print("   Visit: https://ollama.ai/")
            return False
    
    # Start Ollama service
    if not check_ollama_running():
        print("🚀 Starting Ollama service...")
        if not start_ollama_service():
            print("❌ Failed to start Ollama service")
            print("💡 Try running manually: ollama serve")
            return False
        
        # Wait and check again
        time.sleep(5)
        if not check_ollama_running():
            print("❌ Ollama service is not responding")
            return False
    
    # Pull Gemma models
    print("📥 Pulling Gemma models (this may take a while)...")
    if not pull_gemma_models():
        print("⚠️  Some models failed to pull, but continuing...")
    
    # List available models
    list_available_models()
    
    print("\n🎉 Ollama setup completed!")
    print("\n💡 Next steps:")
    print("   1. Run the evaluation: python gemma_ollama_evaluation.py")
    print("   2. Or test a single model: ollama run gemma:2b")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
