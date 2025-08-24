#!/usr/bin/env python3
"""
Simple Ollama Test - Verify Ollama is working
"""

import requests
import json

def test_ollama_simple():
    """Test Ollama with a very simple prompt"""
    print("🧪 Testing Ollama with simple prompt...")
    
    try:
        # Test with a very simple prompt
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "gemma:2b",
            "prompt": "Say hello in one word.",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "max_tokens": 10
            }
        }
        
        print("📤 Sending simple request to Ollama...")
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "")
        
        print(f"✅ Ollama is working! Response: '{response_text.strip()}'")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running!")
        print("💡 Start Ollama with: ollama serve")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Ollama may be overloaded.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_available_models():
    """Check what models are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("📋 Available models:")
            for model in models:
                print(f"  - {model['name']}")
            return models
        else:
            print("❌ Could not get model list")
            return []
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return []

if __name__ == "__main__":
    print("🎯 SIMPLE OLLAMA TEST")
    print("=" * 30)
    
    # Check available models
    models = check_available_models()
    
    if not models:
        print("❌ No models found. Please pull models first:")
        print("   ollama pull gemma:2b")
        exit(1)
    
    # Test simple prompt
    if test_ollama_simple():
        print("\n🎉 Ollama is working correctly!")
        print("💡 You can now run the full evaluation:")
        print("   python gemma_ollama_optimized.py")
    else:
        print("\n❌ Ollama test failed!")
        print("💡 Check if Ollama is running: ollama serve")
