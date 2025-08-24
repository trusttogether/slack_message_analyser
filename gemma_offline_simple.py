#!/usr/bin/env python3
"""
Offline Gemma Test - Minimal Testing to Avoid Timeouts
Uses tiny prompts and very short responses
"""

import json
import time
import requests
import pandas as pd
from typing import Dict, Any, List

def test_gemma_minimal(model_name: str, prompt: str, timeout: int = 30) -> Dict[str, Any]:
    """Test Gemma with minimal settings"""
    start_time = time.time()
    
    print(f"🧪 Testing {model_name}...")
    
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 100,  # Very short response
                "top_k": 10,
                "top_p": 0.9
            }
        }
        
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "")
        
        duration = time.time() - start_time
        
        return {
            "success": True,
            "model": model_name,
            "response": response_text,
            "duration": duration
        }
        
    except requests.exceptions.Timeout:
        duration = time.time() - start_time
        return {
            "success": False,
            "model": model_name,
            "response": "TIMEOUT",
            "duration": duration
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "model": model_name,
            "response": f"ERROR: {str(e)}",
            "duration": duration
        }

def create_mini_dataset() -> str:
    """Create a very small dataset for testing"""
    mini_messages = [
        "Alice: Meeting at 3pm today",
        "Bob: Project deadline is Friday", 
        "Alice: Budget review needed",
        "Bob: Client presentation tomorrow",
        "Alice: Team lunch on Monday"
    ]
    return "\n".join(mini_messages)

def run_offline_gemma_tests():
    """Run minimal offline Gemma tests"""
    print("🎯 OFFLINE GEMMA TESTING - MINIMAL APPROACH")
    print("=" * 50)
    
    # Test 1: Basic functionality
    print("\n1️⃣ Testing basic response...")
    basic_result = test_gemma_minimal("gemma:2b", "Say 'hello'", timeout=15)
    print(f"   Success: {basic_result['success']}")
    print(f"   Duration: {basic_result['duration']:.2f}s")
    print(f"   Response: '{basic_result['response'].strip()}'")
    
    # Test 2: Simple topic detection
    print("\n2️⃣ Testing simple topic detection...")
    mini_data = create_mini_dataset()
    topic_prompt = f"""Find 2-3 topics in these messages:
{mini_data}

Topics:"""
    
    topic_result = test_gemma_minimal("gemma:2b", topic_prompt, timeout=30)
    print(f"   Success: {topic_result['success']}")
    print(f"   Duration: {topic_result['duration']:.2f}s")
    print(f"   Response: '{topic_result['response'].strip()}'")
    
    # Test 3: JSON format (very simple)
    print("\n3️⃣ Testing JSON format...")
    json_prompt = """Return this exact JSON: [{"title": "test"}]"""
    
    json_result = test_gemma_minimal("gemma:2b", json_prompt, timeout=20)
    print(f"   Success: {json_result['success']}")
    print(f"   Duration: {json_result['duration']:.2f}s")
    print(f"   Response: '{json_result['response'].strip()}'")
    
    # Test 4: Compare models
    print("\n4️⃣ Comparing models...")
    models = ["gemma:2b", "gemma:2b-instruct"]
    
    for model in models:
        result = test_gemma_minimal(model, "What is 1+1?", timeout=15)
        status = "✅" if result['success'] else "❌"
        print(f"   {status} {model}: {result['duration']:.2f}s - '{result['response'].strip()}'")
    
    # Save results
    all_results = {
        "basic_test": basic_result,
        "topic_test": topic_result,
        "json_test": json_result,
        "timestamp": time.time()
    }
    
    with open("offline_gemma_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 Results saved to: offline_gemma_results.json")
    
    # Summary
    successful_tests = sum(1 for r in [basic_result, topic_result, json_result] if r['success'])
    print(f"\n📊 Summary: {successful_tests}/3 tests successful")
    
    if successful_tests > 0:
        print("✅ Gemma models are working! You can now proceed with larger tests.")
    else:
        print("❌ All tests failed. Check if Ollama is running: ollama serve")

if __name__ == "__main__":
    run_offline_gemma_tests()
