#!/usr/bin/env python3
"""
VPS Gemma Test - Very Simple Testing for Local Gemma Models
Uses minimal prompts and short timeouts
"""

import json
import time
import requests
from typing import Dict, Any

def test_gemma_basic(model_name: str, prompt: str, timeout: int = 15) -> Dict[str, Any]:
    """Test a Gemma model with a very basic prompt"""
    start_time = time.time()
    
    print(f"🧪 Testing {model_name} with basic prompt...")
    
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "max_tokens": 200,  # Very short response
                "num_predict": 200
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
            "duration": duration,
            "prompt": prompt
        }
        
    except requests.exceptions.Timeout:
        duration = time.time() - start_time
        return {
            "success": False,
            "model": model_name,
            "response": "TIMEOUT",
            "duration": duration,
            "prompt": prompt
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "model": model_name,
            "response": f"ERROR: {str(e)}",
            "duration": duration,
            "prompt": prompt
        }

def run_vps_gemma_tests():
    """Run basic tests on VPS Gemma models"""
    print("🎯 VPS GEMMA MODEL TESTING")
    print("=" * 40)
    
    # Test 1: Very simple prompt
    test1_prompt = "Say hello in one word."
    test1_result = test_gemma_basic("gemma:2b", test1_prompt, timeout=10)
    
    print(f"Test 1 - Simple prompt:")
    print(f"  Success: {test1_result['success']}")
    print(f"  Duration: {test1_result['duration']:.2f}s")
    print(f"  Response: '{test1_result['response'].strip()}'")
    print()
    
    # Test 2: Short topic detection
    test2_prompt = """Find topics in these messages:
Alice: Meeting at 3pm
Bob: Project deadline Friday
Alice: Budget review needed

Topics:"""
    
    test2_result = test_gemma_basic("gemma:2b", test2_prompt, timeout=15)
    
    print(f"Test 2 - Short topic detection:")
    print(f"  Success: {test2_result['success']}")
    print(f"  Duration: {test2_result['duration']:.2f}s")
    print(f"  Response: '{test2_result['response'].strip()}'")
    print()
    
    # Test 3: JSON format request
    test3_prompt = """Return JSON array with 2 topics:
[{"title": "Meeting", "summary": "Team meeting"}, {"title": "Project", "summary": "Project work"}]"""
    
    test3_result = test_gemma_basic("gemma:2b", test3_prompt, timeout=15)
    
    print(f"Test 3 - JSON format:")
    print(f"  Success: {test3_result['success']}")
    print(f"  Duration: {test3_result['duration']:.2f}s")
    print(f"  Response: '{test3_result['response'].strip()}'")
    print()
    
    # Test 4: Test different models
    models_to_test = ["gemma:2b", "gemma:2b-instruct"]
    
    print("Testing different models with simple prompt:")
    for model in models_to_test:
        result = test_gemma_basic(model, "What is 2+2?", timeout=10)
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {model}: {result['duration']:.2f}s - '{result['response'].strip()}'")
    
    # Save results
    all_results = {
        "test1": test1_result,
        "test2": test2_result,
        "test3": test3_result
    }
    
    with open("vps_gemma_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 Results saved to: vps_gemma_test_results.json")
    print("\n🎉 VPS Gemma testing complete!")

if __name__ == "__main__":
    run_vps_gemma_tests()
