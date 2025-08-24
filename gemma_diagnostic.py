#!/usr/bin/env python3
"""
Gemma Diagnostic - Check VPS Hardware and Model Performance
Tests with absolute minimum settings
"""

import json
import time
import requests
import subprocess
import psutil
from typing import Dict, Any

def check_system_resources():
    """Check system resources"""
    print("🖥️  SYSTEM RESOURCES")
    print("=" * 30)
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"CPU: {cpu_count} cores, {cpu_percent}% usage")
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total // (1024**3)}GB total, {memory.percent}% used")
    
    # Disk
    disk = psutil.disk_usage('/')
    print(f"Disk: {disk.total // (1024**3)}GB total, {disk.percent}% used")
    
    # Check if Ollama is running
    try:
        result = subprocess.run(['pgrep', 'ollama'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is running")
        else:
            print("❌ Ollama is not running")
    except:
        print("❌ Cannot check Ollama status")

def test_gemma_minimal(model_name: str, prompt: str, timeout: int = 10) -> Dict[str, Any]:
    """Test with absolute minimum settings"""
    start_time = time.time()
    
    print(f"\n🧪 Testing {model_name} with minimal prompt...")
    print(f"   Prompt: '{prompt}'")
    print(f"   Timeout: {timeout}s")
    
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 50,  # Very short response
                "top_k": 1,
                "top_p": 0.5
            }
        }
        
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "")
        
        duration = time.time() - start_time
        
        return {
            "success": True,
            "response": response_text,
            "duration": duration
        }
        
    except requests.exceptions.Timeout:
        duration = time.time() - start_time
        return {
            "success": False,
            "response": "TIMEOUT",
            "duration": duration
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "response": f"ERROR: {str(e)}",
            "duration": duration
        }

def run_diagnostic():
    """Run comprehensive diagnostic"""
    print("🔍 GEMMA DIAGNOSTIC")
    print("=" * 40)
    
    # Check system
    check_system_resources()
    
    # Test 1: Absolute minimum
    print("\n1️⃣ Testing with single word prompt...")
    result1 = test_gemma_minimal("gemma:2b", "Hello", timeout=5)
    print(f"   Success: {result1['success']}")
    print(f"   Duration: {result1['duration']:.2f}s")
    print(f"   Response: '{result1['response'].strip()}'")
    
    # Test 2: Short prompt
    print("\n2️⃣ Testing with short prompt...")
    result2 = test_gemma_minimal("gemma:2b", "What is 2+2?", timeout=10)
    print(f"   Success: {result2['success']}")
    print(f"   Duration: {result2['duration']:.2f}s")
    print(f"   Response: '{result2['response'].strip()}'")
    
    # Test 3: Different model
    print("\n3️⃣ Testing gemma:2b-instruct...")
    result3 = test_gemma_minimal("gemma:2b-instruct", "Hi", timeout=5)
    print(f"   Success: {result3['success']}")
    print(f"   Duration: {result3['duration']:.2f}s")
    print(f"   Response: '{result3['response'].strip()}'")
    
    # Save results
    results = {
        "system_check": {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total // (1024**3),
            "ollama_running": subprocess.run(['pgrep', 'ollama'], capture_output=True).returncode == 0
        },
        "test1": result1,
        "test2": result2,
        "test3": result3
    }
    
    with open("gemma_diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: gemma_diagnostic_results.json")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    successful_tests = sum(1 for r in [result1, result2, result3] if r['success'])
    
    if successful_tests == 0:
        print("❌ All tests failed. Possible issues:")
        print("   - VPS hardware too weak for Gemma models")
        print("   - Ollama not running properly")
        print("   - Need more RAM/CPU resources")
        print("   - Consider using cloud APIs instead")
    elif successful_tests < 3:
        print("⚠️  Some tests failed. Consider:")
        print("   - Using shorter prompts")
        print("   - Reducing model complexity")
        print("   - Using cloud APIs for complex tasks")
    else:
        print("✅ All tests passed! Gemma models are working.")

if __name__ == "__main__":
    run_diagnostic()
