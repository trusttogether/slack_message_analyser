#!/usr/bin/env python3
"""
Ultra-Simple Gemma Evaluation - Minimal Data, Fast Results
Uses only 10 messages to avoid timeouts
"""

import json
import time
import requests
import pandas as pd
from typing import Dict, Any, List

def call_gemma_fast(model_name: str, prompt: str, timeout: int = 20) -> Dict[str, Any]:
    """Call Gemma with ultra-fast settings"""
    start_time = time.time()
    
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200,  # Very short response
                "top_k": 5,
                "top_p": 0.8
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

def create_tiny_prompt(messages: List[str]) -> str:
    """Create a tiny prompt with only 10 messages"""
    # Take only first 10 messages
    tiny_messages = messages[:10]
    
    prompt = f"""Find topics in these messages:
{chr(10).join(tiny_messages)}

Topics:"""
    
    return prompt

def evaluate_gemma_ultra_simple(model_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate Gemma with ultra-simple approach"""
    print(f"\n🧪 Testing {model_name} with ultra-simple approach...")
    
    # Format only 10 messages
    messages = []
    for idx, row in df.head(10).iterrows():
        username = row.get('user_name', 'unknown')
        text = row.get('text', '')[:50]  # Very short text
        messages.append(f"{username}: {text}")
    
    # Create tiny prompt
    prompt = create_tiny_prompt(messages)
    
    print(f"   Prompt length: {len(prompt)} characters")
    print(f"   Messages: {len(messages)}")
    
    # Call model
    result = call_gemma_fast(model_name, prompt)
    
    if not result['success']:
        return {
            "model": model_name,
            "success": False,
            "error": result['response'],
            "duration": result['duration']
        }
    
    return {
        "model": model_name,
        "success": True,
        "response": result['response'],
        "duration": result['duration'],
        "messages_processed": len(messages)
    }

def run_ultra_simple_evaluation():
    """Run ultra-simple evaluation"""
    print("🎯 ULTRA-SIMPLE GEMMA EVALUATION")
    print("=" * 40)
    
    # Load data
    try:
        df = pd.read_csv("data/Synthetic_Slack_Messages.csv")
        print(f"📊 Loaded {len(df)} messages (will use only first 10)")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Test models
    models = ["gemma:2b", "gemma:2b-instruct"]
    results = {}
    
    for model in models:
        result = evaluate_gemma_ultra_simple(model, df)
        results[model] = result
        
        if result['success']:
            print(f"✅ {model}: {result['duration']:.2f}s")
            print(f"   Response: '{result['response'].strip()}'")
        else:
            print(f"❌ {model}: {result['error']}")
    
    # Save results
    with open("gemma_ultra_simple_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: gemma_ultra_simple_results.json")
    
    # Summary
    successful_models = [m for m, r in results.items() if r['success']]
    print(f"\n📊 Summary: {len(successful_models)}/{len(models)} models successful")
    
    if successful_models:
        print("✅ Gemma models are working with minimal data!")
        print("💡 You can now gradually increase the data size.")

if __name__ == "__main__":
    run_ultra_simple_evaluation()
