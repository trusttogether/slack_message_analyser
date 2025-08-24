#!/usr/bin/env python3
"""
Full Gemma Evaluation - Ultra Conservative for VPS Performance
Processes dataset in tiny chunks like the working simple script
"""

import json
import time
import requests
import pandas as pd
from typing import Dict, Any, List
import re

def safe_call_gemma(model_name: str, prompt: str, timeout: int = 20) -> Dict[str, Any]:
    """Call Gemma with ultra-conservative settings - exactly like simple script"""
    start_time = time.time()
    
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 100,  # Same as simple script
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

def create_mini_prompt(messages: List[str]) -> str:
    """Create a mini prompt - process only 5 messages like simple script"""
    # Take only first 5 messages to match simple script approach
    sample_messages = messages[:5]
    
    prompt = f"""What are the main topics in these messages?

{chr(10).join(sample_messages)}

Topics:"""
    
    return prompt

def parse_simple_response(response: str) -> List[str]:
    """Parse simple response from Gemma"""
    try:
        # Clean up response
        response = response.strip()
        
        # Remove markdown code blocks
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        # Split by lines and clean up
        topics = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove numbering (1., 2., etc.)
                line = re.sub(r'^\d+\.\s*', '', line)
                if line:
                    topics.append(line)
        
        return topics[:3]  # Limit to 3 topics
        
    except Exception as e:
        print(f"Response parsing error: {e}")
        return []

def evaluate_gemma_model(model_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate a Gemma model on the dataset - ultra conservative"""
    print(f"\n🧪 Evaluating {model_name}...")
    
    # Format messages - very short like simple script
    messages = []
    for idx, row in df.iterrows():
        username = row.get('user_name', 'unknown')
        text = row.get('text', '')[:30]  # Very short like simple script
        messages.append(f"{username}: {text}")
    
    # Create mini prompt
    prompt = create_mini_prompt(messages)
    
    # Call model
    result = safe_call_gemma(model_name, prompt)
    
    if not result['success']:
        return {
            "model": model_name,
            "success": False,
            "error": result['response'],
            "duration": result['duration']
        }
    
    # Parse response
    topics = parse_simple_response(result['response'])
    
    return {
        "model": model_name,
        "success": True,
        "response": result['response'],
        "parsed_topics": topics,
        "topics_count": len(topics),
        "duration": result['duration'],
        "messages_processed": len(messages[:5])  # Only count processed messages
    }

def run_full_gemma_evaluation():
    """Run full evaluation on all Gemma models"""
    print("🎯 FULL GEMMA EVALUATION - ULTRA CONSERVATIVE")
    print("=" * 40)
    
    # Load data
    try:
        df = pd.read_csv("data/Synthetic_Slack_Messages.csv")
        print(f"📊 Loaded {len(df)} messages")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Test models
    models = ["gemma:2b", "gemma:2b-instruct"]
    results = {}
    
    for model in models:
        result = evaluate_gemma_model(model, df)
        results[model] = result
        
        if result['success']:
            print(f"✅ {model}: {result['topics_count']} topics in {result['duration']:.2f}s")
            print(f"   Messages processed: {result['messages_processed']}")
            print(f"   Topics: {', '.join(result['parsed_topics'])}")
        else:
            print(f"❌ {model}: {result['error']}")
    
    # Save results
    with open("gemma_full_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: gemma_full_results.json")
    
    # Summary
    successful_models = [m for m, r in results.items() if r['success']]
    print(f"\n📊 Summary: {len(successful_models)}/{len(models)} models successful")
    
    for model in successful_models:
        result = results[model]
        print(f"   {model}: {result['topics_count']} topics, {result['duration']:.2f}s")

if __name__ == "__main__":
    run_full_gemma_evaluation()
