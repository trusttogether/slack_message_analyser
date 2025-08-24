#!/usr/bin/env python3
"""
Optimized Local Gemma Evaluation using Ollama
Uses shorter prompts and better timeout handling
"""

import json
import time
import os
import pandas as pd
import requests
from typing import Dict, Any, List

# Import configuration
from config.config import (
    DATA_DIR, PROMPT_DIR
)

def safe_call_ollama_gemma_optimized(model_name: str, prompt: str, timeout: int = 60) -> str:
    """Safely call Gemma model via Ollama API with optimized settings"""
    try:
        # Ollama API endpoint
        url = "http://localhost:11434/api/generate"
        
        # Prepare the request payload with optimized settings
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower temperature for more consistent output
                "top_p": 0.8,
                "max_tokens": 1000,  # Reduced token limit
                "num_predict": 1000,
                "stop": ["```", "Human:", "Assistant:"]  # Stop tokens
            }
        }
        
        # Make the API call with shorter timeout
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        # Extract the response text
        result = response.json()
        return result.get("response", "")
        
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Please start Ollama with: ollama serve"
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Model may be too slow or overloaded."
    except Exception as e:
        return f"Error: {str(e)}"

def parse_json_response(response: str) -> List[Dict]:
    """Parse JSON response, handling markdown wrappers"""
    try:
        # Clean up response - remove markdown wrappers
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        cleaned_response = cleaned_response.strip()
        parsed = json.loads(cleaned_response)
        
        # Handle both direct array and nested "topics" structure
        if isinstance(parsed, dict) and "topics" in parsed:
            return parsed["topics"]
        elif isinstance(parsed, list):
            return parsed
        else:
            return []
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return []

def format_short_messages(df: pd.DataFrame, sample_size: int = 20) -> str:
    """Format a smaller sample of messages with shorter text"""
    # Take first 20 messages for testing (reduced from 50)
    sample_df = df.head(sample_size)
    
    formatted = []
    for idx, msg in sample_df.iterrows():
        thread_id = msg.get('thread_id', 'unknown')
        username = msg.get('user_name', 'unknown')
        message_text = msg.get('text', '')[:100]  # Truncate to 100 chars
        channel = msg.get('channel', 'unknown')
        
        formatted.append(f"Row {idx+1} - {username}: {message_text}")
    
    return "\n".join(formatted)

def create_simple_prompt(messages_str: str) -> str:
    """Create a simple, short prompt for local models"""
    return f"""Analyze these Slack messages and identify the main topics. Return a JSON array of topics with 'title' and 'summary' fields.

Messages:
{messages_str}

Topics:"""

def evaluate_model_ollama_optimized(model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single Gemma model via Ollama with optimized settings"""
    start_time = time.time()
    
    print(f"Testing {model_name} via Ollama (optimized)...")
    
    # Call the model with shorter timeout
    response = safe_call_ollama_gemma_optimized(model_name, prompt, timeout=60)
    
    # Parse response
    parsed_response = parse_json_response(response)
    json_valid = len(parsed_response) > 0
    
    duration = time.time() - start_time
    
    return {
        "provider": "ollama",
        "model": model_name,
        "response": response,
        "parsed_response": parsed_response,
        "json_valid": json_valid,
        "topics_found": len(parsed_response),
        "duration": duration,
        "timestamp": time.time()
    }

def load_test_data():
    """Load test data and benchmark"""
    input_csv = os.path.join(DATA_DIR, "Synthetic_Slack_Messages.csv")
    ground_truth_path = os.path.join(DATA_DIR, "benchmark_topics_corrected_fixed.json")
    
    df = pd.read_csv(input_csv)
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
    
    # Extract topics from the nested structure
    ground_truth = ground_truth_data.get("topics", [])
    
    return df, ground_truth

def check_ollama_status():
    """Check if Ollama is running and list available models"""
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✅ Ollama is running!")
            print("📋 Available models:")
            for model in models:
                print(f"  - {model['name']}")
            return True
        else:
            print("❌ Ollama is not responding properly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running!")
        print("💡 To start Ollama:")
        print("   1. Install Ollama: https://ollama.ai/")
        print("   2. Start the service: ollama serve")
        print("   3. Pull Gemma models: ollama pull gemma:2b")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def run_ollama_gemma_optimized_evaluation():
    """Run optimized evaluation with local Gemma models via Ollama"""
    print("🎯 OPTIMIZED LOCAL GEMMA EVALUATION VIA OLLAMA")
    print("=" * 50)
    
    # Check Ollama status first
    if not check_ollama_status():
        return
    
    # Load data
    df, ground_truth = load_test_data()
    print(f"📊 Loaded {len(df)} messages from CSV file")
    print(f"📋 Loaded {len(ground_truth)} benchmark topics")
    
    # Format short messages
    messages_str = format_short_messages(df, sample_size=20)
    print(f"📝 Using first 20 messages (truncated) for faster processing")
    
    # Create simple prompt
    prompt = create_simple_prompt(messages_str)
    
    # Define Gemma models to test via Ollama (start with smallest)
    ollama_gemma_models = [
        "gemma:2b",           # Smallest Gemma model (fastest)
        "gemma:2b-instruct",  # Instruct-tuned 2B model
        "gemma:7b",           # Medium Gemma model
        "gemma:7b-instruct"   # Instruct-tuned 7B model
    ]
    
    print(f"🧪 Testing {len(ollama_gemma_models)} Gemma models via Ollama...")
    
    # Test each model
    results = {}
    valid_count = 0
    
    for model_name in ollama_gemma_models:
        result = evaluate_model_ollama_optimized(model_name, prompt)
        results[f"ollama_{model_name}"] = result
        
        status = "✅" if result["json_valid"] else "❌"
        topics = result["topics_found"]
        print(f"  {status} {model_name}: {result['duration']:.2f}s, Topics: {topics}")
        
        if result["json_valid"]:
            valid_count += 1
        
        # Small delay between models
        time.sleep(3)
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "ollama_gemma_optimized_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Optimized Ollama Gemma Results Summary:")
    print(f"  Total models tested: {len(ollama_gemma_models)}")
    print(f"  Valid responses: {valid_count}")
    print(f"  Success rate: {(valid_count/len(ollama_gemma_models)*100):.1f}%")
    print(f"\n📁 Results saved to: {output_file}")
    
    # Show sample results
    print(f"\n📋 Sample Results:")
    for model_key, result in results.items():
        if result["json_valid"]:
            print(f"\n{result['model']}:")
            for i, topic in enumerate(result["parsed_response"][:2]):  # Show first 2 topics
                print(f"  Topic {i+1}: {topic.get('title', 'No title')}")
                print(f"    Summary: {topic.get('summary', 'No summary')[:80]}...")
    
    print("\n🎉 Optimized Ollama Gemma evaluation complete!")

if __name__ == "__main__":
    run_ollama_gemma_optimized_evaluation()
