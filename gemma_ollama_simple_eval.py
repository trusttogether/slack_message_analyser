#!/usr/bin/env python3
"""
Simple Local Gemma Evaluation using Ollama
Uses very simple prompts for local models
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

def safe_call_ollama_simple(model_name: str, prompt: str, timeout: int = 30) -> str:
    """Safely call Gemma model via Ollama API with very simple settings"""
    try:
        # Ollama API endpoint
        url = "http://localhost:11434/api/generate"
        
        # Prepare the request payload with minimal settings
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "max_tokens": 500,  # Very short response
                "num_predict": 500
            }
        }
        
        # Make the API call with short timeout
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        # Extract the response text
        result = response.json()
        return result.get("response", "")
        
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running."
    except requests.exceptions.Timeout:
        return "Error: Request timed out."
    except Exception as e:
        return f"Error: {str(e)}"

def parse_simple_response(response: str) -> List[Dict]:
    """Parse simple response and try to extract topics"""
    try:
        # Try to find JSON in the response
        if "[" in response and "]" in response:
            start = response.find("[")
            end = response.rfind("]") + 1
            json_str = response[start:end]
            parsed = json.loads(json_str)
            
            if isinstance(parsed, list):
                return parsed
            else:
                return []
        else:
            # If no JSON found, try to extract topics manually
            topics = []
            lines = response.split('\n')
            current_topic = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('Title:') or line.startswith('Topic:'):
                    if current_topic:
                        topics.append(current_topic)
                    current_topic = {'title': line.split(':', 1)[1].strip()}
                elif line.startswith('Summary:') and current_topic:
                    current_topic['summary'] = line.split(':', 1)[1].strip()
            
            if current_topic:
                topics.append(current_topic)
            
            return topics
            
    except Exception as e:
        print(f"Response parsing error: {e}")
        return []

def format_minimal_messages(df: pd.DataFrame, sample_size: int = 10) -> str:
    """Format a very small sample of messages"""
    # Take first 10 messages only
    sample_df = df.head(sample_size)
    
    formatted = []
    for idx, msg in sample_df.iterrows():
        username = msg.get('user_name', 'unknown')
        message_text = msg.get('text', '')[:50]  # Very short text
        
        formatted.append(f"{username}: {message_text}")
    
    return "\n".join(formatted)

def create_minimal_prompt(messages_str: str) -> str:
    """Create a very minimal prompt for local models"""
    return f"""Find topics in these messages. Return JSON array with title and summary.

Messages:
{messages_str}

Topics:"""

def evaluate_model_simple(model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single Gemma model with simple settings"""
    start_time = time.time()
    
    print(f"Testing {model_name} (simple)...")
    
    # Call the model
    response = safe_call_ollama_simple(model_name, prompt, timeout=30)
    
    # Parse response
    parsed_response = parse_simple_response(response)
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
    """Load test data"""
    input_csv = os.path.join(DATA_DIR, "Synthetic_Slack_Messages.csv")
    df = pd.read_csv(input_csv)
    return df

def run_simple_gemma_evaluation():
    """Run simple evaluation with local Gemma models"""
    print("🎯 SIMPLE LOCAL GEMMA EVALUATION")
    print("=" * 40)
    
    # Load data
    df = load_test_data()
    print(f"📊 Loaded {len(df)} messages from CSV file")
    
    # Format minimal messages
    messages_str = format_minimal_messages(df, sample_size=10)
    print(f"📝 Using first 10 messages (minimal) for testing")
    
    # Create minimal prompt
    prompt = create_minimal_prompt(messages_str)
    
    # Test only the smallest model first
    test_models = ["gemma:2b"]
    
    print(f"🧪 Testing {len(test_models)} Gemma models...")
    
    # Test each model
    results = {}
    valid_count = 0
    
    for model_name in test_models:
        result = evaluate_model_simple(model_name, prompt)
        results[f"ollama_{model_name}"] = result
        
        status = "✅" if result["json_valid"] else "❌"
        topics = result["topics_found"]
        print(f"  {status} {model_name}: {result['duration']:.2f}s, Topics: {topics}")
        
        if result["json_valid"]:
            valid_count += 1
        
        # Show the actual response for debugging
        print(f"    Response: {result['response'][:200]}...")
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "simple_gemma_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Simple Gemma Results Summary:")
    print(f"  Total models tested: {len(test_models)}")
    print(f"  Valid responses: {valid_count}")
    print(f"  Success rate: {(valid_count/len(test_models)*100):.1f}%")
    print(f"\n📁 Results saved to: {output_file}")
    
    # Show sample results
    print(f"\n📋 Sample Results:")
    for model_key, result in results.items():
        if result["json_valid"]:
            print(f"\n{result['model']}:")
            for i, topic in enumerate(result["parsed_response"][:2]):
                print(f"  Topic {i+1}: {topic.get('title', 'No title')}")
                print(f"    Summary: {topic.get('summary', 'No summary')[:60]}...")
    
    print("\n🎉 Simple Gemma evaluation complete!")

if __name__ == "__main__":
    run_simple_gemma_evaluation()
