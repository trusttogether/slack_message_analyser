#!/usr/bin/env python3
"""
STEP 1: Simple Test - Evaluate models with smaller message subset
Tests if models can handle basic topic clustering before full evaluation
"""

import json
import time
import os
import pandas as pd
from typing import Dict, Any, List

# Import configuration
from config.config import (
    GOOGLE_API_KEY, DATA_DIR, PROMPT_DIR
)

def safe_call_google_gemma(model_name: str, prompt: str) -> str:
    """Safely call Google Gemma model via API"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
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
        print(f"Response: {response[:200]}...")
        return []

def format_sample_messages(df: pd.DataFrame, sample_size: int = 50) -> str:
    """Format a sample of messages for testing"""
    # Take first 50 messages for testing
    sample_df = df.head(sample_size)
    
    formatted = []
    for idx, msg in sample_df.iterrows():
        thread_id = msg.get('thread_id', 'unknown')
        username = msg.get('user_name', 'unknown')
        message_text = msg.get('text', '')
        channel = msg.get('channel', 'unknown')
        
        formatted.append(f"Row {idx+1} - Channel {channel} - Thread {thread_id}: {username}: {message_text}")
    
    return "\n".join(formatted)

def evaluate_model_simple(model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single model with simple test"""
    start_time = time.time()
    
    print(f"Testing {model_name} with simple test...")
    
    # Call the model
    response = safe_call_google_gemma(model_name, prompt)
    
    # Parse response
    parsed_response = parse_json_response(response)
    json_valid = len(parsed_response) > 0
    
    duration = time.time() - start_time
    
    return {
        "provider": "google",
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

def run_simple_test():
    """Run simple test with smaller message subset"""
    print("🎯 STEP 1: SIMPLE MODEL TEST")
    print("=" * 40)
    
    if not GOOGLE_API_KEY:
        print("❌ No Google API key found! Please set GOOGLE_API_KEY in .env file")
        return
    
    # Load data
    df, ground_truth = load_test_data()
    print(f"📊 Loaded {len(df)} messages from CSV file")
    print(f"📋 Loaded {len(ground_truth)} benchmark topics")
    
    # Format sample messages (first 50)
    messages_str = format_sample_messages(df, sample_size=50)
    print(f"📝 Using first 50 messages for testing")
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Define models to test
    test_models = [
        "gemma-3-1b-it",
        "gemma-3-4b-it", 
        "gemma-3-12b-it",
        "gemma-3-27b-it"
    ]
    
    print(f"🧪 Testing {len(test_models)} models with simple approach...")
    
    # Test each model
    results = {}
    valid_count = 0
    
    for model_name in test_models:
        result = evaluate_model_simple(model_name, prompt)
        results[f"google_{model_name}"] = result
        
        status = "✅" if result["json_valid"] else "❌"
        topics = result["topics_found"]
        print(f"  {status} {model_name}: {result['duration']:.2f}s, Topics: {topics}")
        
        if result["json_valid"]:
            valid_count += 1
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "step1_simple_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Simple Test Results Summary:")
    print(f"  Total models tested: {len(test_models)}")
    print(f"  Valid responses: {valid_count}")
    print(f"  Success rate: {(valid_count/len(test_models)*100):.1f}%")
    print(f"\n📁 Results saved to: {output_file}")
    
    # Show sample results
    print(f"\n📋 Sample Results:")
    for model_key, result in results.items():
        if result["json_valid"]:
            print(f"\n{result['model']}:")
            for i, topic in enumerate(result["parsed_response"][:2]):  # Show first 2 topics
                print(f"  Topic {i+1}: {topic.get('title', 'No title')}")
                print(f"    Summary: {topic.get('summary', 'No summary')[:100]}...")
    
    print("\n🎉 Simple test complete!")

if __name__ == "__main__":
    run_simple_test()
