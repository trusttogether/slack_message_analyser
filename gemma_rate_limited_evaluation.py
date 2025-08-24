#!/usr/bin/env python3
"""
Rate-Limited Gemma Evaluation - Respects API Quotas
Includes retry logic and delays between requests
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

def safe_call_google_gemma_with_retry(model_name: str, prompt: str, max_retries: int = 3) -> str:
    """Safely call Google Gemma model with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "quota" in error_str.lower():
                # Extract retry delay from error message
                retry_delay = 60  # Default 60 seconds
                if "retry_delay" in error_str:
                    try:
                        # Extract seconds from retry_delay
                        import re
                        match = re.search(r'seconds: (\d+)', error_str)
                        if match:
                            retry_delay = int(match.group(1))
                    except:
                        pass
                
                print(f"  ⏳ Rate limit hit for {model_name}. Waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                return f"Error: {str(e)}"
    
    return f"Error: Max retries exceeded for {model_name}"

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

def format_sample_messages(df: pd.DataFrame, sample_size: int = 30) -> str:
    """Format a smaller sample of messages to avoid rate limits"""
    # Take first 30 messages for testing (reduced from 50)
    sample_df = df.head(sample_size)
    
    formatted = []
    for idx, msg in sample_df.iterrows():
        thread_id = msg.get('thread_id', 'unknown')
        username = msg.get('user_name', 'unknown')
        message_text = msg.get('text', '')[:150]  # Truncate to 150 chars
        channel = msg.get('channel', 'unknown')
        
        formatted.append(f"Row {idx+1} - Channel {channel} - Thread {thread_id}: {username}: {message_text}")
    
    return "\n".join(formatted)

def evaluate_model_rate_limited(model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single model with rate limiting"""
    start_time = time.time()
    
    print(f"Testing {model_name} with rate limiting...")
    
    # Call the model with retry logic
    response = safe_call_google_gemma_with_retry(model_name, prompt)
    
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

def run_rate_limited_evaluation():
    """Run evaluation with rate limiting"""
    print("🎯 RATE-LIMITED GEMMA EVALUATION")
    print("=" * 45)
    
    if not GOOGLE_API_KEY:
        print("❌ No Google API key found! Please set GOOGLE_API_KEY in .env file")
        return
    
    # Load data
    df, ground_truth = load_test_data()
    print(f"📊 Loaded {len(df)} messages from CSV file")
    print(f"📋 Loaded {len(ground_truth)} benchmark topics")
    
    # Format sample messages (reduced size)
    messages_str = format_sample_messages(df, sample_size=30)
    print(f"📝 Using first 30 messages (truncated) to avoid rate limits")
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Define models to test (start with smaller models)
    test_models = [
        "gemma-3n-e2b-it",  # Smallest model first
        "gemma-3-1b-it",
        "gemma-3-4b-it"
    ]
    
    print(f"🧪 Testing {len(test_models)} models with rate limiting...")
    
    # Test each model with delays
    results = {}
    valid_count = 0
    
    for i, model_name in enumerate(test_models):
        result = evaluate_model_rate_limited(model_name, prompt)
        results[f"google_{model_name}"] = result
        
        status = "✅" if result["json_valid"] else "❌"
        topics = result["topics_found"]
        print(f"  {status} {model_name}: {result['duration']:.2f}s, Topics: {topics}")
        
        if result["json_valid"]:
            valid_count += 1
        
        # Add delay between models to respect rate limits
        if i < len(test_models) - 1:  # Don't delay after last model
            print(f"  ⏳ Waiting 30 seconds before next model...")
            time.sleep(30)
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "rate_limited_gemma_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Rate-Limited Results Summary:")
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
    
    print("\n🎉 Rate-limited evaluation complete!")

if __name__ == "__main__":
    run_rate_limited_evaluation()
