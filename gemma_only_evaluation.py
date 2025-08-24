#!/usr/bin/env python3
"""
Gemma Models Only Evaluation - Fast and Simple
Tests only Google Gemma models via API (no local loading)
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
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return []

def evaluate_gemma_model(model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single Gemma model"""
    start_time = time.time()
    
    print(f"Testing {model_name}...")
    
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
        "duration": duration,
        "timestamp": time.time()
    }

def load_test_data():
    """Load test data for evaluation"""
    input_csv = os.path.join(DATA_DIR, "Synthetic_Slack_Messages.csv")
    ground_truth_path = os.path.join(DATA_DIR, "benchmark_topics_corrected_fixed.json")
    
    df = pd.read_csv(input_csv)
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    return df, ground_truth

def format_messages(df: pd.DataFrame) -> str:
    """Format messages for prompt"""
    # Process ALL messages (not limited to 50)
    messages = df.to_dict('records')
    
    formatted = []
    for msg in messages:
        # Use full message text (not truncated to 200 chars)
        thread_id = msg.get('thread_id', 'unknown')
        username = msg.get('user_name', 'unknown')  # Note: column name is 'user_name', not 'username'
        message_text = msg.get('text', '')
        
        formatted.append(f"Thread {thread_id}: {username}: {message_text}")
    
    return "\n".join(formatted)

def run_gemma_evaluation():
    """Run evaluation with only Gemma models"""
    print("🎯 GEMMA MODELS EVALUATION")
    print("=" * 50)
    
    if not GOOGLE_API_KEY:
        print("❌ No Google API key found! Please set GOOGLE_API_KEY in .env file")
        return
    
    # Load data
    df, ground_truth = load_test_data()
    print(f"📊 Loaded {len(df)} messages from CSV file")
    
    messages_str = format_messages(df)
    print(f"📝 Formatted {len(messages_str.split('Thread'))} message entries")
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Define Gemma models to test
    gemma_models = [
        "gemma-3-1b-it",
        "gemma-3-4b-it", 
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "gemma-3n-e4b-it",
        "gemma-3n-e2b-it"
    ]
    
    print(f"🧪 Testing {len(gemma_models)} Gemma models...")
    
    # Test each model
    results = {}
    valid_count = 0
    
    for model_name in gemma_models:
        result = evaluate_gemma_model(model_name, prompt)
        results[f"google_{model_name}"] = result
        
        status = "✅" if result["json_valid"] else "❌"
        print(f"  {status} {model_name}: {result['duration']:.2f}s, JSON valid: {result['json_valid']}")
        
        if result["json_valid"]:
            valid_count += 1
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "gemma_evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Gemma Results Summary:")
    print(f"  Total models tested: {len(gemma_models)}")
    print(f"  Valid JSON responses: {valid_count}")
    print(f"  Success rate: {(valid_count/len(gemma_models)*100):.1f}%")
    print(f"\n📁 Results saved to: {output_file}")
    
    print("\n🎉 Gemma evaluation complete!")
    print("Models with ✅ are working properly.")
    print("Models with ❌ had JSON parsing issues.")

if __name__ == "__main__":
    run_gemma_evaluation()
