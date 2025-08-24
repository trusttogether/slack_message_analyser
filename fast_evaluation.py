#!/usr/bin/env python3
"""
Fast evaluation using only Google API models (including Gemma)
No slow local model loading!
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

def safe_call_google(model_name: str, prompt: str) -> str:
    """Safely call Google model (including Gemma models)"""
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
        # Clean up response
        cleaned = response.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        # Parse JSON
        return json.loads(cleaned)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return []

def evaluate_model(model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single model"""
    start_time = time.time()
    
    print(f"Testing {model_name}...")
    
    # Call model
    response = safe_call_google(model_name, prompt)
    
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
    # Take first 50 messages to keep prompt manageable
    messages = df.head(50)
    
    formatted = []
    for _, row in messages.iterrows():
        formatted.append(f"Thread {row['thread_id']}: {row['username']}: {row['message'][:200]}...")
    
    return "\n".join(formatted)

def run_fast_evaluation():
    """Run fast evaluation with Google models only"""
    print("🚀 FAST Google Models Evaluation")
    print("=" * 50)
    
    if not GOOGLE_API_KEY:
        print("❌ No Google API key found! Please set GOOGLE_API_KEY in your .env file")
        return
    
    # Load data
    df, ground_truth = load_test_data()
    messages_str = format_messages(df)
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Define Google models to test (including Gemma)
    google_models = [
        "gemini-1.5-flash",
        "gemini-1.5-pro", 
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-002",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemma-3-1b-it",
        "gemma-3-4b-it", 
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "gemma-3n-e4b-it",
        "gemma-3n-e2b-it"
    ]
    
    print(f"🧪 Testing {len(google_models)} Google models (including Gemma)...")
    
    # Test each model
    results = {}
    valid_count = 0
    
    for model_name in google_models:
        result = evaluate_model(model_name, prompt)
        results[f"google_{model_name}"] = result
        
        status = "✅" if result["json_valid"] else "❌"
        print(f"  {status} {model_name}: {result['duration']:.2f}s, JSON valid: {result['json_valid']}")
        
        if result["json_valid"]:
            valid_count += 1
    
    # Save results
    output_file = "fast_google_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Results Summary:")
    print(f"  Total models tested: {len(google_models)}")
    print(f"  Valid JSON responses: {valid_count}")
    print(f"  Success rate: {(valid_count/len(google_models)*100):.1f}%")
    print(f"\n📁 Results saved to: {output_file}")
    
    print("\n🎉 Fast evaluation complete!")
    print("✅ No slow local model loading!")
    print("✅ All models via fast Google API!")

if __name__ == "__main__":
    run_fast_evaluation()
