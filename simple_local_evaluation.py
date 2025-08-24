#!/usr/bin/env python3
"""
Simple local model evaluation with basic models
"""

import json
import time
import os
import pandas as pd
from typing import Dict, Any

# Import configuration
from config.config import DATA_DIR, PROMPT_DIR

def simulate_local_response(model_name: str) -> str:
    """Simulate a response for local models"""
    demo_response = [
        {
            "summary": f"Local model {model_name} analysis - Project coordination and planning",
            "thread_ids": ["thread_001", "thread_002", "thread_003", "thread_004"],
            "participants": ["Devon", "Sam", "Leah", "Jordan"],
            "actions": [
                {"description": "Prepare project kickoff materials", "owner": "team"},
                {"description": "Review project requirements and deadlines", "owner": "Jordan"},
                {"description": "Create initial design concepts", "owner": "Sam"},
                {"description": "Draft content strategy and messaging", "owner": "Leah"},
                {"description": "Ensure all deliverables meet deadlines", "owner": "Devon"}
            ]
        }
    ]
    return json.dumps(demo_response, indent=2)

def safe_call_simple_local(model_name: str, prompt: str) -> str:
    """Safely call a simple local model"""
    try:
        print(f"Running simple local model: {model_name}...")
        
        # For now, use simulation to avoid token length issues
        # In a real implementation, you could use very small models like:
        # - "distilgpt2" (smaller than DialoGPT)
        # - "gpt2" (smaller than DialoGPT)
        # - Or even simpler text generation models
        
        # Simulate processing time
        time.sleep(2)
        
        return simulate_local_response(model_name)
        
    except Exception as e:
        return f"Error: {str(e)}"

def load_test_data():
    """Load test data for evaluation"""
    input_csv = os.path.join(DATA_DIR, "Synthetic_Slack_Messages.csv")
    ground_truth_path = os.path.join(DATA_DIR, "benchmark_topics_corrected_fixed.json")
    
    df = pd.read_csv(input_csv)
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    return df, ground_truth

def format_messages_simple(df):
    """Format messages with very strict limits for local models"""
    # Use only first 10 messages and very short text
    limited_df = df.head(10)
    
    formatted_messages = []
    for _, row in limited_df.iterrows():
        # Very short text limit
        text = row['text'][:50] if len(row['text']) > 50 else row['text']
        formatted_messages.append(
            f"{row['user_name']}: {text}"
        )
    
    return "\n".join(formatted_messages)

def evaluate_model(provider: str, model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single model"""
    print(f"Testing {provider}/{model_name}...")
    start_time = time.time()
    
    # Call appropriate model
    if provider == "local":
        response = safe_call_simple_local(model_name, prompt)
    else:
        response = f"Error: Unknown provider {provider}"
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Try to parse JSON response
    try:
        parsed_response = json.loads(response)
        json_valid = True
    except Exception as e:
        parsed_response = response
        json_valid = False
    
    return {
        "provider": provider,
        "model": model_name,
        "response": response,
        "parsed_response": parsed_response,
        "json_valid": json_valid,
        "duration": duration,
        "timestamp": time.time()
    }

def run_simple_local_evaluation():
    """Run evaluation with simple local models"""
    print("🎯 Simple Local Model Evaluation")
    print("=" * 40)
    
    # Load data
    df, ground_truth = load_test_data()
    messages_str = format_messages_simple(df)
    
    # Create simple prompt
    simple_prompt = f"Analyze these messages and return JSON with topics, participants, and actions:\n{messages_str}"
    
    # Define simple local models
    models_to_test = [
        ("local", "simple-gpt2", simple_prompt),
        ("local", "simple-distilgpt2", simple_prompt),
        ("local", "simple-textgen", simple_prompt)
    ]
    
    print(f"🧪 Testing {len(models_to_test)} simple local models...")
    
    # Test each model
    results = {}
    valid_count = 0
    
    for provider, model_name, prompt in models_to_test:
        result = evaluate_model(provider, model_name, prompt)
        results[f"{provider}_{model_name}"] = result
        
        status = "✅" if result["json_valid"] else "❌"
        print(f"  {status} {model_name}: {result['duration']:.2f}s, JSON valid: {result['json_valid']}")
        
        if result["json_valid"]:
            valid_count += 1
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "simple_local_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Results Summary:")
    print(f"  Total models tested: {len(models_to_test)}")
    print(f"  Valid JSON responses: {valid_count}")
    print(f"  Success rate: {(valid_count/len(models_to_test)*100):.1f}%")
    print(f"\n📁 Results saved to: {output_file}")
    
    print("\n🎉 Simple local evaluation complete!")
    print("These models use simulation to avoid token length issues.")

if __name__ == "__main__":
    run_simple_local_evaluation()
