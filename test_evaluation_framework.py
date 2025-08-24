#!/usr/bin/env python3
"""
Test script to demonstrate the DEEMERGE evaluation framework
without requiring API keys. This shows how the system works.
"""

import os
import json
import time
import pandas as pd
from typing import Dict, List, Any
from config.config import DATA_DIR, PROMPT_DIR, OUTPUT_DIR

def load_test_data():
    """Load test data for evaluation"""
    input_csv = os.path.join(DATA_DIR, "Synthetic_Slack_Messages.csv")
    ground_truth_path = os.path.join(DATA_DIR, "benchmark_topics_corrected_fixed.json")
    
    df = pd.read_csv(input_csv)
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    return df, ground_truth

def format_messages(df):
    """Format messages for LLM input"""
    return "\n".join([
        f"[{row['channel']}] {row['user_name']} ({row['timestamp']}): {row['text']} (thread_id={row['thread_id']})"
        for _, row in df.iterrows()
    ])

def simulate_model_response(model_name: str, prompt: str) -> str:
    """Simulate a model response for testing purposes"""
    # This simulates what different models would return
    simulated_responses = {
        "gpt-4": {
            "topics": [
                {
                    "summary": "EcoBloom Summer Campaign Planning",
                    "thread_ids": [1, 2, 3, 4, 5],
                    "participants": ["@Devon", "@Sam", "@Leah"],
                    "actions": [
                        {"description": "Share initial designs", "owner": "@Sam"},
                        {"description": "Complete content strategy", "owner": "@Leah"}
                    ]
                },
                {
                    "summary": "FitFusion Rebranding Project",
                    "thread_ids": [14, 15, 16, 17, 18],
                    "participants": ["@Devon", "@Sam", "@Jordan"],
                    "actions": [
                        {"description": "Develop brand messaging", "owner": "@Sam"},
                        {"description": "Legal review", "owner": "@Jordan"}
                    ]
                }
            ]
        },
        "gemini-pro": {
            "topics": [
                {
                    "summary": "Campaign Planning and Timeline Discussion",
                    "thread_ids": [1, 2, 3, 4, 5, 6],
                    "participants": ["@Devon", "@Sam", "@Leah", "@Jordan"],
                    "actions": [
                        {"description": "Design review", "owner": "@Sam"},
                        {"description": "Content creation", "owner": "@Leah"}
                    ]
                }
            ]
        },
        "claude-3": {
            "topics": [
                {
                    "summary": "EcoBloom Summer Campaign with July 28 deadline",
                    "thread_ids": [1, 2, 3, 4, 5],
                    "participants": ["@Devon", "@Sam", "@Leah"],
                    "actions": [
                        {"description": "Design concepts", "owner": "@Sam"},
                        {"description": "Content strategy", "owner": "@Leah"},
                        {"description": "Legal review", "owner": "@Jordan"}
                    ]
                },
                {
                    "summary": "FitFusion rebranding project",
                    "thread_ids": [14, 15, 16, 17, 18, 19],
                    "participants": ["@Devon", "@Sam", "@Jordan"],
                    "actions": [
                        {"description": "Brand messaging", "owner": "@Sam"},
                        {"description": "Legal compliance", "owner": "@Jordan"}
                    ]
                }
            ]
        }
    }
    
    # Return simulated response based on model
    for model_key in simulated_responses:
        if model_key in model_name.lower():
            return json.dumps(simulated_responses[model_key])
    
    # Default response
    return json.dumps({
        "topics": [
            {
                "summary": "General project discussion",
                "thread_ids": [1, 2, 3],
                "participants": ["@Devon", "@Sam"],
                "actions": [
                    {"description": "Follow up on tasks", "owner": "@Devon"}
                ]
            }
        ]
    })

def evaluate_model(model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single model (simulated)"""
    print(f"Testing {model_name}...")
    start_time = time.time()
    
    # Simulate API call delay
    time.sleep(0.1)
    
    response = simulate_model_response(model_name, prompt)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Try to parse JSON response
    try:
        parsed_response = json.loads(response)
        json_valid = True
    except:
        parsed_response = response
        json_valid = False
    
    return {
        "provider": model_name.split("-")[0] if "-" in model_name else "unknown",
        "model": model_name,
        "response": response,
        "parsed_response": parsed_response,
        "json_valid": json_valid,
        "duration": duration,
        "timestamp": time.time()
    }

def run_demo_evaluation():
    """Run demonstration evaluation with simulated models"""
    print("🎯 DEEMERGE Evaluation Framework Demo")
    print("=" * 50)
    print("This demo shows how the evaluation framework works")
    print("without requiring actual API keys.")
    print()
    
    # Load data
    df, ground_truth = load_test_data()
    messages_str = format_messages(df)
    
    print(f"📊 Loaded {len(df)} Slack messages")
    print(f"📋 Ground truth contains {len(ground_truth.get('topics', []))} topics")
    print()
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Test simulated models
    models = ["gpt-4", "gemini-pro", "claude-3", "llama3-8b", "gemma-7b"]
    results = {}
    
    print("🧪 Testing Simulated Models:")
    print("-" * 30)
    
    for model_name in models:
        result = evaluate_model(model_name, prompt)
        results[model_name] = result
        
        print(f"  ✓ {model_name}: {result['duration']:.3f}s, JSON valid: {result['json_valid']}")
        
        # Save individual result
        output_path = os.path.join(OUTPUT_DIR, f"demo_output_{model_name.replace('-', '_')}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    
    # Save comprehensive results
    demo_output = os.path.join(OUTPUT_DIR, "demo_evaluation_results.json")
    with open(demo_output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    generate_demo_report(results, ground_truth)
    
    print(f"\n✅ Demo evaluation complete!")
    print(f"📁 Results saved to: {demo_output}")
    print()
    print("🎉 This demonstrates the evaluation framework!")
    print("   To test real models, set your API keys and run:")
    print("   python model_evaluation.py")

def generate_demo_report(results: Dict, ground_truth: Dict):
    """Generate demo evaluation report"""
    report = {
        "demo_info": {
            "description": "DEEMERGE Evaluation Framework Demo",
            "note": "This is a simulation - no real API calls were made",
            "total_models": len(results),
            "providers": list(set(r["provider"] for r in results.values())),
            "json_valid_count": sum(1 for r in results.values() if r["json_valid"]),
            "average_duration": sum(r["duration"] for r in results.values()) / len(results)
        },
        "model_performance": {}
    }
    
    for model_key, result in results.items():
        report["model_performance"][model_key] = {
            "provider": result["provider"],
            "model": result["model"],
            "json_valid": result["json_valid"],
            "duration": result["duration"],
            "status": "success" if result["json_valid"] else "failed"
        }
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "demo_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Demo report saved to: {report_path}")

if __name__ == "__main__":
    run_demo_evaluation()
