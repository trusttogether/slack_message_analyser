import os
import json
import time
import pandas as pd
from typing import Dict, List, Any
import google.generativeai as genai
from config.config import GOOGLE_API_KEY, DATA_DIR, PROMPT_DIR, OUTPUT_DIR

# Configure Google API
genai.configure(api_key=GOOGLE_API_KEY)

# Google Gemini model configurations
GEMINI_MODELS = {
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-pro": "gemini-1.5-pro"
}

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

def call_gemini_model(model_name: str, prompt: str) -> str:
    """Call Google Gemini model"""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_gemini_model(model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single Gemini model"""
    print(f"Testing {model_name}...")
    start_time = time.time()
    
    response = call_gemini_model(model_name, prompt)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Try to parse JSON response
    try:
        # Handle markdown-wrapped JSON (common with Gemini)
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        parsed_response = json.loads(response)
        json_valid = True
    except:
        parsed_response = response
        json_valid = False
    
    return {
        "provider": "google",
        "model": model_name,
        "response": response,
        "parsed_response": parsed_response,
        "json_valid": json_valid,
        "duration": duration,
        "timestamp": time.time()
    }

def run_gemini_evaluation():
    """Run evaluation on all Gemini models"""
    print("Starting Google Gemini model evaluation...")
    
    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not set. Please set the environment variable.")
        return
    
    # Load data
    df, ground_truth = load_test_data()
    messages_str = format_messages(df)
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Test all Gemini models
    results = {}
    
    for model_key, model_name in GEMINI_MODELS.items():
        print(f"\n=== Testing {model_key.upper()} ===")
        
        result = evaluate_gemini_model(model_name, prompt)
        results[model_key] = result
        
        # Save individual result
        output_path = os.path.join(OUTPUT_DIR, f"output_google_{model_key}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"  ✓ {model_key}: {result['duration']:.2f}s, JSON valid: {result['json_valid']}")
    
    # Save comprehensive results
    gemini_output = os.path.join(OUTPUT_DIR, "google_gemini_evaluation.json")
    with open(gemini_output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    generate_gemini_report(results)
    
    print(f"\n✅ Gemini evaluation complete!")
    print(f"Results saved to: {gemini_output}")

def generate_gemini_report(results: Dict):
    """Generate Gemini evaluation report"""
    report = {
        "summary": {
            "total_models": len(results),
            "provider": "google",
            "json_valid_count": sum(1 for r in results.values() if r["json_valid"]),
            "average_duration": sum(r["duration"] for r in results.values()) / len(results) if results else 0
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
    report_path = os.path.join(OUTPUT_DIR, "google_gemini_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Gemini evaluation report saved to: {report_path}")

if __name__ == "__main__":
    run_gemini_evaluation()
