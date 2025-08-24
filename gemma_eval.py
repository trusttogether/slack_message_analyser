import os
import json
import time
import pandas as pd
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.config import DATA_DIR, PROMPT_DIR, OUTPUT_DIR

# Gemma model configurations
GEMMA_MODELS = {
    "gemma-2b": "google/gemma-2b",
    "gemma-7b": "google/gemma-7b",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-7b-it": "google/gemma-7b-it"
}

def load_gemma_model(model_name: str):
    """Load Gemma model and tokenizer"""
    try:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

def call_gemma_model(model, tokenizer, prompt: str, max_length: int = 2048) -> str:
    """Call Gemma model with prompt"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (after the prompt)
        response = response[len(prompt):].strip()
        return response
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

def format_messages(df):
    """Format messages for LLM input"""
    return "\n".join([
        f"[{row['channel']}] {row['user_name']} ({row['timestamp']}): {row['text']} (thread_id={row['thread_id']})"
        for _, row in df.iterrows()
    ])

def evaluate_gemma_model(model_name: str, model, tokenizer, prompt: str) -> Dict[str, Any]:
    """Evaluate a single Gemma model"""
    print(f"Testing {model_name}...")
    start_time = time.time()
    
    response = call_gemma_model(model, tokenizer, prompt)
    
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
        "provider": "gemma",
        "model": model_name,
        "response": response,
        "parsed_response": parsed_response,
        "json_valid": json_valid,
        "duration": duration,
        "timestamp": time.time()
    }

def run_gemma_evaluation():
    """Run evaluation on all Gemma models"""
    print("Starting Gemma model evaluation...")
    
    # Load data
    df, ground_truth = load_test_data()
    messages_str = format_messages(df)
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Test all Gemma models
    results = {}
    
    for model_key, model_path in GEMMA_MODELS.items():
        print(f"\n=== Testing {model_key.upper()} ===")
        
        model, tokenizer = load_gemma_model(model_path)
        if model is not None and tokenizer is not None:
            result = evaluate_gemma_model(model_key, model, tokenizer, prompt)
            results[model_key] = result
            
            # Save individual result
            output_path = os.path.join(OUTPUT_DIR, f"output_gemma_{model_key}.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"  ✓ {model_key}: {result['duration']:.2f}s, JSON valid: {result['json_valid']}")
        else:
            print(f"  ✗ {model_key}: Failed to load model")
    
    # Save comprehensive results
    gemma_output = os.path.join(OUTPUT_DIR, "gemma_model_evaluation.json")
    with open(gemma_output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    generate_gemma_report(results)
    
    print(f"\n✅ Gemma evaluation complete!")
    print(f"Results saved to: {gemma_output}")

def generate_gemma_report(results: Dict):
    """Generate Gemma evaluation report"""
    report = {
        "summary": {
            "total_models": len(results),
            "provider": "gemma",
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
    report_path = os.path.join(OUTPUT_DIR, "gemma_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Gemma evaluation report saved to: {report_path}")

if __name__ == "__main__":
    run_gemma_evaluation()
