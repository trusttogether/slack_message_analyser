#!/usr/bin/env python3
"""
Dedicated Gemma model evaluation script.
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

def format_messages_for_gemma(df):
    """Format messages for Gemma model input"""
    # Use a smaller subset for faster inference
    limited_df = df.head(30)
    
    formatted_messages = []
    for _, row in limited_df.iterrows():
        # Truncate long messages
        text = row['text'][:150] if len(row['text']) > 150 else row['text']
        formatted_messages.append(
            f"[{row['channel']}] {row['user_name']}: {text} (thread_id={row['thread_id']})"
        )
    
    return "\n".join(formatted_messages)

def call_gemma_model(model_name: str, prompt: str) -> str:
    """Call Gemma model with proper formatting"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print(f"Loading Gemma model: {model_name}...")
        
        # Map model names to HuggingFace model IDs
        model_mapping = {
            "gemma-2b": "google/gemma-2b",
            "gemma-7b": "google/gemma-7b",
            "gemma-2b-it": "google/gemma-2b-it",
            "gemma-7b-it": "google/gemma-7b-it"
        }
        
        hf_model_name = model_mapping.get(model_name, model_name)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Format prompt for Gemma
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize and generate
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,  # Reduced for faster inference
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response (remove the input prompt)
        response = response.replace(formatted_prompt, "").strip()
        
        return response
        
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_gemma_model(model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single Gemma model"""
    print(f"Testing Gemma model: {model_name}...")
    start_time = time.time()
    
    response = call_gemma_model(model_name, prompt)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Try to parse JSON response
    try:
        # Handle markdown-wrapped JSON
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        # Additional cleaning
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

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
    """Run Gemma model evaluation"""
    print("🎯 DEEMERGE Gemma Model Evaluation")
    print("=" * 50)
    
    # Load data
    df, ground_truth = load_test_data()
    messages_str = format_messages_for_gemma(df)
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Define Gemma models to test
    gemma_models = [
        "gemma-2b",
        "gemma-7b",
        "gemma-2b-it",
        "gemma-7b-it"
    ]
    
    print(f"🧪 Testing {len(gemma_models)} Gemma models...")
    
    # Test models
    results = {}
    
    for model_name in gemma_models:
        result = evaluate_gemma_model(model_name, prompt)
        results[f"gemma_{model_name}"] = result
        
        # Save individual result
        output_path = os.path.join(OUTPUT_DIR, f"gemma_output_{model_name.replace('-', '_')}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        status = "✅" if result["json_valid"] else "❌"
        print(f"  {status} {model_name}: {result['duration']:.2f}s, JSON valid: {result['json_valid']}")
    
    # Save comprehensive results
    gemma_output = os.path.join(OUTPUT_DIR, "gemma_evaluation_results.json")
    with open(gemma_output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    valid_count = sum(1 for r in results.values() if r["json_valid"])
    total_count = len(results)
    
    print(f"\n📊 Gemma Results Summary:")
    print(f"  Total models tested: {total_count}")
    print(f"  Valid JSON responses: {valid_count}")
    print(f"  Success rate: {valid_count/total_count*100:.1f}%")
    print(f"\n📁 Results saved to: {gemma_output}")
    
    if valid_count > 0:
        print("\n🎉 Gemma evaluation complete!")
        print("You can now analyze the results and compare with other models.")
    else:
        print("\n⚠️  No Gemma models returned valid JSON.")
        print("This might be due to model loading issues or response formatting.")

if __name__ == "__main__":
    run_gemma_evaluation()
