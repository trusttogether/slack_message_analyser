import os
import json
import time
import pandas as pd
from typing import Dict, List, Any
import openai
import google.generativeai as genai
from groq import Groq
import anthropic
from config.config import (
    OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY,
    DATA_DIR, PROMPT_DIR, OUTPUT_DIR
)

# Configure API keys
openai.api_key = OPENAI_API_KEY

# Model configurations
MODELS = {
    "openai": {
        "gpt-4": {"provider": "openai", "model": "gpt-4"},
        "gpt-3.5-turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "gpt-4o": {"provider": "openai", "model": "gpt-4o"}
    },
    "google": {
        "gemini-pro": {"provider": "google", "model": "gemini-1.5-flash"},
        "gemini-1.5-pro": {"provider": "google", "model": "gemini-1.5-pro"}
    },
    "groq": {
        "llama3-8b": {"provider": "groq", "model": "llama3-8b-8192"},
        "llama3-70b": {"provider": "groq", "model": "llama3-70b-8192"},
        "mixtral-8x7b": {"provider": "groq", "model": "mixtral-8x7b-32768"}
    },
    "anthropic": {
        "claude-3-opus": {"provider": "anthropic", "model": "claude-3-opus-20240229"},
        "claude-3-sonnet": {"provider": "anthropic", "model": "claude-3-sonnet-20240229"},
        "claude-3-haiku": {"provider": "anthropic", "model": "claude-3-haiku-20240307"}
    }
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

def call_openai_model(model_name: str, prompt: str) -> str:
    """Call OpenAI model"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def call_google_model(model_name: str, prompt: str) -> str:
    """Call Google Gemini model"""
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def call_groq_model(model_name: str, prompt: str) -> str:
    """Call Groq model"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def call_anthropic_model(model_name: str, prompt: str) -> str:
    """Call Anthropic Claude model"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model_name,
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_model(model_config: Dict, prompt: str) -> Dict[str, Any]:
    """Evaluate a single model"""
    provider = model_config["provider"]
    model_name = model_config["model"]
    
    print(f"Testing {provider}/{model_name}...")
    start_time = time.time()
    
    # Call appropriate model
    if provider == "openai":
        response = call_openai_model(model_name, prompt)
    elif provider == "google":
        response = call_google_model(model_name, prompt)
    elif provider == "groq":
        response = call_groq_model(model_name, prompt)
    elif provider == "anthropic":
        response = call_anthropic_model(model_name, prompt)
    else:
        response = f"Error: Unknown provider {provider}"
    
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
        "provider": provider,
        "model": model_name,
        "response": response,
        "parsed_response": parsed_response,
        "json_valid": json_valid,
        "duration": duration,
        "timestamp": time.time()
    }

def run_comprehensive_evaluation():
    """Run evaluation on all models"""
    print("Starting comprehensive model evaluation...")
    
    # Load data
    df, ground_truth = load_test_data()
    messages_str = format_messages(df)
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Test all models
    results = {}
    
    for provider, models in MODELS.items():
        print(f"\n=== Testing {provider.upper()} Models ===")
        for model_name, model_config in models.items():
            result = evaluate_model(model_config, prompt)
            results[f"{provider}_{model_name}"] = result
            
            # Save individual result
            output_path = os.path.join(OUTPUT_DIR, f"output_{provider}_{model_name}.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"  ✓ {model_name}: {result['duration']:.2f}s, JSON valid: {result['json_valid']}")
    
    # Save comprehensive results
    comprehensive_output = os.path.join(OUTPUT_DIR, "comprehensive_model_evaluation.json")
    with open(comprehensive_output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    generate_evaluation_report(results, ground_truth)
    
    print(f"\n✅ Comprehensive evaluation complete!")
    print(f"Results saved to: {comprehensive_output}")

def generate_evaluation_report(results: Dict, ground_truth: Dict):
    """Generate evaluation report"""
    report = {
        "summary": {
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
    report_path = os.path.join(OUTPUT_DIR, "model_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Evaluation report saved to: {report_path}")

if __name__ == "__main__":
    run_comprehensive_evaluation()
