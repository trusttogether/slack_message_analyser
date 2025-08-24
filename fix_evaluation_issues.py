#!/usr/bin/env python3
"""
Script to fix evaluation issues and provide a working solution.
"""

import os
import subprocess
import sys

def install_missing_packages():
    """Install missing packages"""
    print("📦 Installing missing packages...")
    
    packages = [
        "groq",
        "anthropic", 
        "openai>=1.0.0"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def create_working_evaluation_script():
    """Create a working evaluation script that handles errors gracefully"""
    
    script_content = '''#!/usr/bin/env python3
"""
Working evaluation script that handles API issues gracefully.
"""

import os
import json
import time
import pandas as pd
from typing import Dict, List, Any
from config.config import (
    OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY,
    DATA_DIR, PROMPT_DIR, OUTPUT_DIR
)

def safe_call_openai(model_name: str, prompt: str) -> str:
    """Safely call OpenAI model"""
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

def safe_call_google(model_name: str, prompt: str) -> str:
    """Safely call Google model"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def safe_call_groq(model_name: str, prompt: str) -> str:
    """Safely call Groq model"""
    try:
        from groq import Groq
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

def safe_call_anthropic(model_name: str, prompt: str) -> str:
    """Safely call Anthropic model"""
    try:
        import anthropic
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
    return "\\n".join([
        f"[{row['channel']}] {row['user_name']} ({row['timestamp']}): {row['text']} (thread_id={row['thread_id']})"
        for _, row in df.iterrows()
    ])

def evaluate_model(provider: str, model_name: str, prompt: str) -> Dict[str, Any]:
    """Evaluate a single model with error handling"""
    print(f"Testing {provider}/{model_name}...")
    start_time = time.time()
    
    # Call appropriate model
    if provider == "openai":
        response = safe_call_openai(model_name, prompt)
    elif provider == "google":
        response = safe_call_google(model_name, prompt)
    elif provider == "groq":
        response = safe_call_groq(model_name, prompt)
    elif provider == "anthropic":
        response = safe_call_anthropic(model_name, prompt)
    else:
        response = f"Error: Unknown provider {provider}"
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Try to parse JSON response
    try:
        # Handle markdown-wrapped JSON (common with Gemini)
        clean_response = response
        if response.startswith("```json") and response.endswith("```"):
            clean_response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            clean_response = response[3:-3].strip()
        
        parsed_response = json.loads(clean_response)
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

def run_working_evaluation():
    """Run evaluation with working models only"""
    print("🎯 DEEMERGE Working Evaluation")
    print("=" * 50)
    
    # Load data
    df, ground_truth = load_test_data()
    messages_str = format_messages(df)
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Define working models (only those with valid API keys)
    working_models = []
    
    # Test each provider
    if OPENAI_API_KEY:
        working_models.extend([
            ("openai", "gpt-4"),
            ("openai", "gpt-3.5-turbo"),
            ("openai", "gpt-4o")
        ])
    
    if GOOGLE_API_KEY:
        working_models.extend([
            ("google", "gemini-1.5-flash"),
            ("google", "gemini-1.5-pro")
        ])
    
    if GROQ_API_KEY:
        working_models.extend([
            ("groq", "llama3-8b-8192"),
            ("groq", "llama3-70b-8192"),
            ("groq", "mixtral-8x7b-32768")
        ])
    
    if ANTHROPIC_API_KEY:
        working_models.extend([
            ("anthropic", "claude-3-opus-20240229"),
            ("anthropic", "claude-3-sonnet-20240229"),
            ("anthropic", "claude-3-haiku-20240307")
        ])
    
    if not working_models:
        print("❌ No valid API keys found!")
        print("Please set up your .env file with valid API keys.")
        return
    
    print(f"🧪 Testing {len(working_models)} models with valid API keys...")
    
    # Test models
    results = {}
    
    for provider, model_name in working_models:
        result = evaluate_model(provider, model_name, prompt)
        results[f"{provider}_{model_name}"] = result
        
        # Save individual result
        output_path = os.path.join(OUTPUT_DIR, f"working_output_{provider}_{model_name.replace('-', '_')}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        status = "✅" if result["json_valid"] else "❌"
        print(f"  {status} {model_name}: {result['duration']:.2f}s, JSON valid: {result['json_valid']}")
    
    # Save comprehensive results
    working_output = os.path.join(OUTPUT_DIR, "working_evaluation_results.json")
    with open(working_output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    valid_count = sum(1 for r in results.values() if r["json_valid"])
    total_count = len(results)
    
    print(f"\\n📊 Results Summary:")
    print(f"  Total models tested: {total_count}")
    print(f"  Valid JSON responses: {valid_count}")
    print(f"  Success rate: {valid_count/total_count*100:.1f}%")
    print(f"\\n📁 Results saved to: {working_output}")
    
    if valid_count > 0:
        print("\\n🎉 Working evaluation complete!")
        print("You can now analyze the results and choose the best model.")
    else:
        print("\\n⚠️  No models returned valid JSON.")
        print("This might be due to API key issues or model availability.")

if __name__ == "__main__":
    run_working_evaluation()
'''
    
    with open("working_evaluation.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created working_evaluation.py")

def main():
    """Main fix function"""
    print("🔧 Fixing DEEMERGE Evaluation Issues")
    print("=" * 50)
    
    # Install missing packages
    install_missing_packages()
    
    # Create working evaluation script
    create_working_evaluation_script()
    
    print("\n" + "=" * 50)
    print("🎉 Fixes applied!")
    print("\nNext steps:")
    print("1. Run the working evaluation:")
    print("   python working_evaluation.py")
    print("2. Or test individual providers:")
    print("   python google_model_eval.py")
    print("3. Check API key validation:")
    print("   python validate_api_keys.py")

if __name__ == "__main__":
    main()
