#!/usr/bin/env python3
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
        import os
        
        # Clear any proxy environment variables that might interfere
        env_vars_to_clear = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
        
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

def safe_call_gemma(model_name: str, prompt: str) -> str:
    """Safely call Gemma model locally"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print(f"Loading Gemma model: {model_name}...")
        
        # Map model names to HuggingFace model IDs (using open source alternatives)
        model_mapping = {
            "dialoGPT-medium": "microsoft/DialoGPT-medium",
            "dialoGPT-large": "microsoft/DialoGPT-large"
        }
        
        hf_model_name = model_mapping.get(model_name, model_name)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Format prompt for DialoGPT (simpler format)
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # Tokenize and generate
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
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
    # Limit to first 50 messages to avoid token limits
    limited_df = df.head(50)
    
    formatted_messages = []
    for _, row in limited_df.iterrows():
        # Truncate long messages to avoid token limits
        text = row['text'][:200] if len(row['text']) > 200 else row['text']
        formatted_messages.append(
            f"[{row['channel']}] {row['user_name']} ({row['timestamp']}): {text} (thread_id={row['thread_id']})"
        )
    
    return "\n".join(formatted_messages)

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
    elif provider == "gemma":
        response = safe_call_gemma(model_name, prompt)
    else:
        response = f"Error: Unknown provider {provider}"
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Try to parse JSON response
    try:
        # Handle markdown-wrapped JSON (common with Gemini)
        cleaned_response = response.strip()
        
        # Remove markdown code blocks
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        # Additional cleaning for trailing characters
        cleaned_response = cleaned_response.strip()
        
        # Remove any trailing newlines and backticks
        while cleaned_response.endswith("\n") or cleaned_response.endswith("`"):
            cleaned_response = cleaned_response.rstrip("\n`")
        
        parsed_response = json.loads(cleaned_response)
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

def run_working_evaluation():
    """Run evaluation with working models only"""
    print("🎯 DEEMERGE Working Evaluation")
    print("=" * 50)
    
    # Load data
    df, ground_truth = load_test_data()
    messages_str = format_messages(df)
    
    # Load prompts
    full_prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    concise_prompt_path = os.path.join(PROMPT_DIR, "approach1_concise_prompt.txt")
    
    with open(full_prompt_path, "r") as f:
        full_prompt_template = f.read()
    
    with open(concise_prompt_path, "r") as f:
        concise_prompt_template = f.read()
    
    full_prompt = full_prompt_template.replace("{messages}", messages_str)
    concise_prompt = concise_prompt_template.replace("{messages}", messages_str)
    
    # Define working models (only those with valid API keys)
    working_models = []
    
    # Test each provider
    if OPENAI_API_KEY:
        working_models.extend([
            ("openai", "gpt-4", concise_prompt),  # Use concise prompt for OpenAI
            ("openai", "gpt-3.5-turbo", concise_prompt),
            ("openai", "gpt-4o", concise_prompt)
        ])
    
    if GOOGLE_API_KEY:
        working_models.extend([
            ("google", "gemini-1.5-flash", full_prompt),
            ("google", "gemini-1.5-pro", full_prompt)
        ])
    
    if GROQ_API_KEY:
        working_models.extend([
            ("groq", "llama3-8b-8192", full_prompt),
            ("groq", "llama3-70b-8192", full_prompt),
            ("groq", "mixtral-8x7b-32768", full_prompt)
        ])
    
    if ANTHROPIC_API_KEY:
        working_models.extend([
            ("anthropic", "claude-3-opus-20240229", full_prompt),
            ("anthropic", "claude-3-sonnet-20240229", full_prompt),
            ("anthropic", "claude-3-haiku-20240307", full_prompt)
        ])
    
    # Add local models (open source alternatives, no API key needed)
    working_models.extend([
        ("gemma", "dialoGPT-medium", full_prompt),
        ("gemma", "dialoGPT-large", full_prompt)
    ])
    
    if not working_models:
        print("❌ No models available for testing!")
        return
    
    print(f"🧪 Testing {len(working_models)} models...")
    
    # Test models
    results = {}
    
    for provider, model_name, prompt in working_models:
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
    
    print(f"\n📊 Results Summary:")
    print(f"  Total models tested: {total_count}")
    print(f"  Valid JSON responses: {valid_count}")
    print(f"  Success rate: {valid_count/total_count*100:.1f}%")
    print(f"\n📁 Results saved to: {working_output}")
    
    if valid_count > 0:
        print("\n🎉 Working evaluation complete!")
        print("You can now analyze the results and choose the best model.")
    else:
        print("\n⚠️  No models returned valid JSON.")
        print("This might be due to API key issues or model availability.")

if __name__ == "__main__":
    run_working_evaluation()
