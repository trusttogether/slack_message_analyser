#!/usr/bin/env python3
"""
Comprehensive Gemma Evaluation - Process ALL Messages
Uses smart batching to ensure all 300 messages are covered
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

def create_message_batches(df: pd.DataFrame, batch_size: int = 100) -> List[pd.DataFrame]:
    """Create batches of messages to process all data"""
    batches = []
    total_messages = len(df)
    
    for i in range(0, total_messages, batch_size):
        batch = df.iloc[i:i+batch_size]
        batches.append(batch)
    
    print(f"📦 Created {len(batches)} batches of ~{batch_size} messages each")
    return batches

def format_messages_batch(df_batch: pd.DataFrame, batch_num: int) -> str:
    """Format a batch of messages for prompt"""
    formatted = []
    
    for idx, msg in df_batch.iterrows():
        thread_id = msg.get('thread_id', 'unknown')
        username = msg.get('user_name', 'unknown')
        message_text = msg.get('text', '')
        
        # Include original row number for reference
        formatted.append(f"Row {idx+1} - Thread {thread_id}: {username}: {message_text}")
    
    return f"=== BATCH {batch_num + 1} ===\n" + "\n".join(formatted)

def evaluate_gemma_model_comprehensive(model_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate a single Gemma model with comprehensive message coverage"""
    start_time = time.time()
    
    print(f"Testing {model_name} with comprehensive coverage...")
    
    # Create batches
    batches = create_message_batches(df, batch_size=100)
    
    all_results = []
    total_messages_processed = 0
    
    for batch_num, batch in enumerate(batches):
        print(f"  Processing batch {batch_num + 1}/{len(batches)} ({len(batch)} messages)...")
        
        # Format batch
        messages_str = format_messages_batch(batch, batch_num)
        
        # Load prompt
        prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        
        prompt = prompt_template.replace("{messages}", messages_str)
        
        # Call model
        response = safe_call_google_gemma(model_name, prompt)
        
        # Parse response
        parsed_response = parse_json_response(response)
        
        if parsed_response:
            all_results.extend(parsed_response)
            total_messages_processed += len(batch)
        
        # Small delay between batches
        time.sleep(1)
    
    duration = time.time() - start_time
    
    return {
        "provider": "google",
        "model": model_name,
        "total_messages_processed": total_messages_processed,
        "total_messages_available": len(df),
        "coverage_percentage": (total_messages_processed / len(df)) * 100,
        "batches_processed": len(batches),
        "topics_found": len(all_results),
        "response": "Comprehensive batch processing completed",
        "parsed_response": all_results,
        "json_valid": len(all_results) > 0,
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

def run_comprehensive_gemma_evaluation():
    """Run comprehensive evaluation with ALL messages"""
    print("🎯 COMPREHENSIVE GEMMA EVALUATION")
    print("=" * 50)
    
    if not GOOGLE_API_KEY:
        print("❌ No Google API key found! Please set GOOGLE_API_KEY in .env file")
        return
    
    # Load data
    df, ground_truth = load_test_data()
    print(f"📊 Loaded {len(df)} messages from CSV file")
    print(f"📝 Total characters in all messages: {df['text'].str.len().sum():,}")
    
    # Define Gemma models to test (focus on larger models for comprehensive analysis)
    gemma_models = [
        "gemma-3-12b-it",
        "gemma-3-27b-it"
    ]
    
    print(f"🧪 Testing {len(gemma_models)} Gemma models with comprehensive coverage...")
    
    # Test each model
    results = {}
    valid_count = 0
    
    for model_name in gemma_models:
        result = evaluate_gemma_model_comprehensive(model_name, df)
        results[f"google_{model_name}"] = result
        
        status = "✅" if result["json_valid"] else "❌"
        coverage = result["coverage_percentage"]
        topics = result["topics_found"]
        print(f"  {status} {model_name}: {result['duration']:.2f}s, Coverage: {coverage:.1f}%, Topics: {topics}")
        
        if result["json_valid"]:
            valid_count += 1
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "comprehensive_gemma_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Comprehensive Results Summary:")
    print(f"  Total models tested: {len(gemma_models)}")
    print(f"  Valid responses: {valid_count}")
    print(f"  Success rate: {(valid_count/len(gemma_models)*100):.1f}%")
    print(f"\n📁 Results saved to: {output_file}")
    
    print("\n🎉 Comprehensive evaluation complete!")
    print("This approach processes ALL 300 messages in batches for complete coverage.")

if __name__ == "__main__":
    run_comprehensive_gemma_evaluation()
