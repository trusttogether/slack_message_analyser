#!/usr/bin/env python3
"""
STEP 1: Comprehensive Model Evaluation for Corpus-Wide Topic Clustering
Tests all models against 300 messages and compares with benchmark
"""

import json
import time
import os
import pandas as pd
from typing import Dict, Any, List
from difflib import SequenceMatcher

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
        parsed = json.loads(cleaned_response)
        
        # Handle both direct array and nested "topics" structure
        if isinstance(parsed, dict) and "topics" in parsed:
            return parsed["topics"]
        elif isinstance(parsed, list):
            return parsed
        else:
            return []
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return []

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def evaluate_topic_accuracy(predicted_topics: List[Dict], benchmark_topics: List[Dict]) -> Dict[str, float]:
    """Evaluate predicted topics against benchmark"""
    if not predicted_topics or not benchmark_topics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "similarity": 0.0}
    
    # Calculate topic-level similarity
    similarities = []
    for pred_topic in predicted_topics:
        best_similarity = 0.0
        for bench_topic in benchmark_topics:
            # Compare titles and summaries
            title_sim = calculate_similarity(
                pred_topic.get("title", ""), 
                bench_topic.get("title", "")
            )
            summary_sim = calculate_similarity(
                pred_topic.get("summary", ""), 
                bench_topic.get("summary", "")
            )
            avg_sim = (title_sim + summary_sim) / 2
            best_similarity = max(best_similarity, avg_sim)
        similarities.append(best_similarity)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    
    # Simple precision/recall based on topic count
    predicted_count = len(predicted_topics)
    benchmark_count = len(benchmark_topics)
    
    if predicted_count == 0:
        precision = 0.0
    else:
        precision = min(predicted_count, benchmark_count) / predicted_count
    
    if benchmark_count == 0:
        recall = 0.0
    else:
        recall = min(predicted_count, benchmark_count) / benchmark_count
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall, 
        "f1": f1,
        "similarity": avg_similarity,
        "predicted_topics": predicted_count,
        "benchmark_topics": benchmark_count
    }

def format_all_messages(df: pd.DataFrame) -> str:
    """Format all messages for comprehensive analysis"""
    formatted = []
    
    for idx, msg in df.iterrows():
        thread_id = msg.get('thread_id', 'unknown')
        username = msg.get('user_name', 'unknown')
        message_text = msg.get('text', '')
        channel = msg.get('channel', 'unknown')
        
        formatted.append(f"Row {idx+1} - Channel {channel} - Thread {thread_id}: {username}: {message_text}")
    
    return "\n".join(formatted)

def evaluate_model_step1(model_name: str, prompt: str, benchmark_topics: List[Dict]) -> Dict[str, Any]:
    """Evaluate a single model for STEP 1"""
    start_time = time.time()
    
    print(f"Testing {model_name} for STEP 1...")
    
    # Call the model
    response = safe_call_google_gemma(model_name, prompt)
    
    # Parse response
    parsed_response = parse_json_response(response)
    json_valid = len(parsed_response) > 0
    
    # Calculate accuracy metrics
    accuracy_metrics = evaluate_topic_accuracy(parsed_response, benchmark_topics)
    
    duration = time.time() - start_time
    
    return {
        "provider": "google",
        "model": model_name,
        "response": response,
        "parsed_response": parsed_response,
        "json_valid": json_valid,
        "accuracy_metrics": accuracy_metrics,
        "duration": duration,
        "timestamp": time.time()
    }

def load_test_data():
    """Load test data and benchmark"""
    input_csv = os.path.join(DATA_DIR, "Synthetic_Slack_Messages.csv")
    ground_truth_path = os.path.join(DATA_DIR, "benchmark_topics_corrected_fixed.json")
    
    df = pd.read_csv(input_csv)
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    return df, ground_truth

def rank_models_by_performance(results: Dict[str, Dict]) -> List[Dict]:
    """Rank models by F1 score and cost-effectiveness"""
    ranked_models = []
    
    for model_key, result in results.items():
        if result["json_valid"]:
            metrics = result["accuracy_metrics"]
            model_info = {
                "model": result["model"],
                "f1_score": metrics["f1"],
                "similarity": metrics["similarity"],
                "duration": result["duration"],
                "topics_found": metrics["predicted_topics"],
                "benchmark_topics": metrics["benchmark_topics"],
                "performance_score": metrics["f1"] * metrics["similarity"] / result["duration"]
            }
            ranked_models.append(model_info)
    
    # Sort by performance score (higher is better)
    ranked_models.sort(key=lambda x: x["performance_score"], reverse=True)
    return ranked_models

def run_step1_evaluation():
    """Run comprehensive STEP 1 evaluation"""
    print("🎯 STEP 1: CORPUS-WIDE TOPIC CLUSTERING EVALUATION")
    print("=" * 60)
    
    if not GOOGLE_API_KEY:
        print("❌ No Google API key found! Please set GOOGLE_API_KEY in .env file")
        return
    
    # Load data
    df, ground_truth = load_test_data()
    print(f"📊 Loaded {len(df)} messages from CSV file")
    print(f"📋 Loaded {len(ground_truth)} benchmark topics")
    
    # Format all messages
    messages_str = format_all_messages(df)
    print(f"📝 Formatted {len(messages_str.split('Row'))} message entries")
    
    # Load prompt
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{messages}", messages_str)
    
    # Define models to test for STEP 1
    step1_models = [
        "gemma-3-1b-it",
        "gemma-3-4b-it", 
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "gemma-3n-e4b-it",
        "gemma-3n-e2b-it"
    ]
    
    print(f"🧪 Testing {len(step1_models)} models for STEP 1...")
    
    # Test each model
    results = {}
    valid_count = 0
    
    for model_name in step1_models:
        result = evaluate_model_step1(model_name, prompt, ground_truth)
        results[f"google_{model_name}"] = result
        
        status = "✅" if result["json_valid"] else "❌"
        metrics = result["accuracy_metrics"]
        print(f"  {status} {model_name}: {result['duration']:.2f}s, F1: {metrics['f1']:.3f}, Similarity: {metrics['similarity']:.3f}")
        
        if result["json_valid"]:
            valid_count += 1
    
    # Rank models by performance
    ranked_models = rank_models_by_performance(results)
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "step1_evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print ranking
    print(f"\n🏆 STEP 1 MODEL RANKING (by performance/cost):")
    print("=" * 60)
    for i, model in enumerate(ranked_models, 1):
        print(f"{i}. {model['model']}")
        print(f"   F1 Score: {model['f1_score']:.3f}")
        print(f"   Similarity: {model['similarity']:.3f}")
        print(f"   Duration: {model['duration']:.2f}s")
        print(f"   Topics Found: {model['topics_found']}/{model['benchmark_topics']}")
        print(f"   Performance Score: {model['performance_score']:.4f}")
        print()
    
    # Print summary
    print(f"📊 STEP 1 Results Summary:")
    print(f"  Total models tested: {len(step1_models)}")
    print(f"  Valid responses: {valid_count}")
    print(f"  Success rate: {(valid_count/len(step1_models)*100):.1f}%")
    print(f"\n📁 Results saved to: {output_file}")
    
    if ranked_models:
        best_model = ranked_models[0]
        print(f"\n🎯 RECOMMENDED MODEL FOR STEP 1: {best_model['model']}")
        print(f"   Best F1 Score: {best_model['f1_score']:.3f}")
        print(f"   Best Performance/Cost Ratio: {best_model['performance_score']:.4f}")
    
    print("\n🎉 STEP 1 evaluation complete!")

if __name__ == "__main__":
    run_step1_evaluation()
