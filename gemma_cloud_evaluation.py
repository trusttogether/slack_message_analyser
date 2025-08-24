#!/usr/bin/env python3
"""
Cloud Gemma Evaluation - Using Google's Gemma API
Much faster and more reliable than local models
"""

import json
import time
import google.generativeai as genai
import pandas as pd
from typing import Dict, Any, List
import re
from config.config import GOOGLE_API_KEY

def setup_google_gemma():
    """Setup Google Gemma API"""
    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        return False
    
    genai.configure(api_key=GOOGLE_API_KEY)
    return True

def call_google_gemma(model_name: str, prompt: str) -> Dict[str, Any]:
    """Call Google Gemma API"""
    start_time = time.time()
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        duration = time.time() - start_time
        
        return {
            "success": True,
            "response": response.text,
            "duration": duration
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "response": f"ERROR: {str(e)}",
            "duration": duration
        }

def create_gemma_prompt(messages: List[str]) -> str:
    """Create prompt for Gemma API"""
    # Use first 50 messages
    sample_messages = messages[:50]
    
    prompt = f"""Analyze these Slack messages and identify topics. Return a JSON array with topics.

Messages:
{chr(10).join(sample_messages)}

Return topics in this JSON format:
[{{"title": "Topic Name", "summary": "Brief description", "message_count": 5}}]

Topics:"""
    
    return prompt

def parse_json_response(response: str) -> List[Dict]:
    """Parse JSON response from Gemma"""
    try:
        # Clean up response
        response = response.strip()
        
        # Remove markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        
        # Find JSON array
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        
        # Try parsing the whole response
        return json.loads(response)
        
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return []

def evaluate_google_gemma(model_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate Google Gemma model"""
    print(f"\n🧪 Evaluating {model_name}...")
    
    # Format messages
    messages = []
    for idx, row in df.iterrows():
        thread_id = row.get('thread_id', 'unknown')
        username = row.get('user_name', 'unknown')
        text = row.get('text', '')[:100]  # Limit text length
        messages.append(f"Thread {thread_id}: {username}: {text}")
    
    # Create prompt
    prompt = create_gemma_prompt(messages)
    
    # Call model
    result = call_google_gemma(model_name, prompt)
    
    if not result['success']:
        return {
            "model": model_name,
            "success": False,
            "error": result['response'],
            "duration": result['duration']
        }
    
    # Parse response
    topics = parse_json_response(result['response'])
    
    return {
        "model": model_name,
        "success": True,
        "response": result['response'],
        "parsed_topics": topics,
        "topics_count": len(topics),
        "duration": result['duration'],
        "messages_processed": len(messages)
    }

def run_google_gemma_evaluation():
    """Run evaluation on Google Gemma models"""
    print("🎯 GOOGLE GEMMA API EVALUATION")
    print("=" * 40)
    
    # Setup API
    if not setup_google_gemma():
        return
    
    # Load data
    try:
        df = pd.read_csv("data/Synthetic_Slack_Messages.csv")
        print(f"📊 Loaded {len(df)} messages")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Test Google Gemma models
    models = [
        "gemini-1.5-flash",  # Fast and efficient
        "gemini-1.5-pro",    # More capable
        "gemini-pro"         # Standard model
    ]
    
    results = {}
    
    for model in models:
        result = evaluate_google_gemma(model, df)
        results[model] = result
        
        if result['success']:
            print(f"✅ {model}: {result['topics_count']} topics in {result['duration']:.2f}s")
            print(f"   Messages processed: {result['messages_processed']}")
        else:
            print(f"❌ {model}: {result['error']}")
    
    # Save results
    with open("google_gemma_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: google_gemma_results.json")
    
    # Summary
    successful_models = [m for m, r in results.items() if r['success']]
    print(f"\n📊 Summary: {len(successful_models)}/{len(models)} models successful")
    
    for model in successful_models:
        result = results[model]
        print(f"   {model}: {result['topics_count']} topics, {result['duration']:.2f}s")

if __name__ == "__main__":
    run_google_gemma_evaluation()
