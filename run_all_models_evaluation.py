#!/usr/bin/env python3
"""
Master script to run comprehensive evaluation on ALL models:
- OpenAI (GPT-4, GPT-3.5, GPT-4o)
- Google (Gemini Pro, Gemini 1.5 Pro, Gemini 1.5 Flash)
- Groq (Llama3-8B, Llama3-70B, Mixtral-8x7B)
- Anthropic (Claude-3 Opus, Claude-3 Sonnet, Claude-3 Haiku)
- Gemma (2B, 7B, 2B-IT, 7B-IT)
"""

import os
import json
import time
import subprocess
import sys
from datetime import datetime
from typing import Dict, List

def check_api_keys():
    """Check if required API keys are set"""
    required_keys = {
        "OPENAI_API_KEY": "OpenAI models",
        "GOOGLE_API_KEY": "Google Gemini models", 
        "GROQ_API_KEY": "Groq models",
        "ANTHROPIC_API_KEY": "Anthropic Claude models"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{key} ({description})")
    
    if missing_keys:
        print("⚠️  Missing API keys for some providers:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nYou can still run individual evaluations with available keys.")
        return False
    
    print("✅ All API keys are set!")
    return True

def run_evaluation_script(script_name: str, description: str):
    """Run a specific evaluation script"""
    print(f"\n{'='*60}")
    print(f"🚀 Running {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ {description} failed!")
            if result.stderr:
                print("Error:", result.stderr)
                
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running {description}: {e}")

def run_comprehensive_evaluation():
    """Run comprehensive evaluation on all models"""
    print("🎯 DEEMERGE COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check API keys
    all_keys_available = check_api_keys()
    
    # Define evaluation scripts
    evaluations = [
        ("model_evaluation.py", "Comprehensive Model Evaluation (All APIs)"),
        ("google_model_eval.py", "Google Gemini Models"),
        ("gemma_eval.py", "Local Gemma Models")
    ]
    
    # Run evaluations
    for script, description in evaluations:
        if os.path.exists(script):
            run_evaluation_script(script, description)
        else:
            print(f"⚠️  Script not found: {script}")
    
    # Generate final summary
    generate_final_summary()
    
    print(f"\n{'='*60}")
    print("🎉 COMPREHENSIVE EVALUATION COMPLETE!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def generate_final_summary():
    """Generate a final summary of all evaluations"""
    print(f"\n{'='*60}")
    print("📊 GENERATING FINAL SUMMARY")
    print("=" * 60)
    
    summary = {
        "evaluation_date": datetime.now().isoformat(),
        "total_models_tested": 0,
        "providers": [],
        "best_performing_models": [],
        "summary_files": []
    }
    
    # Look for evaluation result files
    output_dir = "."  # Current directory
    result_files = [
        "comprehensive_model_evaluation.json",
        "google_gemini_evaluation.json", 
        "gemma_model_evaluation.json",
        "model_evaluation_report.json",
        "google_gemini_report.json",
        "gemma_evaluation_report.json"
    ]
    
    for result_file in result_files:
        file_path = os.path.join(output_dir, result_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    summary["summary_files"].append(result_file)
                    
                    # Extract model count
                    if "model_performance" in data:
                        summary["total_models_tested"] += len(data["model_performance"])
                    
                    # Extract providers
                    if "summary" in data and "providers" in data["summary"]:
                        summary["providers"].extend(data["summary"]["providers"])
                        
            except Exception as e:
                print(f"⚠️  Error reading {result_file}: {e}")
    
    # Remove duplicates
    summary["providers"] = list(set(summary["providers"]))
    
    # Save final summary
    summary_path = os.path.join(output_dir, "final_evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Final summary saved to: {summary_path}")
    print(f"📊 Total models tested: {summary['total_models_tested']}")
    print(f"🏢 Providers: {', '.join(summary['providers'])}")

if __name__ == "__main__":
    run_comprehensive_evaluation()
