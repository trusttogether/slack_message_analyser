#!/usr/bin/env python3
"""
Test script to verify that environment variables are being loaded correctly
from the .env file.
"""

from config.config import (
    OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY
)

def test_env_variables():
    """Test if environment variables are loaded correctly"""
    print("🔑 Testing Environment Variables from .env file")
    print("=" * 50)
    
    # Check each API key
    api_keys = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "GROQ_API_KEY": GROQ_API_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY
    }
    
    all_keys_present = True
    
    for key_name, key_value in api_keys.items():
        if key_value:
            # Show first 10 characters and last 4 characters for security
            masked_key = f"{key_value[:10]}...{key_value[-4:]}" if len(key_value) > 14 else "***"
            print(f"✅ {key_name}: {masked_key}")
        else:
            print(f"❌ {key_name}: Not set")
            all_keys_present = False
    
    print("\n" + "=" * 50)
    
    if all_keys_present:
        print("🎉 All API keys are loaded successfully!")
        print("You can now run the full evaluation:")
        print("  python model_evaluation.py")
        print("  python run_all_models_evaluation.py")
    else:
        print("⚠️  Some API keys are missing.")
        print("Please check your .env file and ensure all keys are set.")
        print("You can still run the demo:")
        print("  python test_evaluation_framework.py")

if __name__ == "__main__":
    test_env_variables()
