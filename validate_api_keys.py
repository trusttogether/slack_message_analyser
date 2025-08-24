#!/usr/bin/env python3
"""
Script to validate API keys and provide guidance on fixing issues.
"""

import os
import json
from config.config import (
    OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY
)

def test_openai_key(api_key):
    """Test OpenAI API key"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return True, "Valid"
    except Exception as e:
        if "proxies" in str(e):
            return False, "Invalid client initialization (proxies error)"
        elif "401" in str(e):
            return False, "Invalid API key"
        else:
            return False, f"Error: {str(e)}"

def validate_google_key():
    """Validate Google API key"""
    if not GOOGLE_API_KEY:
        return False, "No API key provided"
    
    if not GOOGLE_API_KEY.startswith("AIza"):
        return False, "Invalid format - should start with 'AIza'"
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        # Try to list models
        models = genai.list_models()
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def validate_groq_key():
    """Validate Groq API key"""
    if not GROQ_API_KEY:
        return False, "No API key provided"
    
    if not GROQ_API_KEY.startswith("xai-"):
        return False, "Invalid format - should start with 'xai-'"
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        # Try a simple test call
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def validate_anthropic_key():
    """Validate Anthropic API key"""
    if not ANTHROPIC_API_KEY:
        return False, "No API key provided"
    
    if not ANTHROPIC_API_KEY.startswith("xai-"):
        return False, "Invalid format - should start with 'xai-'"
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        # Try a simple test call
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=5,
            messages=[{"role": "user", "content": "Hello"}]
        )
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def validate_openai_key():
    """Validate OpenAI API key"""
    if not OPENAI_API_KEY:
        return False, "No API key provided"
    
    if not OPENAI_API_KEY.startswith("sk-"):
        return False, "Invalid format - should start with 'sk-'"
    
    return test_openai_key(OPENAI_API_KEY)

def main():
    """Main validation function"""
    print("🔑 API Key Validation")
    print("=" * 50)
    
    validations = [
        ("OpenAI", validate_openai_key),
        ("Google", validate_google_key),
        ("Groq", validate_groq_key),
        ("Anthropic", validate_anthropic_key)
    ]
    
    print("🔍 Validating API Keys...")
    print("=" * 50)
    
    all_valid = True
    
    for provider, validator in validations:
        is_valid, message = validator()
        status = "✅" if is_valid else "❌"
        print(f"{status} {provider}: {message}")
        if not is_valid:
            all_valid = False
    
    print("\n" + "=" * 50)
    
    if all_valid:
        print("🎉 All API keys are valid!")
        print("You can now run the full evaluation:")
        print("  python model_evaluation.py")
    else:
        print("⚠️  Some API keys have issues.")
        print("\n🔧 How to fix:")
        print("1. Get valid API keys from:")
        print("   - OpenAI: https://platform.openai.com/api-keys")
        print("   - Google: https://makersuite.google.com/app/apikey")
        print("   - Groq: https://console.groq.com/keys")
        print("   - Anthropic: https://console.anthropic.com/")
        print("2. Update your .env file with the correct keys")
        print("3. Run this validation script again")
        print("\n💡 You can still run the demo:")
        print("  python test_evaluation_framework.py")

if __name__ == "__main__":
    main()
