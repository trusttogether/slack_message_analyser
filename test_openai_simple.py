#!/usr/bin/env python3
"""
Simple test for OpenAI client
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def test_openai():
    """Test OpenAI with a simple call"""
    try:
        print("Testing OpenAI client...")
        print(f"API Key: {OPENAI_API_KEY[:10]}..." if OPENAI_API_KEY else "No API key")
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"✅ Success: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_openai()
