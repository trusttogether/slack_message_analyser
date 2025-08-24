# 🔑 Environment Variables Guide

This guide explains how to set up your environment variables for the Slack Message Analyser project.

## 📋 **Required Environment Variables**

### **1. API Keys**
You need to obtain API keys from the following services:

| Variable | Description | Required | Format |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes | `sk-proj-...` |
| `GOOGLE_API_KEY` | Google API key for Gemini models | Yes | `AIzaSy...` |
| `GROQ_API_KEY` | Groq API key for Llama/Mixtral | Yes | `gsk_...` |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | Yes | `sk-ant-...` |

### **2. Example .env File**
```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 🔧 **How It Works**

### **1. Automatic Loading**
The `config/config.py` file automatically loads the `.env` file:

```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now you can access the variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
```

### **2. Usage in Scripts**
All evaluation scripts import from `config.config`:

```python
from config.config import (
    OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY,
    DATA_DIR, PROMPT_DIR, OUTPUT_DIR
)

# Use the variables directly
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
```

## 🧪 **Testing Your Setup**

### **1. Test Environment Variables**
```bash
python test_env_variables.py
```

**Expected Output:**
```
🔑 Testing Environment Variables from .env file
==================================================
✅ OPENAI_API_KEY: your_openai_api_key_here
✅ GOOGLE_API_KEY: your_google_api_key_here
✅ GROQ_API_KEY: your_api_key_here
✅ ANTHROPIC_API_KEY: your_api_key_here

==================================================
🎉 All API keys are loaded successfully!
```

### **2. Run Full Evaluation**
```bash
# Test all models with your API keys
python model_evaluation.py

# Or run the master evaluation script
python run_all_models_evaluation.py
```

## 🔒 **Security Best Practices**

### **1. Never Commit .env to Git**
Make sure `.env` is in your `.gitignore` file:

```bash
# .gitignore
.env
*.env
.env.local
.env.production
```

### **2. Use Different Keys for Different Environments**
```bash
# .env.development
OPENAI_API_KEY=sk-test-...

# .env.production  
OPENAI_API_KEY=sk-prod-...
```

### **3. Validate Keys Before Use**
```python
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not set in .env file")
    exit(1)
```

## 🚀 **Quick Start with .env**

### **Step 1: Get API Keys**
- **OpenAI**: https://platform.openai.com/api-keys
- **Google**: https://makersuite.google.com/app/apikey
- **Groq**: https://console.groq.com/keys
- **Anthropic**: https://console.anthropic.com/

### **Step 2: Create .env File**
```bash
cd deemerge_test
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
EOF
```

### **Step 3: Test Setup**
```bash
python test_env_variables.py
```

### **Step 4: Run Evaluation**
```bash
python model_evaluation.py
```

## 🔍 **Troubleshooting**

### **Problem: Keys Not Loading**
```bash
# Check if .env file exists
ls -la .env

# Check file permissions
chmod 600 .env

# Verify file format (no spaces around =)
cat .env
```

### **Problem: python-dotenv Not Installed**
```bash
pip install python-dotenv
```

### **Problem: Invalid API Key Format**
```bash
# Check key formats
python validate_api_keys.py
```

## 📚 **Additional Resources**

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google AI Studio](https://makersuite.google.com/)
- [Groq API Documentation](https://console.groq.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
