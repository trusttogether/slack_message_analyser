#!/usr/bin/env python3
"""
Setup script for DEEMERGE comprehensive model evaluation
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("🔧 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        # Activate venv and install requirements
        if os.name == 'nt':  # Windows
            pip_cmd = "venv\\Scripts\\pip"
        else:  # Unix/Linux/Mac
            pip_cmd = "venv/bin/pip"
        
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_api_keys():
    """Check and guide API key setup"""
    print("\n🔑 API Key Configuration")
    print("=" * 40)
    
    required_keys = {
        "OPENAI_API_KEY": "OpenAI (GPT-4, GPT-3.5, GPT-4o)",
        "GOOGLE_API_KEY": "Google (Gemini Pro, Gemini 1.5)",
        "GROQ_API_KEY": "Groq (Llama3, Mixtral)",
        "ANTHROPIC_API_KEY": "Anthropic (Claude-3)"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if os.getenv(key):
            print(f"✅ {key} - {description}")
        else:
            print(f"❌ {key} - {description}")
            missing_keys.append(key)
    
    if missing_keys:
        print(f"\n⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("\nTo set API keys, run:")
        print("export OPENAI_API_KEY='your-key'")
        print("export GOOGLE_API_KEY='your-key'")
        print("export GROQ_API_KEY='your-key'")
        print("export ANTHROPIC_API_KEY='your-key'")
        print("\nOr add them to your .env file")
        return False
    
    print("\n✅ All API keys are configured!")
    return True

def check_data_files():
    """Check if required data files exist"""
    print("\n📁 Data Files Check")
    print("=" * 40)
    
    required_files = [
        "data/Synthetic_Slack_Messages.csv",
        "data/benchmark_topics_corrected_fixed.json",
        "data/ground_truth_topics.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing data files: {', '.join(missing_files)}")
        print("Please ensure all required data files are present")
        return False
    
    print("\n✅ All data files are present!")
    return True

def create_env_template():
    """Create .env template file"""
    env_template = """# DEEMERGE API Keys
# Get your API keys from:
# OpenAI: https://platform.openai.com/api-keys
# Google: https://makersuite.google.com/app/apikey
# Groq: https://console.groq.com/keys
# Anthropic: https://console.anthropic.com/

OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_template)
        print("✅ Created .env template file")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function"""
    print("🚀 DEEMERGE Setup - Comprehensive LLM Evaluation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create virtual environment
    if not create_virtual_environment():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Check data files
    if not check_data_files():
        return
    
    # Create .env template
    create_env_template()
    
    # Check API keys
    check_api_keys()
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Set your API keys in the .env file or environment variables")
    print("2. Activate the virtual environment:")
    print("   source venv/bin/activate  # Linux/Mac")
    print("   venv\\Scripts\\activate     # Windows")
    print("3. Run comprehensive evaluation:")
    print("   python run_all_models_evaluation.py")
    print("\nFor individual evaluations:")
    print("   python model_evaluation.py      # All cloud APIs")
    print("   python google_model_eval.py     # Google Gemini only")
    print("   python gemma_eval.py           # Local Gemma models")

if __name__ == "__main__":
    main()
