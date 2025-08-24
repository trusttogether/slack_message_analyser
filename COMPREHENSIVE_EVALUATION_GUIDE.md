# DEEMERGE Comprehensive Evaluation Guide

## 🎯 **Project Overview**

DEEMERGE is a **comprehensive LLM evaluation engine** that benchmarks and compares the performance of **ALL major language models** for Slack message topic detection and analysis.

## 🏢 **Supported Models**

### **Cloud APIs (Require API Keys):**
- **OpenAI**: GPT-4, GPT-3.5-turbo, GPT-4o
- **Google**: Gemini Pro, Gemini 1.5 Pro, Gemini 1.5 Flash
- **Groq**: Llama3-8B, Llama3-70B, Mixtral-8x7B
- **Anthropic**: Claude-3 Opus, Claude-3 Sonnet, Claude-3 Haiku

### **Local Models (No API Keys Required):**
- **Google Gemma**: 2B, 7B, 2B-IT, 7B-IT (local inference)

## 🚀 **Quick Start**

### **Option 1: Demo Mode (No API Keys Required)**
```bash
# Test the framework with simulated models
python test_evaluation_framework.py
```

### **Option 2: Full Evaluation (Requires API Keys)**

#### **Step 1: Setup Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### **Step 2: Get API Keys**
- **OpenAI**: https://platform.openai.com/api-keys
- **Google**: https://makersuite.google.com/app/apikey
- **Groq**: https://console.groq.com/keys
- **Anthropic**: https://console.anthropic.com/

#### **Step 3: Set Environment Variables**
```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

#### **Step 4: Run Evaluation**

**All Models (Recommended):**
```bash
python run_all_models_evaluation.py
```

**Individual Providers:**
```bash
# All cloud APIs
python model_evaluation.py

# Google Gemini only
python google_model_eval.py

# Local Gemma models
python gemma_eval.py
```

## 📊 **Evaluation Metrics**

Each model is evaluated on:
- **Topic Detection Accuracy**: Jaccard similarity for topic overlap
- **Action Item Extraction**: Precision, recall, F1 scores
- **JSON Response Validity**: Success rate of structured outputs
- **Performance**: Response time and token usage
- **Cost Analysis**: API costs per request

## 📁 **Output Files**

### **Demo Mode:**
- `demo_evaluation_results.json` - Simulated model results
- `demo_evaluation_report.json` - Demo performance summary

### **Full Evaluation:**
- `comprehensive_model_evaluation.json` - All model results
- `model_evaluation_report.json` - Performance summary
- `final_evaluation_summary.json` - Master summary
- Individual model outputs in `output_*` files

## 🔧 **Troubleshooting**

### **Missing Dependencies**
```bash
# Install missing packages
pip install torch transformers accelerate
pip install google-generativeai groq anthropic
```

### **API Key Issues**
```bash
# Check if keys are set
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY
echo $GROQ_API_KEY
echo $ANTHROPIC_API_KEY
```

### **Memory Issues (Local Models)**
```bash
# For Gemma models, ensure sufficient RAM
# 2B models: ~4GB RAM
# 7B models: ~16GB RAM
```

### **File Path Issues**
```bash
# Ensure you're in the correct directory
cd deemerge_test
pwd  # Should show /path/to/deemerge/deemerge_test
```

## 📈 **Expected Results**

### **Performance Rankings (Typical):**
1. **Claude-3 Opus** - Highest accuracy, highest cost
2. **GPT-4** - High accuracy, high cost
3. **Gemini 1.5 Pro** - Good accuracy, moderate cost
4. **Claude-3 Sonnet** - Good accuracy, moderate cost
5. **GPT-3.5-turbo** - Moderate accuracy, low cost
6. **Local Gemma** - Variable accuracy, no cost

### **Cost Comparison (per 1K tokens):**
- **OpenAI GPT-4**: ~$0.03-0.06
- **Claude-3 Opus**: ~$0.015-0.075
- **Gemini Pro**: ~$0.0005-0.0015
- **Groq**: ~$0.0001-0.0008
- **Local Gemma**: $0.00

## 🎯 **Use Cases**

### **Research & Development:**
- Model selection for specific tasks
- Performance benchmarking
- Cost optimization analysis

### **Production Deployment:**
- Choose best model for your use case
- Balance accuracy vs. cost
- Consider privacy requirements

### **Academic Research:**
- Reproducible evaluation framework
- Standardized metrics across models
- Comparative analysis

## 🔍 **Advanced Usage**

### **Custom Model Testing**
```python
# Add custom models to model_evaluation.py
CUSTOM_MODELS = {
    "your-provider": {
        "your-model": {"provider": "your-provider", "model": "your-model"}
    }
}
```

### **Batch Evaluation**
```bash
# Run evaluation with specific models only
python model_evaluation.py --models openai,google
```

### **Custom Prompts**
```bash
# Modify prompts in prompts/ directory
# Then run evaluation with custom prompts
```

## 📋 **File Structure**

```
deemerge_test/
├── 🧪 EVALUATION SCRIPTS
│   ├── test_evaluation_framework.py    # Demo mode (no API keys)
│   ├── model_evaluation.py             # All cloud APIs
│   ├── google_model_eval.py            # Google Gemini
│   ├── gemma_eval.py                  # Local Gemma
│   └── run_all_models_evaluation.py    # Master script
├── 🔧 CORE ENGINE
│   ├── src/run_approach1.py           # Single-step detection
│   ├── src/run_approach2.py           # Multi-step pipeline
│   └── src/eval/                      # Evaluation tools
├── 📝 PROMPTS
│   ├── approach1_single_prompt.txt
│   └── approach2_*.txt
├── 📊 DATA
│   ├── Synthetic_Slack_Messages.csv
│   └── benchmark_topics_corrected_fixed.json
└── 📋 CONFIG
    ├── config.py                      # Settings
    └── requirements.txt               # Dependencies
```

## 🎉 **Success Indicators**

✅ **Framework working correctly when you see:**
- Demo runs without errors
- JSON files generated in output directory
- Performance metrics calculated
- Cost analysis completed

✅ **Ready for production when:**
- All API keys configured
- Models tested and ranked
- Best model selected for your use case
- Cost-benefit analysis complete

## 🆘 **Getting Help**

1. **Run demo first**: `python test_evaluation_framework.py`
2. **Check API keys**: Ensure all required keys are set
3. **Verify dependencies**: `pip list | grep -E "(torch|transformers|openai|google)"`
4. **Check file paths**: Ensure you're in the `deemerge_test` directory

---

**🎯 Goal**: Find the best performing LLM for your Slack topic detection needs!
