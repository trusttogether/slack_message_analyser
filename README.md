# DEEMERGE - Comprehensive LLM Evaluation Engine

## 🎯 **Project Goal**

DEEMERGE is a **comprehensive LLM evaluation engine** that benchmarks and compares the performance of **ALL major language models** for Slack message topic detection and analysis. The goal is to find the best performing models across different providers and use cases.

## 🏢 **Supported Model Providers**

### **Cloud APIs:**
- **OpenAI**: GPT-4, GPT-3.5-turbo, GPT-4o
- **Google**: Gemini Pro, Gemini 1.5 Pro, Gemini 1.5 Flash
- **Groq**: Llama3-8B, Llama3-70B, Mixtral-8x7B
- **Anthropic**: Claude-3 Opus, Claude-3 Sonnet, Claude-3 Haiku

### **Local Models:**
- **Google Gemma**: 2B, 7B, 2B-IT, 7B-IT (local inference)

## 🚀 **Quick Start**

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 3. Run Comprehensive Evaluation
```bash
# Test ALL models (recommended)
python run_all_models_evaluation.py

# Or test individual providers:
python model_evaluation.py          # All cloud APIs
python google_model_eval.py         # Google Gemini only
python gemma_eval.py               # Local Gemma models
```

## 📊 **Evaluation Metrics**

Each model is evaluated on:
- **Topic Detection Accuracy**: Jaccard similarity for topic overlap
- **Action Item Extraction**: Precision, recall, F1 scores
- **JSON Response Validity**: Success rate of structured outputs
- **Performance**: Response time and token usage
- **Cost Analysis**: API costs per request

## 📁 **Project Structure**

```
deemerge_test/
├── model_evaluation.py              # Comprehensive cloud API evaluation
├── google_model_eval.py             # Google Gemini evaluation
├── gemma_eval.py                   # Local Gemma evaluation
├── run_all_models_evaluation.py     # Master evaluation script
├── src/
│   ├── run_approach1.py            # Single-step topic detection
│   ├── run_approach2.py            # Multi-step pipeline
│   └── eval/                       # Evaluation utilities
├── prompts/                        # LLM prompt templates
├── data/                           # Test data and ground truth
└── evaluation_results/             # Evaluation outputs
```

## 🎯 **Core Approaches**

### **Approach 1**: Single-step topic detection
- One LLM call to detect topics, participants, and action items
- Faster processing, lower cost
- Good baseline performance

### **Approach 2**: Multi-step pipeline
- Step 1: Topic grouping
- Step 2: Action item enrichment  
- Step 3: Routing recommendations
- More detailed analysis, higher accuracy

## 📈 **Expected Results**

The evaluation will generate:
- **Individual model results**: JSON outputs for each model
- **Comparative analysis**: Performance rankings across providers
- **Cost-benefit analysis**: Performance vs. cost trade-offs
- **Recommendations**: Best models for different use cases

## 🔧 **Advanced Usage**

### Custom Model Testing
```python
# Add custom models to model_evaluation.py
CUSTOM_MODELS = {
    "your-provider": {
        "your-model": {"provider": "your-provider", "model": "your-model"}
    }
}
```

### Batch Evaluation
```bash
# Run evaluation with specific models only
python model_evaluation.py --models openai,google
```

## 📋 **Output Files**

- `comprehensive_model_evaluation.json` - All model results
- `model_evaluation_report.json` - Performance summary
- `final_evaluation_summary.json` - Master summary
- Individual model outputs in `output_*` files

## 🎉 **Key Benefits**

1. **Comprehensive Coverage**: Tests all major LLM providers
2. **Standardized Evaluation**: Same prompts and metrics across all models
3. **Cost Analysis**: Compare performance vs. cost
4. **Local Options**: Include local Gemma models for privacy
5. **Actionable Insights**: Clear recommendations for different use cases

## 🔍 **Research Applications**

- **Model Selection**: Choose best model for your use case
- **Performance Benchmarking**: Compare model capabilities
- **Cost Optimization**: Find best performance/cost ratio
- **Research Validation**: Reproducible evaluation framework 