# Essential Setup Guide

Quick setup guide for the Deemerge engine.

## Prerequisites

- Python 3.8+
- OpenAI API key

## Quick Setup

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set environment variables:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

4. **Run the engine:**
```bash
# Approach 1 (single-step)
python src/run_approach1.py

# Approach 2 (multi-step)
python src/run_approach2.py
```

## Data Requirements

Ensure you have:
- `data/Synthetic_Slack_Messages.csv` - Input Slack messages
- `data/benchmark_topics_corrected_fixed.json` - Ground truth for evaluation

## Troubleshooting

- Check API key is set correctly
- Verify all dependencies are installed
- Ensure input data files exist
