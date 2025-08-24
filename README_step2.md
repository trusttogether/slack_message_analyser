# Step 2 Evaluation Guide

This document provides guidance for evaluating the Step 2 approach of the Deemerge engine.

## Overview

Step 2 evaluation focuses on the multi-step pipeline approach that includes:
1. Topic grouping
2. Action item enrichment  
3. Routing recommendations

## Evaluation Metrics

- **Topic Accuracy**: Jaccard similarity for topic overlap
- **Action Item Precision**: Precision, recall, F1 for action items
- **Routing Quality**: Relevance of routing recommendations

## Running Evaluation

```bash
python src/eval/step2_evaluator.py
```

## Expected Output

The evaluator will generate:
- Comparison metrics against ground truth
- Detailed analysis of each step
- Overall performance summary
