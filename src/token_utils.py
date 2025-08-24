# token_utils.py
# Utility functions for token counting (optional)

# (Implementation placeholder) 

def count_tokens_from_response(response):
    return response.get("usage", {}).get("total_tokens", 0)

def count_tokens_from_prompt(prompt):
    # Approximate: 1 token ≈ 4 chars (for English)
    return len(prompt) // 4 