import os
import json
import pandas as pd
import openai
from config.config import OPENAI_API_KEY, MODEL_NAME, DATA_DIR, PROMPT_DIR, OUTPUT_DIR
from src.token_utils import count_tokens_from_response

openai.api_key = OPENAI_API_KEY

def load_messages(csv_path):
    df = pd.read_csv(csv_path)
    return df

def format_messages(df):
    return "\n".join([
        f"[{row['channel']}] {row['user_name']} ({row['timestamp']}): {row['text']} (thread_id={row['thread_id']})"
        for _, row in df.iterrows()
    ])

def main():
    input_csv = os.path.join(DATA_DIR, "Synthetic_Slack_Messages.csv")
    prompt_path = os.path.join(PROMPT_DIR, "approach1_single_prompt.txt")
    output_path = os.path.join(OUTPUT_DIR, "output_approach1.json")

    df = load_messages(input_csv)
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    messages_str = format_messages(df)
    prompt = prompt_template.replace("{messages}", messages_str)

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048
    )

    output = response["choices"][0]["message"]["content"]
    try:
        topics = json.loads(output)
    except Exception:
        print("LLM output is not valid JSON. Saving raw output.")
        topics = output

    with open(output_path, "w") as f:
        json.dump(topics, f, indent=2)

    print(f"Saved output to {output_path}")
    print(f"Token usage: {count_tokens_from_response(response)}")

if __name__ == "__main__":
    main() 