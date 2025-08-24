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
    prompt_grouping = os.path.join(PROMPT_DIR, "approach2_topic_grouping.txt")
    prompt_enrich = os.path.join(PROMPT_DIR, "approach2_enrichment_prompt.txt")
    prompt_routing = os.path.join(PROMPT_DIR, "approach2_routing_prompt.txt")
    output_path = os.path.join(OUTPUT_DIR, "output_approach2.json")

    df = load_messages(input_csv)
    messages_str = format_messages(df)

    # Step 1: Topic grouping
    with open(prompt_grouping, "r") as f:
        grouping_template = f.read()
    grouping_prompt = grouping_template.replace("{messages}", messages_str)
    grouping_response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": grouping_prompt}],
        temperature=0.2,
        max_tokens=2048
    )
    topics = grouping_response["choices"][0]["message"]["content"]
    try:
        topics_json = json.loads(topics)
    except Exception:
        print("LLM output is not valid JSON. Saving raw output.")
        topics_json = topics

    # Step 2: Enrichment (action items)
    topics_and_messages = json.dumps({
        "topics": topics_json,
        "messages": messages_str
    })
    with open(prompt_enrich, "r") as f:
        enrich_template = f.read()
    enrich_prompt = enrich_template.replace("{topics_and_messages}", topics_and_messages)
    enrich_response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": enrich_prompt}],
        temperature=0.2,
        max_tokens=1024
    )
    actions = enrich_response["choices"][0]["message"]["content"]
    try:
        actions_json = json.loads(actions)
    except Exception:
        print("LLM output is not valid JSON. Saving raw output.")
        actions_json = actions

    # Step 3: Personalization (routing)
    with open(prompt_routing, "r") as f:
        routing_template = f.read()
    routing_prompt = routing_template.replace("{topics}", json.dumps(topics_json))
    routing_response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": routing_prompt}],
        temperature=0.2,
        max_tokens=512
    )
    routing = routing_response["choices"][0]["message"]["content"]
    try:
        routing_json = json.loads(routing)
    except Exception:
        print("LLM output is not valid JSON. Saving raw output.")
        routing_json = routing

    # Combine all outputs
    output = {
        "topics": topics_json,
        "actions": actions_json,
        "routing": routing_json
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved output to {output_path}")
    print(f"Token usage: grouping={count_tokens_from_response(grouping_response)}, enrichment={count_tokens_from_response(enrich_response)}, routing={count_tokens_from_response(routing_response)}")

if __name__ == "__main__":
    main() 