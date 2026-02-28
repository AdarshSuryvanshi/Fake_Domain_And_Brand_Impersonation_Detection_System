import os
import sys
from huggingface_hub import InferenceClient

# Read the Hugging Face token from standard env vars.
# Use `HUGGINGFACEHUB_API_TOKEN` (recommended) or fall back to common alternatives.
token = "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"

if not token:
    print(
        "Error: Missing Hugging Face API token. Set 'HUGGINGFACEHUB_API_TOKEN' (recommended) "
        "or 'HF_API_KEY'/'HF_TOKEN' before running."
    )
    sys.exit(1)

client = InferenceClient(api_key=token)

result = client.text_classification(
"Your Netflix/Amazon subscription is about to expire due to payment issue. Update detail from the link",    model="ealvaradob/bert-finetuned-phishing",
)

print(result)