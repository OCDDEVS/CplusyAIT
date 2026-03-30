"""Download a HuggingFace model for use with the framework.

Usage:
    python download_model.py <model_id> [output_dir]

Examples:
    python download_model.py stas/tiny-random-llama-2
    python download_model.py TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python download_model.py meta-llama/Llama-3.2-1B models/llama-3.2-1b
"""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

model_id = sys.argv[1]
default_dir = f"models/{model_id.split('/')[-1]}"
output_dir = sys.argv[2] if len(sys.argv) > 2 else default_dir

print(f"Downloading {model_id} -> {output_dir}")
snapshot_download(
    model_id,
    local_dir=output_dir,
    allow_patterns=[
        "*.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
    ],
)
print(f"Done! Model saved to: {output_dir}")
