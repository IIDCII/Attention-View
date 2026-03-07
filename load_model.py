# loading the model used for inference
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="Qwen/Qwen3.5-2B")
print(f"Model saved at: {model_path}")
