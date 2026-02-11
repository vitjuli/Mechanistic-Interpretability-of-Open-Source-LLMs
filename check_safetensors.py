from safetensors import safe_open
from huggingface_hub import hf_hub_download
# Скачай layer_15
path = hf_hub_download(
    repo_id="mwhanna/qwen3-4b-transcoders",
    filename="layer_15.safetensors"
)
# Посмотри что внутри
with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
    print("Keys in safetensors file:")
    for key in sorted(keys):
        tensor = f.get_slice(key)
        shape = tensor.get_shape()
        print(f"  {key:40s} : {shape}")
