set -euxo pipefail

# Install the packages in r1-v .
cd src/r1-v 
uv add --dev .

cd -

# Addtional modules
uv add wandb==0.18.3
uv add tensorboardx
uv add qwen_vl_utils torchvision
uv add flash-attn --no-build-isolation

# vLLM support 
uv add vllm==0.7.2

# fix transformers version
uv add git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
