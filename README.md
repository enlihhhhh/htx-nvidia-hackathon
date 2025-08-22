## HTX-Nvidia Hackathon Gradio Demo App

## Overview
....

## Set-up

.... to be added

### Extra dependancies to install


```bash
# # Install these dependenacies first
# uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# uv pip install flash_attn-2.8.2+cu12torch24cxx11abifalse-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl

# # Then, run pip install requirements without the above packages
# uv pip install requirements.txt

# # Finally, install Kimi-Audio
# uv pip install git+https://github.com/MoonshotAI/Kimi-Audio.git
# ```

Then, run this to install the Kimi Audio Package
```bash
uv pip install git+https://github.com/MoonshotAI/Kimi-Audio.git
```

## Set-up

Run this to run NVIDIA's RAG Docker Container
```bash
VECTORSTORE_GPU_DEVICE_ID=0 docker compose -f rag/deploy/compose/vectordb.yaml up
```

Run this to run NVIDIA's 

Run this first to set up venv
```bash
uv run app.py
```
## Contributions
```text
title: Omni Mini
emoji: 🌖
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.0.0b1
app_file: app.py
pinned: false
license: mit

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
```