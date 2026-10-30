# Bernoulli serving image.
#
# Base: vLLM's prebuilt image. vLLM + torch + CUDA + Triton kernels are all
# pre-installed and tested together, which saves us from the gcc / libcuda
# dance that CUDA runtime base images need for Triton JIT at engine init.
#
# cu129 vs our pyproject's cu130 pin on bernoulli: CUDA 12.9 vs 13.0 — both
# work on the A10G/L40S driver. Pyproject's cu130 is for the dev-box uv env;
# the image uses its own baked-in torch/vllm (cu129) because it works.
#
# Model weights: NOT baked into the image. Mount an HF cache at /data/hf-cache
# at run time (`-v $HOME/.cache/huggingface:/data/hf-cache`).

FROM vllm/vllm-openai:v0.31.0-cu129-ubuntu2404

# ffmpeg libs — transformers 5.x in the vllm base image imports torchcodec
# at import time, which needs libavformat/libavcodec. Minor OS overhead;
# a text-only engine still has to load the image/audio pipeline modules.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install bernoulli + the small bits the base image doesn't ship.
# vllm image provides: torch, vllm, transformers, pydantic, numpy, fastapi.
# We add: pydantic-settings (config layer), uvicorn[standard], scipy (calibrate).
COPY pyproject.toml README.md ./
COPY bernoulli ./bernoulli
COPY evals ./evals

RUN pip install --no-cache-dir \
        "pydantic-settings>=2.5" \
        "uvicorn[standard]>=0.32" \
        "scipy>=1.13" \
    && pip install --no-cache-dir --no-deps -e . \
    && pip uninstall -y torchcodec

# torchcodec ships with transformers 5.x and tries to load libnvrtc.so.13 at
# import time — the vllm base image is cu129 and doesn't have cu13 libs. We
# only need text (tokenizer + logits), not image/video pipelines, so removing
# torchcodec entirely is clean. If image support ever lands in M6 we'll need
# to put it back along with the right CUDA libs.

ENV HF_HOME=/data/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    BERNOULLI_SCORER=vllm \
    BERNOULLI_MAX_MODEL_LEN=8192 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Override the base image's ENTRYPOINT (which runs vLLM's OpenAI server).
ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "bernoulli.server:app", "--host", "0.0.0.0", "--port", "8000"]
