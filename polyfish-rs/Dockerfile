# ==========================================
# STAGE 1: The Heavy Builder (thrown away)
# ==========================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential curl pkg-config libssl-dev git && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
# We only need the rust source code to build the binaries for prod
COPY Cargo.toml Cargo.lock ./
# Strip examples/tests that require local test fixtures or Apple Metal
RUN sed -i -e '/^\[\[test\]\]/,$d' -e '/^\[\[example\]\]/,$d' Cargo.toml
COPY src ./src

# Bypass nvidia-smi check during Docker build by hardcoding the compute capability
# Default is 86 (Ampere, RTX 3090/A100). For RTX 4090, pass --build-arg CUDA_COMPUTE_CAP=89
ARG CUDA_COMPUTE_CAP=86
ENV CUDA_COMPUTE_CAP=$CUDA_COMPUTE_CAP

RUN mkdir -p /app/target/bin_sm${CUDA_COMPUTE_CAP} && \
    CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
    cargo build --bin polyfish --bin self_play --bin arena --bin benchmark --release --no-default-features --features cuda && \
    cp /app/target/release/polyfish /app/target/release/self_play /app/target/release/arena /app/target/release/benchmark /app/target/bin_sm${CUDA_COMPUTE_CAP}/ && \
    echo "$CUDA_COMPUTE_CAP" > /app/target/release/.cuda_cap

# ==========================================
# STAGE 2: The Slim Runtime (the final image)
# ==========================================
# We MUST use 'devel' here. The original Dockerfile tried to be clever by using
# 'runtime' to save space, but this silently broke the Rust CUDA bindings 
# (which require NVRTC and unversioned .so symlinks). 
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV POLYFISH_ADVERSARIAL_SEARCH=1
ENV ITER_OFFSET=76

# Install Python 3.11 and minimal runtime tools
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev jq tmux git libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

# Create venv and install python packages without cache bloat
COPY requirements.txt ./
RUN python3.11 -m venv .venv && \
    .venv/bin/pip install --no-cache-dir --upgrade pip && \
    .venv/bin/pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cu128 && \
    .venv/bin/pip install --no-cache-dir -r requirements.txt && \
    .venv/bin/pip install --no-cache-dir supabase python-dotenv

# Copy compiled binaries and architecture caches from the builder stage
COPY --from=builder /app/target/release/polyfish ./target/release/polyfish
COPY --from=builder /app/target/release/self_play ./target/release/self_play
COPY --from=builder /app/target/release/arena ./target/release/arena
COPY --from=builder /app/target/release/benchmark ./target/release/benchmark
COPY --from=builder /app/target/release/.cuda_cap ./target/release/.cuda_cap
COPY --from=builder /app/target/bin_sm* ./target/

# Copy your bash scripts, python files, and foundation weights
COPY *.sh *.py ./
COPY model_dagger_foundation.safetensors* ./
RUN if [ -f model_dagger_foundation.safetensors ]; then cp model_dagger_foundation.safetensors model.safetensors; fi

RUN chmod +x run_training_runpod.sh run_training_loop.sh rate_checkpoints.sh

ENTRYPOINT ["./run_training_runpod.sh"]
