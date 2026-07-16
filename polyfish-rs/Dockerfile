# ==========================================
# STAGE 1: The Heavy Builder (thrown away)
# ==========================================
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential curl pkg-config libssl-dev git && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
# We only need the rust source code to build the binaries for prod
COPY Cargo.docker.toml Cargo.toml
COPY Cargo.lock ./
COPY src ./src

# Bypass nvidia-smi check during Docker build by hardcoding the Ampere compute capability (8.6)
ENV CUDA_COMPUTE_CAP=86

RUN CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
    cargo build --bin polyfish --bin self_play --bin arena --bin benchmark --release --no-default-features --features cuda

# ==========================================
# STAGE 2: The Slim Runtime (the final image)
# ==========================================
# We MUST use 'devel' here. The original Dockerfile tried to be clever by using
# 'runtime' to save space, but this silently broke the Rust CUDA bindings 
# (which require NVRTC and unversioned .so symlinks). 
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and minimal runtime tools
RUN apt-get update && apt-get install -y \
    python3-venv python3-pip jq tmux git libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv and install python packages without cache bloat
COPY requirements.txt ./
RUN python3 -m venv .venv && \
    .venv/bin/pip install --no-cache-dir --upgrade pip && \
    .venv/bin/pip install --no-cache-dir --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu124 && \
    .venv/bin/pip install --no-cache-dir -r requirements.txt && \
    .venv/bin/pip install --no-cache-dir supabase python-dotenv

# Copy ONLY the compiled binaries from the builder stage! (Saves ~2GB of Rust cache)
COPY --from=builder /app/target/release/polyfish ./target/release/polyfish
COPY --from=builder /app/target/release/self_play ./target/release/self_play
COPY --from=builder /app/target/release/arena ./target/release/arena
COPY --from=builder /app/target/release/benchmark ./target/release/benchmark

# Copy your bash scripts and python files
COPY *.sh *.py ./

RUN chmod +x run_training_runpod.sh run_training_loop.sh rate_checkpoints.sh

ENTRYPOINT ["./run_training_runpod.sh"]
