#!/bin/bash
set -e

echo "Generating Cargo.docker.toml (stripping tests and examples)..."
sed '/^\[\[test\]\]/,$d' Cargo.toml > Cargo.docker.toml

CAP_ARG=""
if [ -n "$1" ]; then
    CAP_ARG="--build-arg CUDA_COMPUTE_CAP=$1"
    echo "Building for compute capability: $1"
fi

echo "Building Docker image..."
docker build $CAP_ARG -t henbomb/polyzero:v0.1 -f Dockerfile .

echo "Cleaning up Cargo.docker.toml..."
rm Cargo.docker.toml

echo "Pushing Docker image..."
docker push henbomb/polyzero:v0.1
