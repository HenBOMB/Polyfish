#!/bin/bash
set -e

echo "Generating Cargo.docker.toml (stripping tests and examples)..."
sed -e '/^\[\[test\]\]/,$d' -e '/^\[\[example\]\]/,$d' Cargo.toml > Cargo.docker.toml

CAP_ARG=""
if [ -n "$1" ]; then
    CAP_ARG="--build-arg CUDA_COMPUTE_CAP=$1"
    echo "Building for compute capability: $1"
fi

IMAGE_TAG="${2:-henbomb/polyzero:v0.1}"

echo "Building Docker image ($IMAGE_TAG)..."
docker build $CAP_ARG -t "$IMAGE_TAG" -f Dockerfile .

echo "Cleaning up Cargo.docker.toml..."
rm Cargo.docker.toml

echo "Pushing Docker image ($IMAGE_TAG)..."
docker push "$IMAGE_TAG"
