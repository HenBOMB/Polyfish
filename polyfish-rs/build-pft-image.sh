#!/bin/bash
set -e

echo "Generating Cargo.docker.toml (stripping tests and examples)..."
sed -e '/^\[\[test\]\]/,$d' -e '/^\[\[example\]\]/,$d' Cargo.toml > Cargo.docker.toml

CAP_ARG=""
if [ -n "$1" ]; then
    CAP_ARG="--build-arg CUDA_COMPUTE_CAP=$1"
    echo "Building for compute capability: $1"
fi

IMAGE_TAG="${2:-henbomb/polyfish:latest}"

echo "Building Docker image ($IMAGE_TAG)..."
sudo docker build $CAP_ARG -t "$IMAGE_TAG" -f Dockerfile .

echo "Cleaning up Cargo.docker.toml..."
rm -f Cargo.docker.toml

echo "Pushing Docker image ($IMAGE_TAG)..."
sudo docker tag "$IMAGE_TAG" henbomb/polyfish:latest
sudo docker push "$IMAGE_TAG"


