#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/cuda_arch_helper.sh" ]; then
    source "$SCRIPT_DIR/cuda_arch_helper.sh"
fi

INPUT_TARGET="${1:-86}"
RESOLVED_CAP=""
if command -v resolve_cuda_cap &>/dev/null; then
    RESOLVED_CAP=$(resolve_cuda_cap "$INPUT_TARGET" 2>/dev/null || echo "")
fi

if [ -z "$RESOLVED_CAP" ]; then
    echo "❌ Error: Unrecognized GPU target or compute capability: '$INPUT_TARGET'"
    echo "   Supported examples: '5090', '3090 ti', '4090', 'A6000', 'H100', '120', '86'"
    exit 1
fi

echo "=============================================================================="
echo " Building Polyfish Container for GPU: '$INPUT_TARGET' -> sm_${RESOLVED_CAP}"
echo "=============================================================================="

CAP_ARG="--build-arg CUDA_COMPUTE_CAP=$RESOLVED_CAP"

# Keep default image name as henbomb/polyfish:latest unless explicitly provided as $2
IMAGE_TAG="${2:-henbomb/polyfish:latest}"

echo "Target Image Tag: $IMAGE_TAG"
echo "Generating Cargo.docker.toml (stripping tests and examples)..."
trap 'rm -f Cargo.docker.toml' EXIT
sed -e '/^\[\[test\]\]/,$d' -e '/^\[\[example\]\]/,$d' Cargo.toml > Cargo.docker.toml

echo "Building Docker image ($IMAGE_TAG)..."
sudo docker build $CAP_ARG -t "$IMAGE_TAG" -f Dockerfile .

echo "Cleaning up Cargo.docker.toml..."
rm -f Cargo.docker.toml

echo "Pushing Docker image ($IMAGE_TAG)..."
sudo docker push "$IMAGE_TAG"
if [ "$IMAGE_TAG" != "henbomb/polyfish:latest" ]; then
    sudo docker tag "$IMAGE_TAG" henbomb/polyfish:latest
    sudo docker push henbomb/polyfish:latest
fi
echo "✅ Successfully built and pushed $IMAGE_TAG (CUDA sm_${RESOLVED_CAP})"
