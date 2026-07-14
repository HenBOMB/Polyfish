#!/bin/bash
set -e

echo "Generating Cargo.docker.toml (stripping tests and examples)..."
sed '/^\[\[test\]\]/,$d' Cargo.toml > Cargo.docker.toml

echo "Building Docker image..."
docker build -t henbomb/polyzero:v0.1 -f Dockerfile .

echo "Cleaning up Cargo.docker.toml..."
rm Cargo.docker.toml

echo "Pushing Docker image..."
docker push henbomb/polyzero:v0.1
