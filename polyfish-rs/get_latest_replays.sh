#!/bin/bash

# Default version
VERSION="115"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -V|--version)
      VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      shift # past argument
      ;;
  esac
done

echo "Searching for replays with version: $VERSION"
cd ./replays
grep -l "\"version\":$VERSION" *.json | head -n 10