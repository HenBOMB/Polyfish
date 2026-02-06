#!/bin/bash
MESSAGE="commit"

while getopts "m:" opt; do
  case $opt in
    m) MESSAGE="$OPTARG" ;;
    *) echo "Usage: $0 [-m message]" >&2; exit 1 ;;
  esac
done

git add . && git commit -m "$MESSAGE" && git push
