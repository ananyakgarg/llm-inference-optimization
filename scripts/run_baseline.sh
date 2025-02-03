#!/usr/bin/env bash

cd "$(dirname "$0")/.."

echo "Running baseline inference..."
python src/inference.py