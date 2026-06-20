#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "🚀 Starting TrafficLens backend..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
