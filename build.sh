#!/bin/bash
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p data

echo "Build completed successfully!"