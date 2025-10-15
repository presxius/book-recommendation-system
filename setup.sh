#!/bin/bash

echo "Setting up Book Recommendation System..."

# Create necessary directories
mkdir -p data
mkdir -p logs

# Check if data files exist
if [ ! -f "data/books.csv" ]; then
    echo "Warning: books.csv not found in data directory"
fi

if [ ! -f "data/ratings.csv" ]; then
    echo "Warning: ratings.csv not found in data directory"
fi

if [ ! -f "data/users.csv" ]; then
    echo "Warning: users.csv not found in data directory"
fi

echo "Setup completed!"