#!/bin/bash

echo "Starting Book Recommendation System on Railway..."

# Run the application with Gunicorn
exec gunicorn --config gunicorn.conf.py app:app