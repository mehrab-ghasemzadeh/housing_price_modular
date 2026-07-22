#!/bin/bash
# start.sh

# Run the regression script first
python reg.py

# Then start the API
exec uvicorn api:app --host 0.0.0.0 --port 8050