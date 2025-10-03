#!/bin/bash

# Laptop Price Prediction - Run Script

echo "================================"
echo "Laptop Price Prediction Portfolio"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Run the Streamlit app
echo ""
echo "Starting Streamlit application..."
echo "Open your browser and navigate to http://localhost:8501"
echo ""
streamlit run app.py
