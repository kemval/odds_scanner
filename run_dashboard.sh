#!/bin/bash

# Sports Betting Market Analysis Dashboard Launcher
# This script launches the Streamlit dashboard

echo "🚀 Launching Sports Betting Market Analysis Dashboard..."
echo ""

# Check if data exists
if [ ! -f "data/processed/odds_processed.csv" ]; then
    echo "⚠️  Warning: Processed data not found!"
    echo "   Run the following commands first:"
    echo "   1. python src/data_collection.py"
    echo "   2. python src/data_processing.py"
    echo "   3. python src/inefficiency_scanner.py"
    echo "   4. python src/backtesting.py"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Launch Streamlit
echo "📊 Starting dashboard on http://localhost:8501"
echo ""
streamlit run app.py
