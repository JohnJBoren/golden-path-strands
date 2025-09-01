#!/bin/bash

echo "🌟 Golden Path Strands Framework Setup 🌟"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✓ Python $python_version is installed"
else
    echo "✗ Python 3.8+ is required (found $python_version)"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p logs datasets config
echo "✓ Directories created"

# Copy environment file
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created (please update with your settings)"
fi

# Check Ollama installation
echo ""
echo "Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama server is running"
        
        # Check for gpt-oss:20b model
        if ollama list | grep -q "gpt-oss:20b"; then
            echo "✓ gpt-oss:20b model is available"
        else
            echo ""
            echo "⚠ gpt-oss:20b model not found"
            read -p "Would you like to pull the model now? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "Pulling gpt-oss:20b (this may take a while - 20GB)..."
                ollama pull gpt-oss:20b
                echo "✓ Model pulled successfully"
            else
                echo "You can pull it later with: ollama pull gpt-oss:20b"
            fi
        fi
    else
        echo "⚠ Ollama server is not running"
        echo "Start it with: ollama serve"
    fi
else
    echo "⚠ Ollama is not installed"
    echo "Install from: https://ollama.com/download"
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Ensure Ollama is running: ollama serve"
echo "3. Run the framework: python start.py"
echo ""
echo "For interactive mode: python start.py --mode interactive"
echo "For research mode: python start.py --mode research --topic 'Your topic'"
echo "For coding mode: python start.py --mode coding --requirements 'Your requirements'"
echo ""