# Makefile for Recommander Project

.PHONY: help install test run clean setup

# Default target
help:
	@echo "🎬 Recommander Project - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make install    - Install dependencies"
	@echo "  make setup      - Set up virtual environment and install dependencies"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test       - Run project tests"
	@echo ""
	@echo "🚀 Running:"
	@echo "  make run        - Run complete pipeline"
	@echo "  make sentiment  - Generate sentiment data"
	@echo "  make experiments - Run experiments"
	@echo "  make plots      - Generate plots"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  make clean      - Clean cache and temporary files"
	@echo "  make help       - Show this help message"

# Setup virtual environment and install dependencies
setup:
	@echo "🔧 Setting up virtual environment..."
	python3 -m venv venv
	@echo "📦 Installing dependencies..."
	venv/bin/pip install -r requirements.txt
	@echo "✅ Setup complete! Activate with: source venv/bin/activate"

# Install dependencies (assumes venv is already activated)
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Run tests
test:
	@echo "🧪 Running project tests..."
	python test_project.py

# Run complete pipeline
run:
	@echo "🚀 Running complete pipeline..."
	python final_pipeline.py

# Generate sentiment data
sentiment:
	@echo "📊 Generating sentiment data..."
	python sentiment_analysis.py

# Run experiments
experiments:
	@echo "🧪 Running experiments..."
	python run_experiments.py

# Generate plots
plots:
	@echo "📈 Generating plots..."
	python plot_runner.py

# Clean cache and temporary files
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*.pyd" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "htmlcov" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Quick start - setup and run
quickstart: setup run
	@echo "🎉 Quick start complete!"

# Development setup
dev: setup
	@echo "🔧 Installing development dependencies..."
	venv/bin/pip install -e .
	@echo "✅ Development setup complete!" 