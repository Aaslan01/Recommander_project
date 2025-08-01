# 🚀 Recommender System Project - Running Instructions

## 📋 Project Overview
This is a hybrid movie recommender system that combines collaborative filtering with content-based features and sentiment analysis. The project implements and compares different recommendation approaches using the MovieLens dataset.

## 🛠️ Prerequisites
- Python 3.8+
- Virtual environment (venv)
- Required dependencies (see requirements.txt)

## 📦 Installation & Setup

### 1. Environment Setup
```bash
# Navigate to project directory
cd "/Users/jasmeenkaurmalhotra/Documents/Term 3/Social Media/Recommander_project"

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn torch matplotlib
```

### 2. Verify Installation
```bash
# Check if all dependencies are installed
python -c "import pandas, numpy, sklearn, torch, matplotlib; print('✅ All dependencies installed successfully!')"
```

## 🎯 Running the Project

### Step 1: Generate Sentiment Data
```bash
# Generate synthetic sentiment scores for all movies
python sentiment_analysis.py
```

**Expected Output:**
```
Generating sentiment scores for 1682 movies...
✅ Generated sentiment scores for 1682 movies
📊 Sentiment range: 0.256 to 0.858
📈 Average sentiment: 0.600
💾 Sentiment scores saved to data/sentiment_scores.csv
```

### Step 2: Run Main Experiments
```bash
# Run comparison between Matrix Factorization and Hybrid Recommender
python run_experiments.py
```

**Expected Output:**
```
Running Matrix Factorization...
Iteration: 1, RMSE: 2.4743
...
Final RMSE (MF): 0.8600

Running Hybrid Recommender...
Hybrid - RMSE: 0.9429, MAE: 0.7455, Precision: 0.83, Recall: 0.37, F1: 0.51
```

### Step 3: Run Comprehensive Experiments
```bash
# Run experiments with different latent factors and generate plots
python plot_runner.py
```

**Expected Output:**
```
Running MF with k=20
...
Running Hybrid with k=20
...
[Plots generated: rmse_comparison.png]
```

## 📊 Understanding the Results

### Performance Metrics
- **RMSE (Root Mean Square Error)**: Lower is better
- **MAE (Mean Absolute Error)**: Lower is better
- **Precision**: Higher is better (accuracy of positive predictions)
- **Recall**: Higher is better (coverage of positive cases)
- **F1-Score**: Harmonic mean of precision and recall

### Model Comparison
1. **Matrix Factorization**: Traditional collaborative filtering
2. **Hybrid Recommender**: Combines CF + content features + sentiment

## 🗂️ Project Structure

```
Recommander_project/
├── data/
│   ├── u.data              # User-item ratings
│   ├── u.item              # Movie metadata
│   ├── sentiment_scores.csv # Generated sentiment data
│   └── u.genre             # Genre definitions
├── sentiment_analysis.py   # Generate synthetic sentiment
├── run_experiments.py      # Main experiment runner
├── plot_runner.py          # Comprehensive experiments
├── hybrid_recommender.py   # Neural network model
├── cf_matrix_factorization.py # Matrix factorization
├── data_preprocessing.py   # Data loading and preparation
├── evaluation.py           # Performance metrics
├── topk_evaluation.py      # Top-K evaluation
├── plot_results.py         # Visualization
└── requirements.txt        # Dependencies
```

## 🔧 Individual Scripts Explained

### Core Scripts
- **`sentiment_analysis.py`**: Generates synthetic sentiment scores based on movie characteristics
- **`run_experiments.py`**: Main experiment comparing Matrix Factorization vs Hybrid model
- **`plot_runner.py`**: Runs comprehensive experiments with different hyperparameters

### Supporting Scripts
- **`data_preprocessing.py`**: Loads and merges ratings, metadata, and sentiment data
- **`hybrid_recommender.py`**: PyTorch neural network for hybrid recommendations
- **`cf_matrix_factorization.py`**: Traditional collaborative filtering implementation
- **`evaluation.py`**: Calculates RMSE, MAE, Precision, Recall, F1 metrics
- **`topk_evaluation.py`**: Top-K recommendation evaluation (Precision@K, Recall@K)
- **`plot_results.py`**: Creates comparison plots and visualizations

## 📈 Expected Results

### Matrix Factorization Performance
- **RMSE**: ~0.85-0.87
- **Convergence**: 10 iterations typically sufficient

### Hybrid Recommender Performance
- **RMSE**: ~0.94-0.96
- **Precision**: ~0.80-0.85
- **Recall**: ~0.35-0.40
- **F1-Score**: ~0.50-0.55

### Latent Factor Comparison
- **k=20**: Fastest training, moderate performance
- **k=30**: Good balance of speed and accuracy
- **k=40**: Better accuracy, slower training
- **k=50**: Best accuracy, slowest training

## 🎨 Output Files

### Generated Files
- `data/sentiment_scores.csv`: Synthetic sentiment scores for all movies
- `rmse_comparison.png`: Performance comparison plot
- `__pycache__/`: Python cache files (can be ignored)

### Data Files
- `data/u.data`: Original MovieLens ratings (100K ratings)
- `data/u.item`: Movie metadata with 19 genre columns
- `data/u.genre`: Genre definitions

## 🔍 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated
2. **Memory Issues**: Reduce batch size in `run_experiments.py`
3. **Slow Training**: Reduce iterations or latent factors
4. **Missing Data**: Ensure all data files are in `data/` directory

### Performance Tips
- Use GPU if available for faster neural network training
- Adjust batch size based on available memory
- Reduce iterations for quick testing
- Use smaller latent factors for faster experiments

## 🚀 Quick Start (One Command)
```bash
# Complete pipeline in one go
source venv/bin/activate && python sentiment_analysis.py && python run_experiments.py && python plot_runner.py
```

## 📝 Notes
- Sentiment scores are synthetic (generated, not real Twitter data)
- For real sentiment collection, use `real_sentiment_collector.py` (requires Twitter API)
- Results may vary slightly due to random initialization
- The hybrid model combines multiple features for better recommendations

## 🎯 Next Steps
1. Experiment with different hyperparameters
2. Try real sentiment data collection
3. Implement additional evaluation metrics
4. Add more content features
5. Optimize model architecture

---
**Happy Recommending! 🎬📊** 