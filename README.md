# 🎬 Recommander Project

A comprehensive hybrid movie recommender system that combines collaborative filtering with content-based features and sentiment analysis.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Aaslan01/Recommander_project.git
cd Recommander_project

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python final_pipeline.py
```

## 📋 Project Overview

This project implements and compares different recommendation approaches:

1. **Matrix Factorization (MF)**: Traditional collaborative filtering using SVD
2. **Hybrid Recommender**: Neural network combining CF + content features + sentiment
3. **Final Pipeline**: Complete end-to-end system generating top-10 recommendations

## 🛠️ Features

- **Data Processing**: MovieLens 100K dataset with 19 genre features
- **Sentiment Analysis**: Synthetic sentiment scores for enhanced recommendations
- **Performance Evaluation**: RMSE, MAE, Precision, Recall, F1 metrics
- **Visualization**: Performance comparison plots
- **Top-K Recommendations**: Generate personalized top-10 movie lists

## 📊 Results

The system generates two output files:
- `outputs/mf_top10.csv`: Matrix Factorization top recommendations
- `outputs/hybrid_top10.csv`: Hybrid model top recommendations

## 📁 Project Structure

```
├── final_pipeline.py          # 🆕 Complete end-to-end pipeline
├── data_preprocessing.py      # Data loading and preparation
├── cf_matrix_factorization.py # Matrix factorization implementation
├── hybrid_recommender.py      # Neural network hybrid model
├── sentiment_analysis.py      # Sentiment score generation
├── run_experiments.py         # Model comparison experiments
├── plot_runner.py             # Comprehensive experiments
├── evaluation.py              # Performance metrics
├── topk_evaluation.py         # Top-K evaluation
├── data/                      # MovieLens dataset
│   ├── u.data                # User-item ratings
│   ├── u.item                # Movie metadata
│   └── sentiment_scores.csv  # Generated sentiment data
└── outputs/                   # Generated recommendations
```

## 🔧 Configuration

Key parameters in `final_pipeline.py`:
- `mf_K = 30`: Latent factors for Matrix Factorization
- `hybrid_embed_dim = 30`: Embedding dimension for hybrid model
- `top_k = 10`: Number of top recommendations to generate

## 📈 Performance

- **Matrix Factorization**: RMSE ~0.85-0.87
- **Hybrid Model**: RMSE ~0.94-0.96, Precision ~0.80-0.85
- **Training Time**: ~1-2 minutes for complete pipeline

## 🎯 Use Cases

- Movie recommendation systems
- Content-based filtering research
- Hybrid recommendation algorithms
- Sentiment-aware recommendations
- Performance comparison studies

## 📚 Dependencies

- Python 3.8+
- PyTorch, NumPy, Pandas
- Scikit-learn, Matplotlib
- See `requirements.txt` for complete list

## 🚀 Running Options

### Option 1: Complete Pipeline (Recommended)
```bash
python final_pipeline.py
```

### Option 2: Individual Components
```bash
python sentiment_analysis.py    # Generate sentiment data
python run_experiments.py       # Run model comparison
python plot_runner.py           # Generate performance plots
```

## 📖 Documentation

- **RUN_INSTRUCTIONS.md**: Detailed setup and usage instructions
- **Code Comments**: Comprehensive inline documentation
- **Output Files**: Self-explanatory CSV results with movie titles and scores

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for the Social Media course at University of Toronto**

*Last updated: December 2024*
