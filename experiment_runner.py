
from data_preprocessing import load_data
from cf_matrix_factorization import MatrixFactorization
from run_experiments import train_hybrid_model
from evaluation import evaluate_regression
import torch
import numpy as np

def run_latent_factor_experiment(factors=[20, 30, 40, 50]):
    data, _, _ = load_data()
    mf_rmses = []
    hybrid_rmses = []

    R = np.zeros((data['user'].nunique(), data['item'].nunique()))
    for row in data.itertuples():
        R[row.user, row.item] = row.rating

    for k in factors:
        print(f"Running MF with k={k}")
        mf = MatrixFactorization(R, K=k, alpha=0.01, beta=0.01, iterations=10)
        mf.train()
        mf_rmses.append(mf.rmse())

        print(f"Running Hybrid with k={k}")
        hybrid_model = train_hybrid_model(data, embed_dim=k)
        hybrid_model.eval()
        with torch.no_grad():
            user = torch.LongTensor(data['user'].values)
            item = torch.LongTensor(data['item'].values)
            genre = torch.FloatTensor(data.iloc[:, 5:24].values)
            sent = torch.FloatTensor(data['sentiment'].values)
            y_true = data['rating'].values
            y_pred = hybrid_model(user, item, genre, sent).numpy() * 5
        rmse_h, _ = evaluate_regression(y_true, y_pred)
        hybrid_rmses.append(rmse_h)

    return factors, mf_rmses, hybrid_rmses
