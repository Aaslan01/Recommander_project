
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import load_data
from hybrid_recommender import HybridRecommender
from cf_matrix_factorization import MatrixFactorization
from evaluation import evaluate_classification, evaluate_regression

def train_hybrid_model(data, embed_dim, epochs=5):
    model = HybridRecommender(num_users=data['user'].nunique(), num_items=data['item'].nunique(), num_genres=19, embed_dim=embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    X_user = torch.LongTensor(data['user'].values)
    X_item = torch.LongTensor(data['item'].values)
    X_genre = torch.FloatTensor(data.iloc[:, 5:24].values)
    X_sentiment = torch.FloatTensor(data['sentiment'].values)
    y = torch.FloatTensor(data['rating'].values / 5.0)

    dataset = TensorDataset(X_user, X_item, X_genre, X_sentiment, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for user, item, genre, sent, label in loader:
            optimizer.zero_grad()
            preds = model(user, item, genre, sent)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
    return model

def run_all():
    data, _, _ = load_data()
    print("\nRunning Matrix Factorization...")
    R = np.zeros((data['user'].nunique(), data['item'].nunique()))
    for row in data.itertuples():
        R[row.user, row.item] = row.rating
    mf = MatrixFactorization(R, K=30, alpha=0.01, beta=0.01, iterations=10)
    mf.train()
    rmse = mf.rmse()
    print(f"Final RMSE (MF): {rmse:.4f}")

    print("\nRunning Hybrid Recommender...")
    hybrid_model = train_hybrid_model(data, embed_dim=30)
    hybrid_model.eval()
    with torch.no_grad():
        user = torch.LongTensor(data['user'].values)
        item = torch.LongTensor(data['item'].values)
        genre = torch.FloatTensor(data.iloc[:, 5:24].values)
        sent = torch.FloatTensor(data['sentiment'].values)
        y_true = data['rating'].values
        y_pred = hybrid_model(user, item, genre, sent).numpy() * 5
    rmse_h, mae_h = evaluate_regression(y_true, y_pred)
    prec, rec, f1 = evaluate_classification((y_true >= 4).astype(int), (y_pred >= 4).astype(int))
    print(f"Hybrid - RMSE: {rmse_h:.4f}, MAE: {mae_h:.4f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

if __name__ == "__main__":
    run_all()
