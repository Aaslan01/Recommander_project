# final_pipeline.py
import os
import numpy as np
import pandas as pd
import torch

# Import your project modules (they must be in same folder / PYTHONPATH)
from data_preprocessing import load_data
from cf_matrix_factorization import MatrixFactorization
from run_experiments import train_hybrid_model

# -------------------------
# Config
# -------------------------
mf_K = 30            # latent factors for MF
mf_alpha = 0.01
mf_beta = 0.01
mf_iters = 10        # iterations for MF training (lower for quick tests)

hybrid_embed_dim = 30
hybrid_epochs = 5    # reduce for quick tests

top_k = 10

data_dir = "data"
sentiment_path = os.path.join(data_dir, "sentiment_scores.csv")
u_item_path = os.path.join(data_dir, "u.item")

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cpu")
# -------------------------
# Load preprocessed merged data
# data_preprocessing.load_data returns merged dataset with columns:
#   user_id, item_id, rating, timestamp, genre cols, sentiment, user, item (encoded)
# -------------------------
print("Loading and preprocessing data...")
merged, user_enc, item_enc = load_data()  # uses data/u.data, data/u.item, data/sentiment_scores.csv
# merged: pandas DataFrame; has columns 'user' (encoded), 'item' (encoded), 'rating', 'item_id' original, etc.

# we need number of users and items (encoded)
num_users = merged['user'].nunique()
num_items = merged['item'].nunique()
print(f" - users: {num_users}, items: {num_items}, ratings rows: {len(merged)}")

# -------------------------
# MATRIX FACTORIZATION
# -------------------------
print("\n==> Training Matrix Factorization (MF) ...")
# Build rating matrix R (users x items), fill zeros where no rating
R = np.zeros((num_users, num_items), dtype=np.float32)
for row in merged.itertuples(index=False):
    # merged rows: include 'user' and 'item' and 'rating'
    R[row.user, row.item] = row.rating

mf = MatrixFactorization(R, K=mf_K, alpha=mf_alpha, beta=mf_beta, iterations=mf_iters)
mf.train()
mf_pred_full = mf.full_matrix()  # shape: (num_users, num_items)

# Compute per-item average predicted rating (global)
mf_item_avg = np.mean(mf_pred_full, axis=0)  # length num_items
# Build dataframe: encoded_item -> avg_pred
mf_item_df = pd.DataFrame({
    'item_encoded': np.arange(num_items),
    'mf_pred_avg': mf_item_avg
})
# Map back to original item_id (u.item movie id)
# merged has mapping from encoded item to original item_id; get unique mapping
enc_to_orig = merged[['item', 'item_id']].drop_duplicates().set_index('item')['item_id'].to_dict()
mf_item_df['item_id'] = mf_item_df['item_encoded'].map(enc_to_orig)

# -------------------------
# HYBRID RECOMMENDER
# -------------------------
print("\n==> Training Hybrid Recommender ...")
hybrid_model = train_hybrid_model(merged, embed_dim=hybrid_embed_dim, epochs=hybrid_epochs)
hybrid_model.eval()

# Prepare tensors for every rating row in merged dataset (we'll average predictions per item)
with torch.no_grad():
    users_tensor = torch.LongTensor(merged['user'].values).to(device)
    items_tensor = torch.LongTensor(merged['item'].values).to(device)
    genres_tensor = torch.FloatTensor(merged.iloc[:, 5:24].values).to(device)  # data_preprocessing used cols 5..23 for genres
    sentiments_tensor = torch.FloatTensor(merged['sentiment'].values).to(device)

    preds = hybrid_model(users_tensor, items_tensor, genres_tensor, sentiments_tensor).cpu().numpy() * 5.0
    # note: training scaled rating by /5 (in earlier code), so multiply back

# Merge preds with item encoded IDs and compute per-item mean predicted rating
hybrid_pred_df = pd.DataFrame({
    'item': merged['item'].values,
    'item_id': merged['item_id'].values,
    'pred': preds
})
hybrid_item_avg = hybrid_pred_df.groupby('item').pred.mean().reset_index().rename(columns={'pred': 'hybrid_pred_avg'})
# map encoded->orig id
hybrid_item_avg['item_id'] = hybrid_item_avg['item'].map(enc_to_orig)

# -------------------------
# Create combined table with titles and sentiment
# -------------------------
# Load u.item to get titles
uitem = pd.read_csv(u_item_path, sep='|', header=None, encoding='latin-1')
uitem = uitem[[0, 1]]  # item_id, title
uitem.columns = ['item_id', 'title']

# sentiment file
if os.path.exists(sentiment_path):
    sentiment_df = pd.read_csv(sentiment_path)
else:
    print("Warning: sentiment_scores.csv not found. Creating placeholder with neutral sentiment.")
    sentiment_df = pd.DataFrame({'item_id': uitem['item_id'], 'sentiment': 0.5})

# Assemble MF top scores
mf_item_df = mf_item_df.merge(uitem, on='item_id', how='left')
mf_item_df = mf_item_df.merge(sentiment_df, on='item_id', how='left')
# Assemble Hybrid top scores
hybrid_item_avg = hybrid_item_avg.merge(uitem, on='item_id', how='left')
hybrid_item_avg = hybrid_item_avg.merge(sentiment_df, on='item_id', how='left')

# Some items may be missing sentiment -> fill neutral
mf_item_df['sentiment'] = mf_item_df['sentiment'].fillna(0.5)
hybrid_item_avg['sentiment'] = hybrid_item_avg['sentiment'].fillna(0.5)

# -------------------------
# Top-K extraction
# -------------------------
mf_top = mf_item_df.sort_values('mf_pred_avg', ascending=False).head(top_k)[['item_id', 'title', 'mf_pred_avg', 'sentiment']]
hybrid_top = hybrid_item_avg.sort_values('hybrid_pred_avg', ascending=False).head(top_k)[['item_id', 'title', 'hybrid_pred_avg', 'sentiment']]

# Save top lists
mf_top.to_csv(os.path.join(output_dir, f"mf_top{top_k}.csv"), index=False)
hybrid_top.to_csv(os.path.join(output_dir, f"hybrid_top{top_k}.csv"), index=False)

print(f"\nSaved top-{top_k} lists to {output_dir}/")

# -------------------------
# Comparison
# -------------------------
mf_set = set(mf_top['item_id'].values)
hybrid_set = set(hybrid_top['item_id'].values)
overlap = mf_set & hybrid_set
only_mf = mf_set - hybrid_set
only_hybrid = hybrid_set - mf_set

def avg_sent_for(item_ids):
    if not item_ids:
        return None
    s = sentiment_df[sentiment_df['item_id'].isin(item_ids)]['sentiment']
    return round(s.mean(), 3) if len(s) > 0 else None

print("\n==== TOP-10 COMPARISON ====")
print(f"Overlap ({len(overlap)} items): {overlap}")
print("Average sentiment of overlap:", avg_sent_for(overlap))
print()
print(f"Only in MF ({len(only_mf)} items): {only_mf}")
print("Average sentiment only in MF:", avg_sent_for(only_mf))
print()
print(f"Only in Hybrid ({len(only_hybrid)} items): {only_hybrid}")
print("Average sentiment only in Hybrid:", avg_sent_for(only_hybrid))
print("\nDetailed top-10 files saved; open the CSVs for titles + scores + sentiment.")

# Also print readable versions
print("\nMF Top-10:")
print(mf_top.to_string(index=False))
print("\nHybrid Top-10:")
print(hybrid_top.to_string(index=False))

# Done
print("\nAll finished.")
