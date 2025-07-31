
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data():
    ratings = pd.read_csv('data/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    items = pd.read_csv('data/u.item', sep='|', encoding='latin-1', header=None)
    sentiments = pd.read_csv('data/sentiment_scores.csv')

    items = items[[0] + list(range(5, 24))]  # movie id + 19 genre cols
    items.columns = ['item_id'] + [f'genre_{i}' for i in range(19)]

    merged = pd.merge(ratings, items, on='item_id')
    merged = pd.merge(merged, sentiments, on='item_id', how='left')
    merged['sentiment'] = merged['sentiment'].fillna(0.5)

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    merged['user'] = user_enc.fit_transform(merged['user_id'])
    merged['item'] = item_enc.fit_transform(merged['item_id'])

    return merged, user_enc, item_enc
