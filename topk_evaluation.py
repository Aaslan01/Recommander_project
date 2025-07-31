
import numpy as np

def precision_at_k(y_true, y_scores, k):
    precision_list = []
    for user in np.unique(y_true[:, 0]):
        user_data = y_true[y_true[:, 0] == user]
        user_scores = y_scores[y_true[:, 0] == user]
        top_k_idx = np.argsort(user_scores)[-k:]
        hits = np.sum(user_data[top_k_idx, 2] >= 4)  # assume relevant if rating >= 4
        precision_list.append(hits / k)
    return np.mean(precision_list)

def recall_at_k(y_true, y_scores, k):
    recall_list = []
    for user in np.unique(y_true[:, 0]):
        user_data = y_true[y_true[:, 0] == user]
        user_scores = y_scores[y_true[:, 0] == user]
        rel_items = np.sum(user_data[:, 2] >= 4)
        if rel_items == 0:
            continue
        top_k_idx = np.argsort(user_scores)[-k:]
        hits = np.sum(user_data[top_k_idx, 2] >= 4)
        recall_list.append(hits / rel_items)
    return np.mean(recall_list)
