
import torch
import torch.nn as nn

class HybridRecommender(nn.Module):
    def __init__(self, num_users, num_items, num_genres, embed_dim):
        super(HybridRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.genre_embedding = nn.Linear(num_genres, embed_dim)
        self.sentiment_dense = nn.Linear(1, embed_dim)
        self.output = nn.Linear(embed_dim * 4, 1)

    def forward(self, user_ids, item_ids, genres, sentiments):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        genre_emb = self.genre_embedding(genres.float())
        sentiment_emb = self.sentiment_dense(sentiments.view(-1, 1))
        concat = torch.cat([user_emb, item_emb, genre_emb, sentiment_emb], dim=1)
        return torch.sigmoid(self.output(concat)).squeeze()
