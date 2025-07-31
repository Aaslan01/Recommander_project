
import pandas as pd
import random

def generate_fake_sentiment(movie_ids):
    sentiment_data = []
    for mid in movie_ids:
        sentiment_score = round(random.uniform(0.4, 0.9), 2)
        sentiment_data.append({'item_id': mid, 'sentiment': sentiment_score})
    df = pd.DataFrame(sentiment_data)
    df.to_csv('data/sentiment_scores.csv', index=False)
    print("Fake sentiment scores saved to data/sentiment_scores.csv")
