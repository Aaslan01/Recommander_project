
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

def generate_sentiment_for_all_movies():
    # Load all movie IDs from u.item
    movies_df = pd.read_csv('data/u.item', sep='|', header=None, encoding='latin-1')
    all_movie_ids = movies_df[0].tolist()  # First column contains movie IDs
    
    print(f"Generating sentiment scores for {len(all_movie_ids)} movies...")
    
    # Generate synthetic sentiment scores
    sentiment_data = []
    for mid in all_movie_ids:
        # Generate more realistic sentiment scores based on movie characteristics
        # Action/Thriller movies tend to have higher sentiment
        # Drama movies might have more varied sentiment
        # Comedy movies generally have positive sentiment
        
        # Get movie title to make sentiment more realistic
        movie_title = movies_df[movies_df[0] == mid][1].iloc[0]
        
        # Generate sentiment based on movie characteristics
        if any(word in movie_title.lower() for word in ['comedy', 'funny', 'humor', 'joke']):
            sentiment_score = round(random.uniform(0.6, 0.9), 3)  # Comedy = positive
        elif any(word in movie_title.lower() for word in ['action', 'adventure', 'thriller', 'war']):
            sentiment_score = round(random.uniform(0.5, 0.8), 3)  # Action = moderately positive
        elif any(word in movie_title.lower() for word in ['drama', 'serious', 'dark', 'sad']):
            sentiment_score = round(random.uniform(0.3, 0.7), 3)  # Drama = varied
        elif any(word in movie_title.lower() for word in ['horror', 'scary', 'fear', 'terror']):
            sentiment_score = round(random.uniform(0.2, 0.6), 3)  # Horror = lower sentiment
        else:
            sentiment_score = round(random.uniform(0.4, 0.8), 3)  # Default range
        
        sentiment_data.append({'item_id': mid, 'sentiment': sentiment_score})
    
    df = pd.DataFrame(sentiment_data)
    df.to_csv('data/sentiment_scores.csv', index=False)
    print(f"✅ Generated sentiment scores for {len(sentiment_data)} movies")
    print(f"📊 Sentiment range: {df['sentiment'].min():.3f} to {df['sentiment'].max():.3f}")
    print(f"📈 Average sentiment: {df['sentiment'].mean():.3f}")
    print("💾 Sentiment scores saved to data/sentiment_scores.csv")

if __name__ == "__main__":
    generate_sentiment_for_all_movies()
