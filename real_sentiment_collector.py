import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import os
from tweepy.errors import TooManyRequests

# Paste your valid bearer token here
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAKBi3QEAAAAAbJQuA2OJ5XKhUfEC8i0zbjKlnhk%3DJbZYBZrv8ssKIZ3m9M1fuAUcCuIDU7andWFZOjFMRo0QlPxyuR'

client = tweepy.Client(bearer_token=BEARER_TOKEN)
analyzer = SentimentIntensityAnalyzer()
print("Connected to Twitter")

# Load movie data
df = pd.read_csv("data/u.item", sep="|", header=None, encoding='latin-1')
print(f"Loaded {len(df)} movies from u.item")

# Load existing sentiment scores if they exist (resume mode)
output_path = "data/sentiment_scores.csv"
if os.path.exists(output_path):
    done_df = pd.read_csv(output_path)
    done_ids = set(done_df['item_id'])
    print(f"Resuming from last run: {len(done_ids)} movies already processed")
else:
    done_df = pd.DataFrame(columns=['item_id', 'sentiment'])
    done_ids = set()

results = done_df.to_dict('records')  # existing results

# Start processing
for i, (mid, title) in enumerate(zip(df[0], df[1])):
    if mid in done_ids:
        continue

    print(f"[{i+1}/{len(df)}] Searching tweets for: {title}")
    try:
        query = f'"{title}" lang:en -is:retweet'
        tweets = client.search_recent_tweets(query=query, max_results=10)
        texts = [t.text for t in tweets.data] if tweets.data else []
        scores = [analyzer.polarity_scores(t)['compound'] for t in texts]
        avg_score = round(sum(scores) / len(scores), 3) if scores else 0.5

        print(f"Sentiment: {avg_score}")
        results.append({'item_id': mid, 'sentiment': avg_score})

        # Save after each movie
        pd.DataFrame(results).to_csv(output_path, index=False)
        time.sleep(1.5)

    except TooManyRequests:
        print("Rate limit hit. Sleeping for 15 minutes...")
        time.sleep(15 * 60)
        continue
    except Exception as e:
        print(f"Error for {title}: {e}")
        results.append({'item_id': mid, 'sentiment': 0.5})
        pd.DataFrame(results).to_csv(output_path, index=False)
        continue

print("All sentiment scores saved to data/sentiment_scores.csv")
