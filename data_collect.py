import os
import feedparser
import pandas as pd
from datetime import datetime

def get_yahoo_rss_headlines(ticker, max_items=50):
    """
    Fetch latest news headlines for a stock ticker from Yahoo Finance RSS feed.

    Returns a pandas DataFrame with columns: ['ticker', 'date', 'headline']
    """
    url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
    feed = feedparser.parse(url)

    data = []
    if 'entries' in feed and feed.entries:
        for entry in feed.entries[:max_items]:
            try:
                title = entry.title
                published = pd.to_datetime(entry.published, errors='coerce')

                if pd.isna(published):
                    published = datetime.today()

                data.append({
                    'ticker': ticker,
                    'date': published.date(),
                    'headline': title
                })
            except Exception as e:
                print(f"Skipping a headline due to error: {e}")

    else:
        print(f"⚠️ No news entries found for {ticker} at this time.")

    return pd.DataFrame(data)

def append_to_csv(df_new, csv_file):
    """
    Append new headlines to the CSV file, avoiding duplicates (based on ticker + date + headline).
    """
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)

        # Combine and drop duplicates
        combined = pd.concat([df_existing, df_new], ignore_index=True)
        combined.drop_duplicates(subset=['ticker', 'date', 'headline'], inplace=True)
    else:
        combined = df_new

    combined.sort_values(by=['date'], inplace=True)
    combined.to_csv(csv_file, index=False)
    print(f"✅ Saved {len(df_new)} new headlines to {csv_file}")

if __name__ == "__main__":
    ticker = "AAPL"
    csv_file = f"{ticker}_news_rss_history.csv"

    today_headlines = get_yahoo_rss_headlines(ticker, max_items=50)

    if not today_headlines.empty:
        append_to_csv(today_headlines, csv_file)
    else:
        print("✅ No new headlines today.")
