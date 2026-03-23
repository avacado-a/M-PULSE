import sqlite3
import time
import feedparser
import os
import re
import urllib.parse
from atproto import Client
from datetime import datetime

DB_NAME = 'm_pulse.db'

# YOUR BLUESKY CREDENTIALS
BSKY_HANDLE = 'sparik7633.bsky.social'
BSKY_APP_PASSWORD = 'sqxf-cuyn-cyes-mftn'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS macro_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            title TEXT,
            link TEXT,
            published TEXT,
            clean_text TEXT,
            source TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS micro_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            author TEXT,
            clean_text TEXT,
            created_utc REAL,
            source TEXT,
            type TEXT
        )
    ''')
    conn.commit()
    conn.close()

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', str(raw_html))

def scrape_macro(topic):
    print(f"Scraping Macro Data (RSS Feeds) for topic: {topic}")
    rss_feeds = [
        "https://www.firstinspires.org/robotics/frc/blog/rss.xml",
        "https://spectrum.ieee.org/feeds/feed.rss?tag=robotics",
        "https://www.therobotreport.com/feed/"
    ]
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    for feed_url in rss_feeds:
        print(f"Fetching from {feed_url}...")
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:20]:
            title = entry.get('title', '')
            link = entry.get('link', '')
            published = entry.get('published', '')
            summary = clean_html(entry.get('summary', ''))
            
            cursor.execute('''
                INSERT INTO macro_data (topic, title, link, published, clean_text, source)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (topic, title, link, published, title + " " + summary, feed_url))
            
    conn.commit()
    conn.close()
    print("Macro data saved.")

def scrape_micro_bluesky(topic, max_pages=5):
    print(f"Scraping Micro Data (Bluesky) for topic: {topic}")
    client = Client()
    
    try:
        print(f"Logging in to Bluesky as {BSKY_HANDLE}...")
        client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)
        
        conn = sqlite3.connect(DB_NAME)
        cursor_db = conn.cursor()
        
        cursor_token = None
        total_count = 0
        
        for page in range(max_pages):
            print(f"  Fetching page {page+1} of Bluesky search...")
            params = {'q': topic, 'limit': 100}
            if cursor_token:
                params['cursor'] = cursor_token
                
            response = client.app.bsky.feed.search_posts(params=params)
            
            for post in response.posts:
                author_handle = post.author.handle
                text = post.record.text
                created_at = post.record.created_at 
                
                try:
                    dt_str = created_at.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(dt_str)
                    ts = dt.timestamp()
                except:
                    ts = time.time()

                cursor_db.execute('''
                    INSERT INTO micro_data (topic, author, clean_text, created_utc, source, type)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (topic, author_handle, text, ts, "bluesky", "post"))
                total_count += 1
            
            cursor_token = response.cursor
            if not cursor_token:
                break
                
        conn.commit()
        conn.close()
        print(f"Bluesky data saved ({total_count} posts total).")
    except Exception as e:
        print(f"Bluesky Scraper Error: {e}")

def backfill_macro_csv(file_path, topic):
    """
    Call this if you have a Kaggle CSV with columns: [title, summary, date, url]
    """
    if not os.path.exists(file_path):
        print(f"CSV for backfill not found: {file_path}")
        return
    
    import pandas as pd
    print(f"Backfilling Macro data from {file_path}...")
    df = pd.read_csv(file_path)
    
    conn = sqlite3.connect(DB_NAME)
    for _, row in df.iterrows():
        conn.execute('''
            INSERT INTO macro_data (topic, title, link, published, clean_text, source)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (topic, row['title'], row['url'], row['date'], str(row['title']) + " " + str(row['summary']), "historical_csv"))
    conn.commit()
    conn.close()
    print("Backfill complete.")

from gdeltdoc import GdeltDoc, Filters

def scrape_macro_gdelt(topic, start_year=2023):
    print(f"--- M-PULSE PATIENT INGESTION: BUILDING 3-YEAR HISTORY ---")
    print(f"Target: {topic} | Start: {start_year} | Mode: Accuracy-First")
    gd = GdeltDoc()
    conn = sqlite3.connect(DB_NAME)
    
    import datetime
    current_date = datetime.date(start_year, 1, 1)
    end_goal = datetime.date(2026, 3, 19)
    
    while current_date < end_goal:
        chunk_end = current_date + datetime.timedelta(days=90)
        s_str = current_date.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")
        
        f = Filters(keyword=topic, start_date=s_str, end_date=e_str)
        
        # PATIENT RETRY LOOP
        while True:
            try:
                print(f"  [PENDING] Querying {s_str} to {e_str}...")
                articles = gd.article_search(f)
                
                if not articles.empty:
                    print(f"    [DEBUG] GDELT found {len(articles)} articles. Sample titles:")
                    for t in articles['title'].iloc[:3]:
                        print(f"      - {t}")
                        
                    count = 0
                    for _, row in articles.iterrows():
                        link = row.get('url', '')
                        exists = conn.execute("SELECT 1 FROM macro_data WHERE link = ?", (link,)).fetchone()
                        if not exists:
                            conn.execute('''
                                INSERT INTO macro_data (topic, title, link, published, clean_text, source)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (topic, row.get('title',''), link, row.get('seendate',''), row.get('title',''), "gdelt_historical"))
                            count += 1
                    conn.commit()
                    print(f"  [SUCCESS] Saved {count} new articles.")
                
                # Mandatory cooldown to respect API
                print("  [COOLDOWN] Waiting 10 seconds...")
                time.sleep(10)
                break # Exit retry loop on success
                
            except Exception as e:
                if "RateLimitError" in str(type(e)):
                    print("  [RATE LIMIT] Paywall/Velocity hit. Patiently waiting 65 seconds...")
                    time.sleep(65) # Wait over a minute to reset API timer
                else:
                    print(f"  [ERROR] Unexpected issue: {e}. Skipping chunk.")
                    break
        
        current_date = chunk_end
    
    conn.close()
    print("--- 3-Year Macro Ingestion Complete ---")

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "Middle East"
    print(f"Initializing Data Ingestion Pipeline for topic: {topic}")
    init_db()
    # scrape_macro(topic) # Legacy RSS
    scrape_macro_gdelt(topic, start_year=2024) # Reduced to 2024 to speed up tests, adjust as needed
    scrape_micro_bluesky(topic, max_pages=3)
    print("Step 1 complete. Check m_pulse.db")
