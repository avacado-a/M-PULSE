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

def scrape_micro_bluesky(topic, limit=50):
    print(f"Scraping Micro Data (Bluesky) for topic: {topic}")
    client = Client()
    
    try:
        # Authenticate with Bluesky
        print(f"Logging in to Bluesky as {BSKY_HANDLE}...")
        client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Bluesky search for the topic
        print(f"Searching Bluesky for '{topic}'...")
        params = {'q': topic, 'limit': limit}
        response = client.app.bsky.feed.search_posts(params=params)
        
        count = 0
        for post in response.posts:
            author_handle = post.author.handle
            text = post.record.text
            created_at = post.record.created_at # ISO string (e.g., 2024-03-19T14:00:00Z)
            
            # Convert ISO to timestamp for database consistency
            try:
                # Handle Z and other offsets
                dt_str = created_at.replace("Z", "+00:00")
                dt = datetime.fromisoformat(dt_str)
                ts = dt.timestamp()
            except Exception as e:
                ts = time.time()

            cursor.execute('''
                INSERT INTO micro_data (topic, author, clean_text, created_utc, source, type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (topic, author_handle, text, ts, "bluesky", "post"))
            count += 1
            
        conn.commit()
        conn.close()
        print(f"Bluesky data saved ({count} posts).")
    except Exception as e:
        print(f"Bluesky Scraper Error: {e}")

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "FRC Kraken Motors"
    print(f"Initializing Data Ingestion Pipeline for topic: {topic}")
    init_db()
    scrape_macro(topic)
    scrape_micro_bluesky(topic)
    print("Step 1 complete. Check m_pulse.db")
