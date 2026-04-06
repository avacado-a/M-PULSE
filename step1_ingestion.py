import sqlite3
import time
import feedparser
import os
import re
import urllib.parse
from atproto import Client
from datetime import datetime, date, timedelta
from gdeltdoc import GdeltDoc, Filters

"""
M-PULSE Data Ingestion Pipeline
Step 1: Multi-Resolution Temporal Collection
Reference: Delucia et al. (2022) - Optimized Tokenization for Social Context

This module utilizes a dual-pathway ingestion strategy:
1. Macro-Stream: Historical backfilling of conventional news via GDELT 2.0.
2. Micro-Stream: Live micro-parametric chatter collection via Bluesky AT Protocol.
"""

DB_NAME = 'm_pulse.db'
# Credentials loaded from environment variables for GitHub security
BSKY_HANDLE = os.getenv('BSKY_HANDLE', 'your_handle.bsky.social')
BSKY_APP_PASSWORD = os.getenv('BSKY_APP_PASSWORD', 'xxxx-xxxx-xxxx-xxxx')

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS macro_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT, title TEXT, link TEXT, published TEXT, clean_text TEXT, source TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS micro_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT, author TEXT, clean_text TEXT, created_utc REAL, source TEXT, type TEXT
        )
    ''')
    conn.commit(); conn.close()

def scrape_macro_gdelt(topic, start_year=2024):
    """
    Historical Macro Backfill Logic.
    Bypasses standard 30-day API paywalls by utilizing chunked GDELT 2.0 Doc queries.
    """
    print(f"🛰️ Initiating Macro-Stream Backfill for: {topic}...")
    gd = GdeltDoc()
    conn = sqlite3.connect(DB_NAME)
    
    current_date = date(start_year, 1, 1)
    end_goal = date.today()
    
    while current_date < end_goal:
        chunk_end = current_date + timedelta(days=90)
        s_str, e_str = current_date.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        f = Filters(keyword=topic, start_date=s_str, end_date=e_str)
        
        while True: # Patient Retry Protocol
            try:
                time.sleep(5) # Respect GDELT 1-req/5-sec threshold
                articles = gd.article_search(f)
                if not articles.empty:
                    count = 0
                    for _, row in articles.iterrows():
                        link = row.get('url', '')
                        # Deduplication Layer
                        exists = conn.execute("SELECT 1 FROM macro_data WHERE link = ?", (link,)).fetchone()
                        if not exists:
                            conn.execute('INSERT INTO macro_data (topic, title, link, published, clean_text, source) VALUES (?,?,?,?,?,?)',
                                         (topic, row.get('title',''), link, row.get('seendate',''), row.get('title',''), "gdelt_historical"))
                            count += 1
                    conn.commit()
                    print(f"  [CHUNK SUCCESS] {s_str} -> {e_str} | Ingested: {count}")
                break
            except Exception as e:
                if "RateLimitError" in str(type(e)):
                    print("  [RATE LIMIT] Respecting paywall threshold. Backing off 65s...")
                    time.sleep(65)
                else: break
        current_date = chunk_end
    conn.close()

def scrape_micro_bluesky(topic, max_pages=5):
    """
    Live Micro-Parametric Social Stream collection.
    Utilizes AT Protocol cursor pagination to build a deep social context window.
    """
    print(f"📡 Initiating Micro-Stream Live Ingestion: {topic}...")
    client = Client()
    try:
        client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)
        conn = sqlite3.connect(DB_NAME)
        cursor_db = conn.cursor()
        
        cursor_token = None
        for page in range(max_pages):
            params = {'q': topic, 'limit': 100}
            if cursor_token: params['cursor'] = cursor_token
            response = client.app.bsky.feed.search_posts(params=params)
            
            for post in response.posts:
                try:
                    ts = datetime.fromisoformat(post.record.created_at.replace("Z", "+00:00")).timestamp()
                except: ts = time.time()
                cursor_db.execute('INSERT INTO micro_data (topic, author, clean_text, created_utc, source, type) VALUES (?,?,?,?,?,?)',
                                 (topic, post.author.handle, post.record.text, ts, "bluesky", "post"))
            
            cursor_token = response.cursor
            if not cursor_token: break
        conn.commit(); conn.close()
        print(f"  [SOCIAL SUCCESS] Bluesky sequence window updated.")
    except Exception as e: print(f"  [MICRO ERROR] {e}")

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "robotics"
    init_db()
    scrape_macro_gdelt(topic)
    scrape_micro_bluesky(topic)
