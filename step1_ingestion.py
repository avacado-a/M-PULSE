import sqlite3
import time
import feedparser
import praw
import os
import re

DB_NAME = 'm_pulse.db'

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
    return re.sub(cleanr, '', raw_html)

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
        for entry in feed.entries[:20]: # limit to recent
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

def scrape_micro(topic, duration=60):
    print(f"Scraping Micro Data (Reddit) for topic: {topic}")
    CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'YOUR_CLIENT_ID')
    CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', 'YOUR_CLIENT_SECRET')
    USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'MPulseBot/0.1')
    
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        
        subreddits = reddit.subreddit("FRC+robotics")
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        print(f"Simulating a {duration} seconds stream by pulling recent posts...")
        for submission in subreddits.new(limit=50):
            clean_text = submission.title + " " + getattr(submission, 'selftext', '')
            cursor.execute('''
                INSERT INTO micro_data (topic, author, clean_text, created_utc, source, type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (topic, str(submission.author), clean_text, submission.created_utc, "reddit", "submission"))
        conn.commit()
    except Exception as e:
        print(f"Reddit Scraper Error (Expected if API keys missing): {e}")

    print("Micro data scraping completed.")

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "FRC Kraken Motors"
    print(f"Initializing Data Ingestion Pipeline for topic: {topic}")
    init_db()
    scrape_macro(topic)
    scrape_micro(topic, duration=60)
    print("Step 1 complete. Check m_pulse.db")
