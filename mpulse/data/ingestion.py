import sqlite3
import time
import os
import logging
from datetime import datetime, date, timedelta
from gdeltdoc import GdeltDoc, Filters
from atproto import Client

logger = logging.getLogger(__name__)

class DataIngestor:
    """
    Handles temporal multi-resolution data collection.
    """
    def __init__(self, db_name: str = 'm_pulse.db'):
        self.db_name = db_name
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_name)
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
        conn.commit()
        conn.close()

    def fetch_macro_gdelt(self, topic: str, start_year: int = 2024):
        """
        Ingests historical news data using chunked GDELT queries.
        """
        logger.info(f"Initiating Macro ingestion for {topic}")
        gd = GdeltDoc()
        conn = sqlite3.connect(self.db_name)
        
        current_date = date(start_year, 1, 1)
        end_goal = date.today()
        
        while current_date < end_goal:
            chunk_end = current_date + timedelta(days=90)
            s_str = current_date.strftime("%Y-%m-%d")
            e_str = chunk_end.strftime("%Y-%m-%d")
            
            # Exact phrase match optimizes GDELT queries and prevents timeout/rate limits
            query_keyword = f'"{topic}"' if ' ' in topic else topic
            filters = Filters(keyword=query_keyword, start_date=s_str, end_date=e_str)
            
            # Robust retry logic for API limits
            retries = 3
            while retries > 0:
                try:
                    time.sleep(5)
                    articles = gd.article_search(filters)
                    if not articles.empty:
                        count = 0
                        for _, row in articles.iterrows():
                            link = row.get('url', '')
                            exists = conn.execute("SELECT 1 FROM macro_data WHERE link = ?", (link,)).fetchone()
                            if not exists:
                                conn.execute(
                                    'INSERT INTO macro_data (topic, title, link, published, clean_text, source) VALUES (?,?,?,?,?,?)',
                                    (topic, row.get('title',''), link, row.get('seendate',''), row.get('title',''), "gdelt")
                                )
                                count += 1
                        conn.commit()
                        logger.info(f"Ingested {count} articles for chunk {s_str}")
                    break
                except Exception as e:
                    if "RateLimit" in str(type(e).__name__):
                        logger.warning("Rate limit hit. Backing off 65s.")
                        time.sleep(65)
                    else:
                        logger.error(f"GDELT fetch error: {e}")
                        break
                retries -= 1
            current_date = chunk_end
        conn.close()

    def fetch_micro_bluesky(self, topic: str, max_pages: int = 5):
        """
        Ingests social data using AT Protocol pagination.
        """
        bsky_handle = os.getenv('BSKY_HANDLE')
        bsky_pass = os.getenv('BSKY_APP_PASSWORD')
        
        if not bsky_handle or not bsky_pass:
            logger.warning("Bluesky credentials not found in environment. Skipping micro ingestion.")
            return

        logger.info(f"Initiating Micro ingestion for {topic}")
        client = Client()
        try:
            client.login(bsky_handle, bsky_pass)
            conn = sqlite3.connect(self.db_name)
            
            cursor_token = None
            total_saved = 0
            for _ in range(max_pages):
                params = {'q': topic, 'limit': 100}
                if cursor_token: 
                    params['cursor'] = cursor_token
                    
                response = client.app.bsky.feed.search_posts(params=params)
                
                for post in response.posts:
                    try:
                        ts = datetime.fromisoformat(post.record.created_at.replace("Z", "+00:00")).timestamp()
                    except ValueError: 
                        ts = time.time()
                        
                    conn.execute(
                        'INSERT INTO micro_data (topic, author, clean_text, created_utc, source, type) VALUES (?,?,?,?,?,?)',
                        (topic, post.author.handle, post.record.text, ts, "bluesky", "post")
                    )
                    total_saved += 1
                
                cursor_token = response.cursor
                if not cursor_token: 
                    break
                    
            conn.commit()
            conn.close()
            logger.info(f"Ingested {total_saved} social posts.")
        except Exception as e:
            logger.error(f"Bluesky ingestion error: {e}")
