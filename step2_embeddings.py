import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import re
import os

"""
M-PULSE Semantic Encoder
Step 2: Ephemeral Vector Space Mapping
Reference: Kaur et al. (2025) - Dynamic Context Embeddings for Volatile Streams

This module filters raw media corpus based on semantic relevance to the 
target topic and generates localized Word2Vec embeddings. Localized encoding 
is prioritized over generalized LLMs to preserve platform-specific 
social context (Delucia et al., 2022).
"""

def filter_and_embed(topic="robotics", save_dir="."):
    """
    Applies Semantic Noise Thresholding to isolate relevant documents.
    Generates a task-specific vector space for the downstream LSTM.
    """
    print(f"🧠 Initiating Semantic Encoding for: {topic}...")
    if not os.path.exists('m_pulse.db'): return

    conn = sqlite3.connect('m_pulse.db')
    micro_df = pd.read_sql_query("SELECT clean_text FROM micro_data WHERE topic=?", conn, params=(topic,))
    macro_df = pd.read_sql_query("SELECT clean_text FROM macro_data WHERE topic=?", conn, params=(topic,))
    conn.close()
    
    # Utilizing SentenceTransformers for high-precision semantic filtering
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    topic_emb = encoder.encode([topic])
    
    all_filtered_texts = []
    
    # Semantic Thresholding Protocol
    for df in [micro_df, macro_df]:
        if not df.empty:
            texts = df['clean_text'].fillna('').tolist()
            embs = encoder.encode(texts)
            sims = cosine_similarity(embs, topic_emb).flatten()
            # Filter: Removing low-relevance semantic noise
            filtered = [texts[i] for i, score in enumerate(sims) if score > 0.15]
            all_filtered_texts.extend(filtered)

    if not all_filtered_texts:
        print("  [CORPUS ERROR] No documents passed semantic threshold.")
        return
        
    print(f"  [ENCODER SUCCESS] Corpus size: {len(all_filtered_texts)} documents.")
    
    # Ephemeral Word2Vec Generation
    # Optimized for short-form volatility and technical terminology.
    tokenized_data = [re.sub(r'[^\w\s]', '', str(text).lower()).split() for text in all_filtered_texts]
    model = Word2Vec(sentences=tokenized_data, vector_size=300, window=5, min_count=1, workers=4)
    
    model_path = os.path.join(save_dir, "current_context.model")
    model.save(model_path)
    print(f"  [MODEL SAVED] Task-specific embeddings mapped to {model_path}")

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "robotics"
    filter_and_embed(topic)
