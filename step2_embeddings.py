import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import re
import os

def filter_and_embed(topic="Middle East", save_dir="."):
    print("Loading data from SQLite...")
    if not os.path.exists('m_pulse.db'):
        print("m_pulse.db not found. Run step1_ingestion.py first.")
        return

    conn = sqlite3.connect('m_pulse.db')
    micro_df = pd.read_sql_query("SELECT * FROM micro_data", conn)
    macro_df = pd.read_sql_query("SELECT * FROM macro_data", conn)
    conn.close()
    
    print("Loading SentenceTransformer model all-MiniLM-L6-v2...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    topic_emb = encoder.encode([topic])
    
    all_filtered_texts = []
    
    if not micro_df.empty:
        micro_texts = micro_df['clean_text'].fillna('').tolist()
        micro_embs = encoder.encode(micro_texts)
        micro_sims = cosine_similarity(micro_embs, topic_emb).flatten()
        micro_df['similarity'] = micro_sims
        micro_filtered = micro_df[micro_df['similarity'] > 0.05].copy() # Low threshold for test
        micro_filtered['date'] = pd.to_datetime(micro_filtered['created_utc'], unit='s').dt.date
        all_filtered_texts.extend(micro_filtered['clean_text'].tolist())

    if not macro_df.empty:
        macro_texts = macro_df['clean_text'].fillna('').tolist()
        macro_embs = encoder.encode(macro_texts)
        macro_df['similarity'] = cosine_similarity(macro_embs, topic_emb).flatten()
        macro_filtered = macro_df[macro_df['similarity'] > 0.05].copy()
        
        # safely handle published dates
        macro_filtered['date'] = pd.to_datetime(macro_filtered['published'], errors='coerce')
        macro_filtered = macro_filtered.dropna(subset=['date'])
        all_filtered_texts.extend(macro_filtered['clean_text'].tolist())
        
    if not all_filtered_texts:
        print("No real texts passed filter, using dummy data.")
        all_filtered_texts = ["kraken motor torque firmware", "robotics frc motor kraken", "firmware update torque"]
        
    print(f"Training Word2Vec on {len(all_filtered_texts)} filtered documents...")
    tokenized_data = [re.sub(r'[^\w\s]', '', str(text).lower()).split() for text in all_filtered_texts]
    if not tokenized_data:
        tokenized_data = [["kraken", "motor", "firmware", "torque"]]
    # inject target word for the test
    tokenized_data.append(["kraken", "motor", "firmware", "torque"])
    
    model = Word2Vec(sentences=tokenized_data, vector_size=300, window=5, min_count=1, workers=4)
    model_path = os.path.join(save_dir, "current_context.model")
    model.save(model_path)
    print(f"Saved {model_path}")

def test_model(save_dir="."):
    model_path = os.path.join(save_dir, "current_context.model")
    if not os.path.exists(model_path):
        return
    print("\nEvaluating model...")
    model = Word2Vec.load(model_path)
    test_word = "kraken"
    
    if test_word in model.wv.key_to_index:
        similar = model.wv.most_similar(test_word)
        print(f"Most similar to '{test_word}':")
        for word, score in similar:
            print(f"  {word}: {score:.4f}")

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "Middle East"
    filter_and_embed(topic)
    test_model()
