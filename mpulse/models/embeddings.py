import os
import re
import sqlite3
import pandas as pd
import logging
import numpy as np
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class SemanticEncoder:
    """
    Generates localized Word2Vec embeddings from a semantic subset of the corpus.
    Implements DBSCAN Outlier Stripping to filter anomalous bias and noise.
    """
    def __init__(self, db_path: str = 'm_pulse.db', vector_size: int = 300):
        self.db_path = db_path
        self.vector_size = vector_size

    def generate_embeddings(self, topic: str, save_path: str = "current_context.model"):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database {self.db_path} not found.")

        logger.info(f"Filtering corpus for topic: {topic}")
        conn = sqlite3.connect(self.db_path)
        macro_df = pd.read_sql_query("SELECT clean_text FROM macro_data WHERE topic=?", conn, params=(topic,))
        micro_df = pd.read_sql_query("SELECT clean_text FROM micro_data WHERE topic=?", conn, params=(topic,))
        conn.close()

        # Semantic Thresholding using a pretrained transformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        topic_emb = encoder.encode([topic])
        
        filtered_corpus = []
        
        # Process and merge both streams
        all_texts = []
        if not micro_df.empty: all_texts.extend(micro_df['clean_text'].dropna().tolist())
        if not macro_df.empty: all_texts.extend(macro_df['clean_text'].dropna().tolist())
            
        if not all_texts:
            logger.warning("No documents available for semantic filtering.")
            return False

        # Encode full text set
        logger.info("Encoding text for DBSCAN Bias Mitigation...")
        embs = encoder.encode(all_texts)
        sims = cosine_similarity(embs, topic_emb).flatten()
        
        # 1. Semantic Relevance Filtering (Keep related docs)
        relevance_threshold = 0.15
        relevant_indices = np.where(sims > relevance_threshold)[0]
        
        if len(relevant_indices) == 0:
            logger.warning("No documents passed relevance threshold.")
            return False
            
        relevant_texts = [all_texts[i] for i in relevant_indices]
        relevant_embs = embs[relevant_indices]

        # 2. DBSCAN Outlier Stripping (Alphonse et al., 2025 Implementation)
        # Drops extreme outliers (e.g., severe biased anomalies) marked as -1 by DBSCAN
        logger.info("Executing DBSCAN Outlier Removal...")
        clustering = DBSCAN(eps=0.5, min_samples=3, metric='cosine').fit(relevant_embs)
        
        for i, label in enumerate(clustering.labels_):
            if label != -1:  # -1 means outlier
                filtered_corpus.append(relevant_texts[i])

        if not filtered_corpus:
            logger.warning("DBSCAN stripped all documents. Loosening constraints.")
            filtered_corpus = relevant_texts # Fallback if data is too sparse

        logger.info(f"Training Word2Vec on {len(filtered_corpus)} cleaned documents.")
        tokenized_data = [re.sub(r'[^\w\s]', '', str(text).lower()).split() for text in filtered_corpus]
        
        # Word2Vec training with the cleaned corpus
        model = Word2Vec(
            sentences=tokenized_data, 
            vector_size=self.vector_size, 
            window=5, 
            min_count=1, 
            workers=4
        )
        model.save(save_path)
        logger.info(f"Embeddings saved to {save_path}")
        return True
