import os
import re
import sqlite3
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import logging
from mpulse.data.sentiment import LexiconSentiment

logger = logging.getLogger(__name__)

class MPulseDataset(Dataset):
    """
    PyTorch Dataset for multi-resolution temporal alignment.
    Generates sliding windows of macro and micro sequences.
    """
    def __init__(self, X_mac: np.ndarray, X_mic: np.ndarray, Y: np.ndarray):
        self.X_mac = torch.tensor(X_mac, dtype=torch.float32)
        self.X_mic = torch.tensor(X_mic, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_mac[idx], self.X_mic[idx], self.Y[idx]

def get_semantic_mean(texts: list, w2v_model: Word2Vec, dim: int = 300) -> np.ndarray:
    """Computes the mean word vector for a collection of texts."""
    vectors = []
    for text in texts:
        words = str(text).lower().split()
        doc_vecs = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
        if doc_vecs:
            vectors.extend(doc_vecs)
    
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

def extract_real_data(topic: str, db_path: str = "m_pulse.db", w2v_path: str = "current_context.model", window_size: int = 3):
    """
    Extracts the last 60 days of data and aligns the streams.
    Returns the raw features, target volume arrays, and the real sentiment array.
    """
    if not os.path.exists(db_path) or not os.path.exists(w2v_path):
        raise FileNotFoundError("Missing DB or Word2Vec model.")

    conn = sqlite3.connect(db_path)
    macro_df = pd.read_sql_query("SELECT published as ts, clean_text as text FROM macro_data WHERE topic=?", conn, params=(topic,))
    micro_df = pd.read_sql_query("SELECT created_utc as ts, clean_text as text FROM micro_data WHERE topic=?", conn, params=(topic,))
    conn.close()

    if macro_df.empty or micro_df.empty:
        raise ValueError(f"Insufficient data for topic: {topic}")

    macro_df['date'] = pd.to_datetime(macro_df['ts'], errors='coerce').dt.date
    micro_df['date'] = pd.to_datetime(micro_df['ts'], unit='s', errors='coerce').dt.date
    
    w2v_model = Word2Vec.load(w2v_path)
    sentiment_analyzer = LexiconSentiment()
    
    daily_micro = micro_df.groupby('date')['text'].apply(list).to_dict()
    daily_macro = macro_df.groupby('date')['text'].apply(list).to_dict()
    
    # Get sorted dates and ENFORCE 60-DAY MAX TIMEFRAME
    all_dates = sorted(list(set(daily_micro.keys()) | set(daily_macro.keys())))
    if len(all_dates) > 60:
        all_dates = all_dates[-60:]
        
    if len(all_dates) <= window_size:
        raise ValueError(f"Not enough data points ({len(all_dates)}) to create sequences with window size {window_size}.")
    
    day_vecs_mic = []
    day_vecs_mac = []
    raw_volumes = []
    raw_sentiments = []
    
    last_mac = np.zeros(w2v_model.vector_size)
    
    # Process each day sequentially
    for d in all_dates:
        mic_texts = daily_micro.get(d, [])
        mac_texts = daily_macro.get(d, [])
        
        day_vecs_mic.append(get_semantic_mean(mic_texts, w2v_model))
        
        mac_vec = get_semantic_mean(mac_texts, w2v_model)
        if np.count_nonzero(mac_vec) == 0:
            mac_vec = last_mac * 0.9 # Decay
        else:
            mac_vec = mac_vec + (last_mac * 0.9)
            last_mac = mac_vec
        day_vecs_mac.append(mac_vec)
        
        raw_volumes.append(len(mic_texts))
        raw_sentiments.append(sentiment_analyzer.score_daily_aggregate(mic_texts))

    # Scale the volumes
    volumes_array = np.array(raw_volumes).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_volumes = scaler.fit_transform(volumes_array).flatten()

    X_mac_seq, X_mic_seq, Y_seq = [], [], []
    for i in range(window_size, len(all_dates)):
        X_mac_seq.append(day_vecs_mac[i-window_size:i])
        X_mic_seq.append(day_vecs_mic[i-window_size:i])
        Y_seq.append(scaled_volumes[i])

    X_mac_arr = np.array(X_mac_seq)
    X_mic_arr = np.array(X_mic_seq)
    Y_arr = np.array(Y_seq)
    
    # Return the raw sentiments and volumes shifted to match the Y sequence outputs
    aligned_volumes = scaled_volumes[window_size:]
    aligned_sentiments = np.array(raw_sentiments)[window_size:]
    
    return X_mac_arr, X_mic_arr, Y_arr, aligned_volumes, aligned_sentiments

def create_dataloaders(X_mac_arr, X_mic_arr, Y_arr, batch_size: int = 32, train_split: float = 0.7):
    """Yields PyTorch DataLoaders from the extracted arrays."""
    split_idx = int(len(Y_arr) * train_split)
    
    train_dataset = MPulseDataset(X_mac_arr[:split_idx], X_mic_arr[:split_idx], Y_arr[:split_idx])
    test_dataset = MPulseDataset(X_mac_arr[split_idx:], X_mic_arr[split_idx:], Y_arr[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, split_idx
