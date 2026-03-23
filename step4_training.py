import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from step3_model import MPulseNet
from gensim.models import Word2Vec
import sqlite3
import pandas as pd
import os
import re
from datetime import datetime

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_day_vector(text_list, w2v_model, feature_dim=300):
    """Averages all text for a single day into one meaningful vector."""
    all_vecs = []
    for text in text_list:
        words = str(text).lower().split()
        vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
        if vectors:
            all_vecs.append(np.mean(vectors, axis=0))
    return np.mean(all_vecs, axis=0) if all_vecs else np.zeros(feature_dim)

def load_real_data(db_path="m_pulse.db", w2v_path="current_context.model"):
    print("Locked In: Loading and Aggregating Daily Time-Series...")
    conn = sqlite3.connect(db_path)
    macro_df = pd.read_sql_query("SELECT published as ts, clean_text as text FROM macro_data", conn)
    micro_df = pd.read_sql_query("SELECT created_utc as ts, clean_text as text FROM micro_data", conn)
    conn.close()

    # Standardize Dates
    macro_df['date'] = pd.to_datetime(macro_df['ts'], errors='coerce').dt.date
    micro_df['date'] = pd.to_datetime(micro_df['ts'], unit='s', errors='coerce').dt.date
    
    w2v_model = Word2Vec.load(w2v_path)
    
    # 1. Aggregate daily volume and text
    daily_micro = micro_df.groupby('date')['text'].apply(list).to_dict()
    daily_macro = macro_df.groupby('date')['text'].apply(list).to_dict()
    
    all_dates = sorted(list(set(daily_micro.keys()) | set(daily_macro.keys())))
    
    # 2. Build Daily Vectors and Targets
    day_vecs_mic = []
    day_vecs_mac = []
    raw_volumes = []
    
    last_mac = np.zeros(w2v_model.vector_size)
    decay_rate = 0.8  # News impact decays by 20% each day without new news
    
    for d in all_dates:
        # Micro is typically dense
        day_vecs_mic.append(get_day_vector(daily_micro.get(d, []), w2v_model))
        
        # Macro is sparse, apply EMA/forward fill to smooth it out
        mac_vec = get_day_vector(daily_macro.get(d, []), w2v_model)
        if np.count_nonzero(mac_vec) == 0:
            mac_vec = last_mac * decay_rate
        else:
            # Combine new news with lingering old news
            mac_vec = mac_vec + (last_mac * decay_rate)
            last_mac = mac_vec
        day_vecs_mac.append(mac_vec)
        
        # Target: Log of daily post count 
        raw_volumes.append(np.log1p(len(daily_micro.get(d, []))))

    # Smooth the target volume to remove artificial spikes (e.g. last 48 hours)
    # Using a 3-day rolling average
    smooth_volumes = pd.Series(raw_volumes).rolling(window=3, min_periods=1).mean().tolist()

    # 3. Create Sliding Window (7-day lookback to predict day 8)
    window = 7
    X_mac, X_mic, Y = [], [], []
    
    for i in range(window, len(all_dates)):
        X_mac.append(day_vecs_mac[i-window:i])
        X_mic.append(day_vecs_mic[i-window:i])
        Y.append(smooth_volumes[i])
        
    print(f"✅ Created {len(Y)} daily time-steps for training.")
    
    return (torch.tensor(np.array(X_mac), dtype=torch.float32), 
            torch.tensor(np.array(X_mic), dtype=torch.float32), 
            torch.tensor(np.array(Y), dtype=torch.float32).view(-1, 1),
            window)

def run_final_dual_stream():
    # Setup
    run_id = f"FINAL_LOCKED_IN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    X_mac, X_mic, Y, win_size = load_real_data()
    
    # Dynamic Model Creation
    model = MPulseNet(macro_seq_len=win_size, micro_seq_len=win_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train/Test Split (Time-series aware)
    split = int(len(Y) * 0.8)
    X_mac_tr, X_mac_te = X_mac[:split].to(device), X_mac[split:].to(device)
    X_mic_tr, X_mic_te = X_mic[:split].to(device), X_mic[split:].to(device)
    Y_tr, Y_te = Y[:split].to(device), Y[split:].to(device)
    
    print(f"🚀 Training Dual-Stream on GPU ({device})...")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_mac_tr, X_mic_tr), Y_tr)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/200] | Loss: {loss.item():.6f}")

    # Eval
    model.eval()
    with torch.no_grad():
        preds = model(X_mac_te, X_mic_te).cpu().numpy().flatten()
    
    mse = mean_squared_error(Y_te.cpu().numpy().flatten(), preds)
    
    # Final Organized Saving
    plt.figure(figsize=(12,6))
    actuals = Y.numpy().flatten()
    plt.plot(actuals, label='Actual Daily Log-Volume', color='black', alpha=0.4)
    padded_preds = np.full(len(actuals), np.nan)
    padded_preds[split:] = preds
    plt.plot(padded_preds, label=f'M-PULSE Forecast (MSE: {mse:.4f})', color='green', linewidth=2)
    plt.axvline(x=split, color='red', linestyle='--', label='Forecast Cutoff')
    plt.title("M-PULSE Final Result: Daily Aggregated Forecasting")
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(run_dir, "final_forecast.png"))
    plt.savefig("final_forecast_root.png") # Also in root for easy viewing
    
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    print(f"\n✅ FINAL RUN COMPLETE. See: {run_dir}")
    print(f"MSE: {mse:.6f}")

if __name__ == "__main__":
    run_final_dual_stream()
