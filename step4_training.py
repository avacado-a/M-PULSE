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
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_day_vector(text_list, w2v_model, feature_dim=300):
    all_vecs = []
    for text in text_list:
        words = str(text).lower().split()
        vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
        if vectors: all_vecs.append(np.mean(vectors, axis=0))
    return np.mean(all_vecs, axis=0) if all_vecs else np.zeros(feature_dim)

def load_real_data(db_path="m_pulse.db", w2v_path="current_context.model"):
    print("Loading and Aggregating Daily Time-Series...")
    conn = sqlite3.connect(db_path)
    macro_df = pd.read_sql_query("SELECT published as ts, clean_text as text FROM macro_data", conn)
    micro_df = pd.read_sql_query("SELECT created_utc as ts, clean_text as text FROM micro_data", conn)
    conn.close()

    macro_df['date'] = pd.to_datetime(macro_df['ts'], errors='coerce').dt.date
    micro_df['date'] = pd.to_datetime(micro_df['ts'], unit='s', errors='coerce').dt.date
    
    w2v_model = Word2Vec.load(w2v_path)
    
    daily_micro = micro_df.groupby('date')['text'].apply(list).to_dict()
    daily_macro = macro_df.groupby('date')['text'].apply(list).to_dict()
    all_dates = sorted(list(set(daily_micro.keys()) | set(daily_macro.keys())))
    
    day_vecs_mic, day_vecs_mac, raw_volumes = [], [], []
    last_mac = np.zeros(w2v_model.vector_size)
    
    for d in all_dates:
        day_vecs_mic.append(get_day_vector(daily_micro.get(d, []), w2v_model))
        
        mac_vec = get_day_vector(daily_macro.get(d, []), w2v_model)
        if np.count_nonzero(mac_vec) == 0:
            mac_vec = last_mac * 0.8
        else:
            mac_vec = mac_vec + (last_mac * 0.8)
            last_mac = mac_vec
        day_vecs_mac.append(mac_vec)
        raw_volumes.append(np.log1p(len(daily_micro.get(d, []))))

    smooth_volumes = pd.Series(raw_volumes).rolling(window=3, min_periods=1).mean().tolist()

    window = 7
    X_mac, X_mic, Y = [], [], []
    for i in range(window, len(all_dates)):
        X_mac.append(day_vecs_mac[i-window:i])
        X_mic.append(day_vecs_mic[i-window:i])
        Y.append(smooth_volumes[i])
        
    return (torch.tensor(np.array(X_mac), dtype=torch.float32), 
            torch.tensor(np.array(X_mic), dtype=torch.float32), 
            torch.tensor(np.array(Y), dtype=torch.float32).view(-1, 1))

def run_ablation_experiment(name, use_mac, use_mic, X_mac, X_mic, Y, split):
    model = MPulseNet(use_macro=use_mac, use_micro=use_mic, seq_len=7).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    X_mac_tr, X_mac_te = X_mac[:split].to(device), X_mac[split:].to(device)
    X_mic_tr, X_mic_te = X_mic[:split].to(device), X_mic[split:].to(device)
    Y_tr, Y_te = Y[:split].to(device), Y[split:].to(device)
    
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_mac_tr, X_mic_tr), Y_tr)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(X_mac_te, X_mic_te).cpu().numpy().flatten()
    mse = mean_squared_error(Y_te.cpu().numpy().flatten(), preds)
    return preds, mse

if __name__ == "__main__":
    run_dir = f"runs/RESEARCH_PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    
    X_mac, X_mic, Y = load_real_data()
    split = int(len(Y) * 0.75) # 75% train, 25% blind prediction
    
    print(f"Dataset: {len(Y)} days. Training on {split}, Forecasting next {len(Y)-split} days.")
    
    # Run the Ablation Study
    print("Training Macro-Only (News)...")
    preds_mac, mse_mac = run_ablation_experiment("Macro", True, False, X_mac, X_mic, Y, split)
    
    print("Training Micro-Only (Bluesky)...")
    preds_mic, mse_mic = run_ablation_experiment("Micro", False, True, X_mac, X_mic, Y, split)
    
    print("Training Dual-Stream (M-PULSE)...")
    preds_dual, mse_dual = run_ablation_experiment("Dual", True, True, X_mac, X_mic, Y, split)
    
    # Generate Research Paper Graph
    plt.figure(figsize=(14,7))
    actuals = Y.numpy().flatten()
    
    # Plot history vs future
    plt.plot(range(len(actuals)), actuals, label='Actual Trend Volume', color='black', linewidth=2)
    plt.axvline(x=split, color='gray', linestyle='--', label='Forecasting Cutoff')
    
    # Pad predictions to align with the x-axis
    x_axis = range(split, len(actuals))
    plt.plot(x_axis, preds_mac, label=f'Macro Only (MSE: {mse_mac:.4f})', color='red', linestyle='-.', alpha=0.7)
    plt.plot(x_axis, preds_mic, label=f'Micro Only (MSE: {mse_mic:.4f})', color='blue', linestyle=':', alpha=0.7)
    plt.plot(x_axis, preds_dual, label=f'M-PULSE Dual-Stream (MSE: {mse_dual:.4f})', color='green', linewidth=3)
    
    plt.title("Ablation Study: Predicting Micro-Parametric Social Events using Multi-Resolution Data")
    plt.xlabel("Days")
    plt.ylabel("Normalized Topic Volume")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    graph_path = "ablation_study_hero_graph.png"
    plt.savefig(graph_path, dpi=300)
    print(f"\n✅ RESEARCH READY. Open {graph_path} right now.")
    print(f"Final Stats to cite -> Macro MSE: {mse_mac:.4f} | Micro MSE: {mse_mic:.4f} | Dual MSE: {mse_dual:.4f}")
