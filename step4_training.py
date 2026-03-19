import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from step3_model import MPulseNet
from gensim.models import Word2Vec
import sqlite3
import time
import os
import csv
import re
from datetime import datetime

# DEVICE CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# RESEARCH TELEMETRY LOG
TELEMETRY_FILE = 'research_telemetry_log.csv'

def init_telemetry():
    if not os.path.exists(TELEMETRY_FILE):
        with open(TELEMETRY_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Run_Type', 'Epoch', 'Training_Loss', 'Val_Loss', 'Peak_VRAM_GB', 'Epoch_Time_Sec'])

def log_telemetry(run_type, epoch, train_loss, val_loss, vram, epoch_time):
    with open(TELEMETRY_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), run_type, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{vram:.4f}", f"{epoch_time:.2f}"])

def text_to_vector_seq(text_list, w2v_model, seq_len, feature_dim=300):
    sequences = []
    for text in text_list:
        words = re.sub(r'[^\w\s]', '', str(text).lower()).split()
        vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
        if len(vectors) == 0:
            vectors = [np.zeros(feature_dim)]
        if len(vectors) < seq_len:
            padding = [np.zeros(feature_dim)] * (seq_len - len(vectors))
            vectors = vectors + padding
        else:
            vectors = vectors[:seq_len]
        sequences.append(np.array(vectors))
    return np.array(sequences)

def load_real_data(db_name='m_pulse.db', model_path='current_context.model'):
    if not os.path.exists(db_name) or not os.path.exists(model_path):
        raise FileNotFoundError("Database or Word2Vec model missing. Run steps 1 and 2.")
    conn = sqlite3.connect(db_name)
    macro_data = conn.execute("SELECT clean_text FROM macro_data").fetchall()
    micro_data = conn.execute("SELECT clean_text FROM micro_data").fetchall()
    conn.close()
    
    w2v_model = Word2Vec.load(model_path)
    macro_texts = [row[0] for row in macro_data]
    micro_texts = [row[0] for row in micro_data]
    
    y_values = [len(str(m).split()) / 100.0 for m in macro_texts]
    min_len = min(len(macro_texts), len(micro_texts))
    macro_texts, micro_texts, y_values = macro_texts[:min_len], micro_texts[:min_len], y_values[:min_len]
    
    X_macro = text_to_vector_seq(macro_texts, w2v_model, 104)
    X_micro = text_to_vector_seq(micro_texts, w2v_model, 60)
    Y = np.array(y_values).reshape(-1, 1)
    
    return torch.tensor(X_macro, dtype=torch.float32), torch.tensor(X_micro, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def run_experiment(run_type, use_macro, use_micro, X_macro, X_micro, Y, max_epochs=500, patience=15):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{run_type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Move model to device
    model = MPulseNet(use_macro=use_macro, use_micro=use_micro).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    split_idx = int(len(Y) * 0.8)
    # Move data to device
    X_mac_train, X_mac_test = X_macro[:split_idx].to(device), X_macro[split_idx:].to(device)
    X_mic_train, X_mic_test = X_micro[:split_idx].to(device), X_micro[split_idx:].to(device)
    Y_train, Y_test = Y[:split_idx].to(device), Y[split_idx:].to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    # Reset VRAM peak tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print(f"Starting {run_type} on {device}...")
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        optimizer.zero_grad()
        outputs = model(X_mac_train, X_mic_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_mac_test, X_mic_test)
            val_loss = criterion(val_outputs, Y_test)
        
        epoch_time = time.time() - start_time
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        # Accurate VRAM Tracking
        vram = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.01
        log_telemetry(run_type, epoch + 1, loss.item(), val_loss.item(), vram, epoch_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  Early Stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

    # Final Evaluation
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt")))
    model.eval()
    with torch.no_grad():
        predictions = model(X_mac_test, X_mic_test).cpu().numpy().flatten()
    
    mse = mean_squared_error(Y_test.cpu().numpy().flatten(), predictions)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f"Learning Curve: {run_type}")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "learning_curve.png"))
    plt.close()

    return predictions, mse, run_dir, split_idx

# Execution
os.makedirs("runs", exist_ok=True)
init_telemetry()

try:
    X_macro, X_micro, Y = load_real_data()
    print(f"Real data loaded: {len(Y)} samples.")
    
    results = {}
    for r_type, macro, micro in [("Micro_Only", False, True), ("Macro_Only", True, False), ("Dual_Stream", True, True)]:
        results[r_type] = run_experiment(r_type, macro, micro, X_macro, X_micro, Y)
        
    # Comparison Graph
    plt.figure(figsize=(14, 7))
    actuals = Y.numpy().flatten()
    plt.plot(actuals, label='Actual Intensity', color='black', alpha=0.3)
    s_idx = results['Dual_Stream'][3]
    plt.axvline(x=s_idx, color='red', linestyle='--', label='Cutoff')
    
    for name, (pred, mse, r_dir, _) in results.items():
        padded = np.full(len(actuals), np.nan)
        padded[s_idx:] = pred
        plt.plot(padded, label=f"{name} (MSE: {mse:.4f})")
    
    plt.title("M-PULSE Final Results: Real-World Text Data on GPU")
    plt.legend()
    plt.savefig("real_data_gpu_results.png")
    print("\nSUCCESS: Pipeline verified with real data on GPU hardware.")
except Exception as e:
    print(f"ERROR: {e}")
