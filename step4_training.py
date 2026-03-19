import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from step3_model import MPulseNet
import time
import os
import csv
from datetime import datetime

# RESEARCH TELEMETRY LOG
TELEMETRY_FILE = 'research_telemetry_log.csv'

def init_telemetry():
    if not os.path.exists(TELEMETRY_FILE):
        with open(TELEMETRY_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Run_Type', 'Epoch', 'Training_Loss', 'Peak_VRAM_GB', 'Epoch_Time_Sec'])

def log_telemetry(run_type, epoch, loss, vram, epoch_time):
    with open(TELEMETRY_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), run_type, epoch, f"{loss:.6f}", f"{vram:.4f}", f"{epoch_time:.2f}"])

def run_experiment(run_type, use_macro, use_micro, X_macro, X_micro, Y, epochs=100):
    # Create Dynamic Run Folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_type}_{timestamp}"
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    model = MPulseNet(use_macro=use_macro, use_micro=use_micro)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # --- TRAIN/TEST SPLIT (Fixing Data Leakage) ---
    split_idx = int(len(Y) * 0.75) 
    X_mac_train, X_mac_test = X_macro[:split_idx], X_macro[split_idx:]
    X_mic_train, X_mic_test = X_micro[:split_idx], X_micro[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    vram_usage = []
    losses = []
    
    print(f"Starting {run_type}...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        optimizer.zero_grad()
        
        # Train ONLY on the Training Set
        outputs = model(X_mac_train, X_mic_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        
        epoch_time = time.time() - start_time
        losses.append(loss.item())
        
        vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.01
        vram_usage.append(vram)
        log_telemetry(run_type, epoch + 1, loss.item(), vram, epoch_time)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # Save Model Weights
    torch.save(model.state_dict(), os.path.join(run_dir, "model_weights.pt"))
    
    model.eval()
    with torch.no_grad():
        # Evaluate ONLY on the unseen Test Set
        predictions = model(X_mac_test, X_mic_test).numpy().flatten()
        
    mse = mean_squared_error(Y_test.numpy().flatten(), predictions)
    
    # Save Learning Curve Plot
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title(f"Learning Curve: {run_type}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(run_dir, "learning_curve.png"))
    plt.close()

    return predictions, mse, vram_usage, run_dir, split_idx

# Scaffolding
os.makedirs("runs", exist_ok=True)
init_telemetry()

# Generate Realistic Data
print("Generating experimental data for Ablation Study...")
time_steps = 400
t = np.linspace(0, 20, time_steps)
seasonal_trend = np.sin(t * 0.5) 
sudden_spike = np.zeros(time_steps)
sudden_spike[320:330] = 2.0 # Spike happens in the "future" (test set)
actual_data = seasonal_trend + sudden_spike + np.random.normal(0, 0.05, time_steps)

X_macro_list, X_micro_list, Y_list = [], [], []
for i in range(104, time_steps - 1):
    m_seq = np.zeros((104, 300))
    m_seq[:, 0] = seasonal_trend[i-104:i]
    mi_seq = np.zeros((60, 300))
    mi_seq[:, 0] = actual_data[i-60:i]
    X_macro_list.append(m_seq)
    X_micro_list.append(mi_seq)
    Y_list.append(actual_data[i+1])

X_macro = torch.tensor(np.array(X_macro_list), dtype=torch.float32)
X_micro = torch.tensor(np.array(X_micro_list), dtype=torch.float32)
Y = torch.tensor(np.array(Y_list), dtype=torch.float32).view(-1, 1)

print("\n--- Starting Automated Research Runs (with Train/Test Split) ---")
results = {}

# Run 1: Micro-Only
results['Micro-Only'] = run_experiment("Micro_Only", False, True, X_macro, X_micro, Y)

# Run 2: Macro-Only
results['Macro-Only'] = run_experiment("Macro_Only", True, False, X_macro, X_micro, Y)

# Run 3: Dual-Stream
results['Dual-Stream'] = run_experiment("Dual_Stream", True, True, X_macro, X_micro, Y)

# Final Accuracy Plot with Data Alignment
plt.figure(figsize=(14, 7))
actuals = Y.numpy().flatten()
plt.plot(actuals, label='Actual Data (Full)', color='black', alpha=0.3)

# Add Vertical Line for Train/Test Cutoff
split_idx = results['Dual-Stream'][4]
plt.axvline(x=split_idx, color='red', linestyle='--', label='Train/Test Cutoff')

for name, (pred, mse, vram, r_dir, s_idx) in results.items():
    # Pad predictions with NaN to align with the original timeline
    padded_preds = np.full(len(actuals), np.nan)
    padded_preds[s_idx:] = pred
    
    color = 'green' if 'Dual' in name else None
    linewidth = 2.5 if 'Dual' in name else 1.5
    plt.plot(padded_preds, label=f'{name} (MSE: {mse:.4f})', linewidth=linewidth, color=color)

plt.title("M-PULSE: Forecasting Unseen Future Data (Ablation Study)")
plt.xlabel("Time Step")
plt.ylabel("Sentiment / Trend Volume")
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig("final_research_comparison.png")
print("\nFinal aligned comparison chart saved to final_research_comparison.png")

# Final MSE Table
print("\n" + "="*60)
print(f"{'Configuration':<25} | {'Test MSE':<15}")
print("-" * 60)
for name, (pred, mse, vram, r_dir, s_idx) in results.items():
    print(f"{name:<25} | {mse:.6f}")
    print(f"  -> Data saved in: {r_dir}")
print("="*60)
