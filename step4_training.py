import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from step3_model import MPulseNet

print("The Sine Wave Test: Training Loop Evaluation")

time_steps = 200
t = np.linspace(0, 10, time_steps)
actual_volume = np.sin(t) + np.random.normal(0, 0.1, time_steps)

X_macro = []
X_micro = []
Y = []

for i in range(104, time_steps - 1):
    macro_seq = np.zeros((104, 300))
    macro_seq[:, 0] = actual_volume[i-104:i]
    
    micro_seq = np.zeros((60, 300))
    micro_seq[:, 0] = actual_volume[i-60:i]
    
    X_macro.append(macro_seq)
    X_micro.append(micro_seq)
    Y.append(actual_volume[i+1])

X_macro = torch.tensor(X_macro, dtype=torch.float32)
X_micro = torch.tensor(X_micro, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

model = MPulseNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50
print("Starting training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_macro, X_micro)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions = model(X_macro, X_micro).numpy().flatten()

actuals = Y.numpy().flatten()
mse = mean_squared_error(actuals, predictions)
print(f"Final Mean Squared Error: {mse:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(actuals, label='Actual Volume (Noisy Sine)', color='blue')
plt.plot(predictions, label='Model Prediction', color='red', linestyle='dashed')
plt.title("Step 4: The Sine Wave Test")
plt.xlabel("Time Step")
plt.ylabel("Volume / Trend")
plt.legend()
plt.grid(True)
plt.savefig("step4_evaluation_chart.png")
print("Saved chart to step4_evaluation_chart.png")
# plt.show() # Commented out so it doesn't block automated execution
