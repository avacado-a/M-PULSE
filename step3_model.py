import torch
import torch.nn as nn

class MPulseNet(nn.Module):
    def __init__(self, use_macro=True, use_micro=True, feature_dim=300, seq_len=7):
        super(MPulseNet, self).__init__()
        self.use_macro = use_macro
        self.use_micro = use_micro
        
        if self.use_macro:
            self.macro_lstm = nn.LSTM(input_size=feature_dim, hidden_size=64, batch_first=True)
            
        if self.use_micro:
            # Padding=1 ensures the 7-day sequence doesn't collapse through the CNN
            self.micro_cnn = nn.Conv1d(in_channels=feature_dim, out_channels=128, kernel_size=3, padding=1)
            self.micro_lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
            
        combined_dim = 0
        if self.use_macro: combined_dim += 64
        if self.use_micro: combined_dim += 64
        if combined_dim == 0: combined_dim = 1 # Fallback
        
        self.fc1 = nn.Linear(combined_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x_mac, x_mic):
        batch_size = x_mac.size(0) if self.use_macro else x_mic.size(0)
        features = []
        
        if self.use_macro:
            out_mac, _ = self.macro_lstm(x_mac)
            features.append(out_mac[:, -1, :]) # Grab the final day's prediction state
            
        if self.use_micro:
            x_mic_cnn = x_mic.transpose(1, 2)
            out_mic_cnn = self.relu(self.micro_cnn(x_mic_cnn))
            out_mic_cnn = out_mic_cnn.transpose(1, 2)
            out_mic, _ = self.micro_lstm(out_mic_cnn)
            features.append(out_mic[:, -1, :])
            
        if features:
            fused = torch.cat(features, dim=1)
        else:
            fused = torch.zeros(batch_size, 1, device=x_mac.device)
            
        return self.fc2(self.relu(self.fc1(fused)))
