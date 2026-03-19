import torch
import torch.nn as nn

class MPulseNet(nn.Module):
    def __init__(self, macro_seq_len=104, micro_seq_len=60, feature_dim=300):
        super(MPulseNet, self).__init__()
        
        # Pathway A: Macro (LSTM)
        self.macro_lstm = nn.LSTM(input_size=feature_dim, hidden_size=64, batch_first=True)
        
        # Pathway B: Micro (1D Conv + LSTM)
        self.micro_conv = nn.Conv1d(in_channels=feature_dim, out_channels=128, kernel_size=3, padding=1)
        self.micro_relu = nn.ReLU()
        self.micro_lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        
        # Fusion Layer
        self.fc1 = nn.Linear(64 + 64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, macro_x, micro_x):
        # Pathway A
        _, (macro_hidden, _) = self.macro_lstm(macro_x)
        macro_out = macro_hidden[-1] # Shape: (batch, 64)
        
        # Pathway B
        micro_x = micro_x.permute(0, 2, 1) # Shape: (batch, 300, 60)
        micro_conv_out = self.micro_relu(self.micro_conv(micro_x))
        micro_conv_out = micro_conv_out.permute(0, 2, 1) # Shape: (batch, 60, 128)
        _, (micro_hidden, _) = self.micro_lstm(micro_conv_out)
        micro_out = micro_hidden[-1] # Shape: (batch, 64)
        
        # Fusion
        fused = torch.cat((macro_out, micro_out), dim=1) # Shape: (batch, 128)
        x = self.relu(self.fc1(fused))
        return self.fc2(x) # Shape: (batch, 1)

if __name__ == "__main__":
    print("Running The Dummy Tensor Test...")
    model = MPulseNet()
    macro_dummy = torch.randn(1, 104, 300)
    micro_dummy = torch.randn(1, 60, 300)
    
    print(f"Macro Dummy Shape: {macro_dummy.shape}")
    print(f"Micro Dummy Shape: {micro_dummy.shape}")
    
    try:
        output = model(macro_dummy, micro_dummy)
        print(f"Architecture structurally sound. Output: {output.item():.4f}")
    except Exception as e:
        print(f"Dimensionality Mismatch Error: {e}")
