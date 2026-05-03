import torch
import torch.nn as nn

class MPulseNet(nn.Module):
    """
    Dual-Stream architecture for multi-resolution temporal forecasting.
    Fuses Macro (institutional) and Micro (ephemeral) latent states.
    """
    def __init__(self, use_macro: bool = True, use_micro: bool = True, feature_dim: int = 300):
        super(MPulseNet, self).__init__()
        self.use_macro = use_macro
        self.use_micro = use_micro
        
        self.macro_hidden_dim = 64
        self.micro_hidden_dim = 64
        
        if self.use_macro:
            self.macro_lstm = nn.LSTM(
                input_size=feature_dim, 
                hidden_size=self.macro_hidden_dim, 
                batch_first=True
            )
            
        if self.use_micro:
            self.micro_lstm = nn.LSTM(
                input_size=feature_dim, 
                hidden_size=self.micro_hidden_dim, 
                batch_first=True
            )
            
        # Determine fusion dimension dynamically
        combined_dim = 0
        if self.use_macro: combined_dim += self.macro_hidden_dim
        if self.use_micro: combined_dim += self.micro_hidden_dim
        
        if combined_dim == 0:
            raise ValueError("Model instantiated with both streams disabled.")
            
        self.fc1 = nn.Linear(combined_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x_mac: torch.Tensor, x_mic: torch.Tensor) -> torch.Tensor:
        features = []
        
        if self.use_macro:
            out_mac, _ = self.macro_lstm(x_mac)
            # Extract the latent state of the final timestep
            features.append(out_mac[:, -1, :]) 
            
        if self.use_micro:
            out_mic, _ = self.micro_lstm(x_mic)
            features.append(out_mic[:, -1, :])
            
        fused = torch.cat(features, dim=1)
        x = self.relu(self.fc1(fused))
        return self.fc2(x)
