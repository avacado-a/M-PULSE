import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import logging
from typing import Tuple

from mpulse.models.networks import MPulseNet
from mpulse.data.dataset import create_dataloaders

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles the training lifecycle of the MPulseNet architecture.
    """
    def __init__(self, device: torch.device):
        self.device = device

    def train_evaluate(self, X_mac_arr, X_mic_arr, Y_arr, run_name: str, use_macro: bool, use_micro: bool, 
                       epochs: int = 150) -> Tuple[list, float]:
        """
        Trains the model and returns the predictions and MSE for the ablation study.
        """
        logger.info(f"Initializing {run_name} Architecture")
        
        train_loader, test_loader, split_idx = create_dataloaders(X_mac_arr, X_mic_arr, Y_arr)
        
        model = MPulseNet(use_macro=use_macro, use_micro=use_micro).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        model.train()
        for epoch in range(epochs):
            for x_mac, x_mic, y in train_loader:
                x_mac, x_mic, y = x_mac.to(self.device), x_mic.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(x_mac, x_mic)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()
                
        # Evaluation Phase
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_mac, x_mic, y in test_loader:
                x_mac, x_mic = x_mac.to(self.device), x_mic.to(self.device)
                preds = model(x_mac, x_mic)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(y.numpy().flatten())
                
        if len(all_targets) == 0:
            logger.warning(f"No test data available for evaluation in {run_name}")
            return [], float('inf')
            
        mse = mean_squared_error(all_targets, all_preds)
        logger.info(f"{run_name} Test MSE: {mse:.4f}")
        
        return all_preds, mse
