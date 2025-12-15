import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import PushBlockDataset
from model_cvae import VAE
import numpy as np

import argparse
import matplotlib.pyplot as plt

version = 'clean_human'

def loss_function(recon_x, x, mu, log_var, kl_weight=0.0001):
    """ VAE Loss = MSE + KL Divergence """
    mse = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kld = kld / x.size(0)
    
    return mse + kl_weight * kld, mse, kld

def train(dataset_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    BATCH_SIZE = 4096
    EPOCHS = 500
    LR = 1e-3
    KL_WEIGHT = 0.01 
    
    print(f"Using device: {DEVICE}")
    
    full_dataset = PushBlockDataset(dataset_path, device=DEVICE)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)   
    model = VAE(state_dim=8, action_dim=3, latent_dim=32, hidden_dim=512).to(DEVICE)
    best_val_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    loss_history = []
    loss_val_history = []
    print("Start VAE Training (Absolute Coords)...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for state, action in train_loader:
            optimizer.zero_grad()
            recon_action, mu, log_var = model(state, action)
            loss, mse, kld = loss_function(recon_action, action, mu, log_var, KL_WEIGHT)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for state, action in val_loader:
                recon_action, mu, log_var = model(state, action)
                loss, mse, kld = loss_function(recon_action, action, mu, log_var, KL_WEIGHT)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        loss_val_history.append(avg_val_loss)
        loss_history.append(avg_train_loss)
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_info = {
                "model_state": model.state_dict(),
                "obs_mean": full_dataset.obs_mean,
                "obs_std": full_dataset.obs_std,
                "action_mean": full_dataset.action_mean,
                "action_std": full_dataset.action_std
            }
            torch.save(save_info, f"vae_policy_{version}_best.pth")
    save_info = {
        "model_state": model.state_dict(),
        "obs_mean": full_dataset.obs_mean,
        "obs_std": full_dataset.obs_std,
        "action_mean": full_dataset.action_mean,
        "action_std": full_dataset.action_std
    }
    torch.save(save_info, f"vae_policy{version}.pth")
    print("Model saved.")
    
    plt.plot(loss_history)
    plt.savefig(f"vae_loss_{version}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=f"../dataset/{version}_data.npz")
    args = parser.parse_args()
    train(args.dataset)