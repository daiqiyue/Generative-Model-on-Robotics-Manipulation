import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import PushBlockDataset
from model_ar import ARPolicy
import numpy as np

import argparse
import matplotlib.pyplot as plt

version = "clean_human"  

def train(dataset_path, resume_path=None):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    

    BATCH_SIZE = 2048
    EPOCHS = 500
    LR = 1e-4
    NUM_BINS = 256 
    
    print(f"Using device: {DEVICE}")
    
    full_dataset = PushBlockDataset(dataset_path, device=DEVICE)
    total_size = len(full_dataset)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False) 
    
    print(f"Data Split: {len(train_set)} Train, {len(val_set)} Val")
    
    state_dim = full_dataset.obs.shape[1]
    model = ARPolicy(state_dim=state_dim, action_dim=3, num_bins=NUM_BINS, hidden_dim=512).to(DEVICE)
    best_val_loss = float('inf')
    # Checkpoint loading
    if resume_path:
        if os.path.exists(resume_path):
            print(f"Loading checkpoint from: {resume_path} ...")
            checkpoint = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state'])
            print("Model weights loaded successfully! Resuming training...")
            
            if checkpoint.get('num_bins') != NUM_BINS:
                print("Warning: num_bins in checkpoint mismatch!")
        else:
            print(f"Resume path specified but file not found: {resume_path}")
            return
    else:
        print("Starting training from scratch...")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    loss_val_history = []
    print("Start Autoregressive Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_acc = 0
        
        for state, action in train_loader:
            optimizer.zero_grad()
            logits_x, logits_y, logits_z = model(state, action)
            target_tokens = model.encode(action) # [Batch, 3]
            
            loss_x = criterion(logits_x, target_tokens[:, 0])
            loss_y = criterion(logits_y, target_tokens[:, 1])
            loss_z = criterion(logits_z, target_tokens[:, 2])
            
            loss = loss_x + loss_y + loss_z
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
            with torch.no_grad():
                pred_x = torch.argmax(logits_x, dim=1)
                acc = (pred_x == target_tokens[:, 0]).float().mean()
                total_acc += acc.item()

        avg_loss = total_loss / len(train_loader)       
        avg_acc = total_acc / len(train_loader)        
        loss_history.append(avg_loss)

        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        with torch.no_grad(): 
            for state, action in val_loader:
                logits_x, logits_y, logits_z = model(state, action)
                target_tokens = model.encode(action)
                

                loss = criterion(logits_x, target_tokens[:, 0]) + \
                       criterion(logits_y, target_tokens[:, 1]) + \
                       criterion(logits_z, target_tokens[:, 2])
                total_val_loss += loss.item()
                
                pred_x = torch.argmax(logits_x, dim=1)
                acc = (pred_x == target_tokens[:, 0]).float().mean().item()
                total_val_acc += acc
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        loss_val_history.append(avg_val_loss)
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.5f} | Train Acc(x): {avg_acc:.2f}| Val Loss: {avg_val_loss:.5f} Val Acc(X): {avg_val_acc:.2%}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_info = {
                "model_state": model.state_dict(),
                "obs_mean": full_dataset.obs_mean,
                "obs_std": full_dataset.obs_std,
                "action_mean": full_dataset.action_mean,
                "action_std": full_dataset.action_std,
                "state_dim": state_dim,
                "num_bins": NUM_BINS
            }
            torch.save(save_info, f"ar_policy_{version}_best.pth")
    save_info = {
        "model_state": model.state_dict(),
        "obs_mean": full_dataset.obs_mean,
        "obs_std": full_dataset.obs_std,
        "action_mean": full_dataset.action_mean,
        "action_std": full_dataset.action_std,
        "state_dim": state_dim,
        "num_bins": NUM_BINS
    }
    torch.save(save_info, f"ar_policy_{version}.pth")
    print("Model saved.")
    
    plt.plot(loss_history)
    plt.title("Autoregressive Training Loss")
    plt.savefig(f"ar_{version}_loss.png")
    plt.plot(loss_val_history)
    plt.title("Autoregressive Validation Loss")
    plt.savefig(f"ar_{version}_val_loss.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=f"../dataset/{version}_data.npz")
    parser.add_argument("--resume", type =str, default=None, help="Path to resume training from a checkpoint")
    args = parser.parse_args()
    train(args.dataset, args.resume)