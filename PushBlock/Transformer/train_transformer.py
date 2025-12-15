import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import PushBlockDatasetSequence
from model_transformer import TransformerARPolicy
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

version = 'human' 

def train(dataset_path, resume_path=None):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1024
    EPOCHS = 300 
    LR = 5e-4 
    CONTEXT_LEN = 5 
    
    print(f"Using device: {DEVICE}")
    full_dataset = PushBlockDatasetSequence(dataset_path, context_len=CONTEXT_LEN, device=DEVICE)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    model = TransformerARPolicy(state_dim=6, action_dim=3, context_len=CONTEXT_LEN).to(DEVICE)
    
    # Check if there is a checkpoint to resume from
    if resume_path:
        if os.path.exists(resume_path):
            print(f"🔄 Loading checkpoint from: {resume_path} ...")
            checkpoint = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state'])
            print("✅ Model weights loaded successfully! Resuming training...")
            
        else:
            print(f"❌ Resume path specified but file not found: {resume_path}")
            return
    else:
        print("🆕 Starting training from scratch...")


    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) # preventing overfitting 
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("Start Transformer AR Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for state, history, target_action in train_loader:
            optimizer.zero_grad()
            
            # Forward: [Batch, 3, 256]
            logits = model(state, history)
            
            # Label
            target_tokens = model.encode(target_action)
            
            # Loss: Sum of XYZ
            loss = criterion(logits[:, 0, :], target_tokens[:, 0]) + \
                   criterion(logits[:, 1, :], target_tokens[:, 1]) + \
                   criterion(logits[:, 2, :], target_tokens[:, 2])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # === Validation ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for state, history, target_action in val_loader:
                logits = model(state, history)
                target_tokens = model.encode(target_action)
                loss = criterion(logits[:, 0, :], target_tokens[:, 0]) + \
                       criterion(logits[:, 1, :], target_tokens[:, 1]) + \
                       criterion(logits[:, 2, :], target_tokens[:, 2])
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_info = {
                "model_state": model.state_dict(),
                "obs_mean": full_dataset.obs_mean,
                "obs_std": full_dataset.obs_std,
                "action_mean": full_dataset.action_mean,
                "action_std": full_dataset.action_std,
                "state_dim": 6,
                "context_len": CONTEXT_LEN
            }
            torch.save(save_info, f"transformer_policy_{version}_best.pth")

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.savefig(f"transformer_{version}_loss.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=f"../dataset/{version}_data.npz")
    parser.add_argument("--resume", type =str, default=None, help="Path to resume training from a checkpoint")
    args = parser.parse_args()
    train(args.dataset, args.resume)