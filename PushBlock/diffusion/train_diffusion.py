import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import PushBlockDataset
from model_diffusion import DiffusionModel
import numpy as np
import matplotlib.pyplot as plt  
import argparse

version = "clean"  
def train(dataset_path, resume_path=None):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    # Hyperparameters
    BATCH_SIZE = 256
    EPOCHS = 100  
    LR = 1e-3           
    
    print(f"Using device: {DEVICE}")
    
    full_dataset = PushBlockDataset(dataset_path, device=DEVICE)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    model = DiffusionModel(state_dim=8, action_dim=3, hidden_dim=512, n_timesteps=100, device=DEVICE)
    best_val_loss = float('inf')

    # Checkpoint loading
    if resume_path:
        if os.path.exists(resume_path):
            print(f"Loading checkpoint from: {resume_path} ...")
            checkpoint = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state'])
            print("Model weights loaded successfully! Resuming training...")
            
        else:
            print(f"Resume path specified but file not found: {resume_path}")
            return
    else:
        print("Starting training from scratch...")


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print(f"Start Training!")
    
    loss_history = []
    loss_val_history = []
    for epoch in range(EPOCHS):
        model.train() 
        total_loss = 0
        for obs_batch, action_batch in train_loader:
            optimizer.zero_grad()
            loss = model.compute_loss(obs_batch, action_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval() 
        total_val_loss = 0
        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                loss = model.compute_loss(obs_batch, action_batch)
                total_val_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        avg_val_loss = total_val_loss / len(val_loader)
        loss_val_history.append(avg_val_loss)
        
        if (epoch+1) % 50 == 0: 
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss} | LR: {scheduler.get_last_lr()[0]:.5f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # save best model
            save_info = {
                "model_state": model.state_dict(),
                "obs_mean": full_dataset.obs_mean,
                "obs_std": full_dataset.obs_std,
                "action_mean": full_dataset.action_mean,
                "action_std": full_dataset.action_std
            }
            torch.save(save_info, f"diffusion_policy_{version}_best.pth")
    save_info = {
        "model_state": model.state_dict(),
        "obs_mean": full_dataset.obs_mean,
        "obs_std": full_dataset.obs_std,
        "action_mean": full_dataset.action_mean,
        "action_std": full_dataset.action_std
    }
    torch.save(save_info, f"diffusion_policy_{version}.pth")
    print(f"Model saved to diffusion_policy_{version}.pth")

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss', color='blue')
    plt.title('Diffusion Policy Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f'training_loss_diffusion_{version}.png')
    print("Loss plot saved to 'training_loss.png'")
    # plt.show() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Model for Push Block Task")
    parser.add_argument("--dataset", type=str, default=f"../dataset/{version}_data.npz", help="Path to the training dataset")
    parser.add_argument("--resume", type =str, default=None, help="Path to resume training from a checkpoint")
    args = parser.parse_args()
    train(args.dataset, args.resume)