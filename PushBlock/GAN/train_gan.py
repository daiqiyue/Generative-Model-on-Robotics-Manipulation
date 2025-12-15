import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import PushBlockDataset
from model_gan import Generator, Discriminator
import numpy as np
import argparse
import matplotlib.pyplot as plt

def train(dataset_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 512 
    EPOCHS = 300     
    LR = 2e-4        
    LATENT_DIM = 32  
    HIDDEN_DIM = 512
    print(f"Using device: {DEVICE}")
    print(f"Dataset: {dataset_path}")
    dataset = PushBlockDataset(dataset_path, device=DEVICE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    generator = Generator(state_dim=8, action_dim=3, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    discriminator = Discriminator(state_dim=8, action_dim=3, hidden_dim=HIDDEN_DIM).to(DEVICE)

    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    g_losses = []
    d_losses = []
    
    print("Start GAN Training...")
    
    for epoch in range(EPOCHS):
        total_g_loss = 0
        total_d_loss = 0
        
        for i, (real_state, real_action) in enumerate(dataloader):
            batch_size = real_state.size(0)
            
            label_real = torch.ones(batch_size, 1).to(DEVICE)
            label_fake = torch.zeros(batch_size, 1).to(DEVICE)
            optimizer_D.zero_grad()
            pred_real = discriminator(real_state, real_action)
            loss_d_real = criterion(pred_real, label_real)
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_action = generator(real_state, z)
            
            pred_fake = discriminator(real_state, fake_action.detach()) 
            loss_d_fake = criterion(pred_fake, label_fake)
            
            # total loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            pred_fake_for_g = discriminator(real_state, fake_action)
            loss_g = criterion(pred_fake_for_g, label_real) 
            loss_g.backward()
            optimizer_G.step()
            
            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            
        avg_g_loss = total_g_loss / len(dataloader)
        avg_d_loss = total_d_loss / len(dataloader)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")
            
    save_info = {
        "model_state": generator.state_dict(),
        "obs_mean": dataset.obs_mean,
        "obs_std": dataset.obs_std,
        "action_mean": dataset.action_mean,
        "action_std": dataset.action_std,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM
    }
    torch.save(save_info, "gan_policy.pth")
    print("Model saved to gan_policy.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title('GAN Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('gan_training_loss.png')
    print("Plot saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="..\dataset\\final_data.npz")
    args = parser.parse_args()
    train(args.dataset)