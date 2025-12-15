import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import PushBlockDataset
from model_gan import Generator, Discriminator
import numpy as np

import argparse
import matplotlib.pyplot as plt
import torch.autograd as autograd

# Gradient Penalty
def compute_gradient_penalty(D, real_samples, fake_samples, state, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # random weight
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    # insert between real and fake
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(state, interpolates)
    fake = torch.ones((real_samples.size(0), 1)).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train(dataset_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 256
    EPOCHS = 300
    LR = 1e-4
    LATENT_DIM = 32
    HIDDEN_DIM = 512
    LAMBDA_GP = 10  
    N_CRITIC = 5    
    
    print(f"Using device: {DEVICE} (WGAN-GP Mode)")
    
    dataset = PushBlockDataset(dataset_path, device=DEVICE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    generator = Generator(state_dim=8, action_dim=3, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    discriminator = Discriminator(state_dim=8, action_dim=3, hidden_dim=HIDDEN_DIM).to(DEVICE)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.0, 0.9))
    
    g_losses = []
    d_losses = []
    
    for epoch in range(EPOCHS):
        for i, (real_state, real_action) in enumerate(dataloader):
            batch_size = real_state.size(0)
            
            # Discriminator
            optimizer_D.zero_grad()
            
            # Generate fake data
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_action = generator(real_state, z).detach() 
            
            real_validity = discriminator(real_state, real_action)
            fake_validity = discriminator(real_state, fake_action)
            gradient_penalty = compute_gradient_penalty(discriminator, real_action, fake_action, real_state, DEVICE)
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty
            
            d_loss.backward()
            optimizer_D.step()
            
            #  Train Generator
            if i % N_CRITIC == 0:
                optimizer_G.zero_grad()
                fake_action = generator(real_state, z)
                fake_validity = discriminator(real_state, fake_action)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

        print(f"Epoch {epoch+1}/{EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    save_info = {
        "model_state": generator.state_dict(),
        "obs_mean": dataset.obs_mean,
        "obs_std": dataset.obs_std,
        "action_mean": dataset.action_mean,
        "action_std": dataset.action_std,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM
    }
    torch.save(save_info, "wgan_policy.pth")
    np.savez("wgan_losses.npz", g_losses=np.array(g_losses), d_losses=np.array(d_losses))
    print("WGAN-GP Training Finished.")
    
    plt.figure(figsize=(10,6))
    plt.plot(g_losses, label="G Loss")
    plt.plot(d_losses, label="D Loss")
    plt.title("WGAN-GP Loss")
    plt.legend()
    plt.savefig("wgan_loss.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../dataset/clean_data.npz")
    args = parser.parse_args()
    train(args.dataset)