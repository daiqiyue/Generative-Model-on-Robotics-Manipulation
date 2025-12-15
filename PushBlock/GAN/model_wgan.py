import torch
import torch.nn as nn

# Generator 
class Generator(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=32, hidden_dim=512):
        super(Generator, self).__init__()
        input_dim = state_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, z):
        x = torch.cat([state, z], dim=1)
        return self.net(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(Discriminator, self).__init__()
        input_dim = state_dim + action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)