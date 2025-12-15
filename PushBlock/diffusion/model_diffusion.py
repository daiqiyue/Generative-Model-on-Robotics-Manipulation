import torch
import torch.nn as nn
import numpy as np
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512): # 默认 512
        super().__init__()
        self.time_dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.Mish(),
        )

        input_dim = state_dim + action_dim + self.time_dim
        
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )
        
        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, action, state, time):
        t_emb = self.time_mlp(time)
        x = torch.cat([action, state, t_emb], dim=-1)
        x = self.mid_layer(x)
        return self.final_layer(x)


class DiffusionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, n_timesteps=100, device="cuda"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_timesteps = n_timesteps
        self.device = device
        
        self.model = DiffusionNetwork(state_dim, action_dim, hidden_dim).to(device)
        beta_start = 1e-4
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def compute_loss(self, state, action):
        batch_size = state.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(action)
        
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
        noisy_action = torch.sqrt(alpha_bar_t) * action + torch.sqrt(1 - alpha_bar_t) * noise
        
        pred_noise = self.model(noisy_action, state, t)
        return nn.MSELoss()(pred_noise, noise)

    def sample(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = torch.randn(shape, device=self.device)
        
        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                pred_noise = self.model(action, state, t)
            
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            noise_factor = beta / torch.sqrt(1 - alpha_bar)
            mean = (1 / torch.sqrt(alpha)) * (action - noise_factor * pred_noise)
            
            if i > 0:
                z = torch.randn_like(action)
                sigma = torch.sqrt(beta)
                action = mean + sigma * z
            else:
                action = mean
                
        return action