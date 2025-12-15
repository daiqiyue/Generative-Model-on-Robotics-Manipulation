import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=32, hidden_dim=256):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state, action):
        # encoder
        x = torch.cat([state, action], dim=1)
        feat = self.encoder(x)
        mu = self.fc_mu(feat)
        log_var = self.fc_var(feat)
        
        # Sample z
        z = self.reparameterize(mu, log_var)
        
        # Decoder
        # Condition on state
        decoder_input = torch.cat([state, z], dim=1)
        recon_action = self.decoder(decoder_input)
        
        return recon_action, mu, log_var
    
    def inference(self, state):
        """ 推理模式: 直接从标准正态分布采样 z """
        batch_size = state.size(0)
        device = state.device
        z = torch.randn(batch_size, self.latent_dim).to(device)
        
        decoder_input = torch.cat([state, z], dim=1)
        action = self.decoder(decoder_input)
        return action