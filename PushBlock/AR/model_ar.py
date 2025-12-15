import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ARPolicy(nn.Module):
    def __init__(self, state_dim, action_dim=3, num_bins=256, hidden_dim=512):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_bins = num_bins
        
        self.min_val = -3.0
        self.max_val = 3.0
        self.bin_width = (self.max_val - self.min_val) / num_bins

        self.net_x = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, num_bins) 
        )
        
        self.emb_x = nn.Embedding(num_bins, hidden_dim)
        self.net_y = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, num_bins)
        )
        
        self.emb_y = nn.Embedding(num_bins, hidden_dim)
        self.net_z = nn.Sequential(
            nn.Linear(state_dim + hidden_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, num_bins)
        )

    def encode(self, action):
        action = torch.clamp(action, self.min_val, self.max_val)
        action_norm = (action - self.min_val) / (self.max_val - self.min_val)
        token_indices = (action_norm * (self.num_bins - 1)).long()
        return token_indices


    def decode(self, token_indices):
        action_norm = token_indices.float() / (self.num_bins - 1)
        action = action_norm * (self.max_val - self.min_val) + self.min_val
        return action

    def forward(self, state, action=None):

        tokens = self.encode(action)
        token_x = tokens[:, 0]
        token_y = tokens[:, 1]
        logits_x = self.net_x(state)
        emb_x = self.emb_x(token_x)
        logits_y = self.net_y(torch.cat([state, emb_x], dim=1))
        emb_y = self.emb_y(token_y)
        logits_z = self.net_z(torch.cat([state, emb_x, emb_y], dim=1))
        
        return logits_x, logits_y, logits_z

    def sample(self, state, temperature=1.2):
        batch_size = state.size(0)
        device = state.device
        

        def sample_token(logits):
            probs = F.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            return token.squeeze(-1) # [Batch]

        logits_x = self.net_x(state)
        token_x = sample_token(logits_x)
        emb_x = self.emb_x(token_x)
        #predict y
        logits_y = self.net_y(torch.cat([state, emb_x], dim=1))
        token_y = sample_token(logits_y)
        # predict z
        emb_y = self.emb_y(token_y)
        logits_z = self.net_z(torch.cat([state, emb_x, emb_y], dim=1))
        token_z = sample_token(logits_z)
        tokens = torch.stack([token_x, token_y, token_z], dim=1)
        action = self.decode(tokens)
        return action