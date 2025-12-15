import torch
import torch.nn as nn
import numpy as np

class TransformerARPolicy(nn.Module):
    def __init__(self, state_dim, action_dim=3, context_len=5, num_bins=256, embed_dim=256, num_layers=4, nhead=4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.num_bins = num_bins
        self.embed_dim = embed_dim
        
        # range of actions
        self.min_val = -3.0
        self.max_val = 3.0
        self.state_emb = nn.Linear(state_dim, embed_dim)
        self.action_emb = nn.Linear(action_dim, embed_dim)
        
        # position encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + context_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.predict_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, action_dim * num_bins)
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

    def forward(self, state, action_history):
        batch_size = state.shape[0]
        # [Batch, state_dim] -> [Batch, 1, embed_dim]
        state_token = self.state_emb(state).unsqueeze(1)
        # [Batch, K, action_dim] -> [Batch, K, embed_dim]
        history_tokens = self.action_emb(action_history)
        # Shape: [Batch, K+1, embed_dim]
        seq = torch.cat([state_token, history_tokens], dim=1)
        seq = seq + self.pos_emb
        feat = self.transformer(seq)
        last_token_feat = feat[:, -1, :] # [Batch, embed_dim]
        logits_all = self.predict_head(last_token_feat) # [Batch, 3*256]
        logits_reshaped = logits_all.view(batch_size, self.action_dim, self.num_bins)
        
        return logits_reshaped
    
    def sample(self, state, action_history, temperature=1.0):
        logits = self.forward(state, action_history) # [B, 3, 256]
        pred_tokens = []
        for i in range(self.action_dim):
            dim_logits = logits[:, i, :]
            probs = torch.nn.functional.softmax(dim_logits / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            pred_tokens.append(token)
            
        tokens = torch.stack(pred_tokens, dim=1)
        action = self.decode(tokens)
        return action