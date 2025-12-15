import torch
import numpy as np
from torch.utils.data import Dataset

class PushBlockDataset(Dataset):
    def __init__(self, data_path="human_data.npz", device="cpu"):
        # load dataset
        data = np.load(data_path)
        self.obs = torch.from_numpy(data['obs']).float().to(device)
        self.actions = torch.from_numpy(data['actions']).float().to(device)
        
        # Normalization
        self.obs_mean = self.obs.mean(dim=0)
        self.obs_std = self.obs.std(dim=0) + 1e-6 
        
        self.action_mean = self.actions.mean(dim=0)
        self.action_std = self.actions.std(dim=0) + 1e-6
        self.obs_norm = (self.obs - self.obs_mean) / self.obs_std
        self.actions_norm = (self.actions - self.action_mean) / self.action_std
        
        print(f"Dataset Loaded. Samples: {len(self.obs)}")
        print(f"Obs Dim: {self.obs.shape[1]}, Action Dim: {self.actions.shape[1]}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs_norm[idx], self.actions_norm[idx]

    # de-normalization
    def unnormalize_action(self, action_norm):
        # action_norm: [Batch, 3]
        return action_norm * self.action_std + self.action_mean
        
    def normalize_obs(self, obs):
        # obs: [Batch, 8]
        return (obs - self.obs_mean) / self.obs_std