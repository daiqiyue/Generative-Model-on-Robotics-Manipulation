import torch
import numpy as np
from torch.utils.data import Dataset

class PushBlockDatasetSequence(Dataset):
    def __init__(self, data_path="human_data.npz", context_len=5, device="cpu"):
        self.context_len = context_len
        self.device = device
        
        data = np.load(data_path)
        obs_raw = data['obs'] 
        actions = data['actions']
        
        # relative coordinate
        rel_ee_block = obs_raw[:, 3:6] - obs_raw[:, 0:3]
        rel_block_target = obs_raw[:, 6:8] - obs_raw[:, 3:5]
        ee_z = obs_raw[:, 2:3]
        self.obs = np.concatenate([rel_ee_block, rel_block_target, ee_z], axis=1)
        self.obs = torch.from_numpy(self.obs).float()
        self.actions = torch.from_numpy(actions).float()
        
        # Normalization
        self.obs_mean = self.obs.mean(dim=0)
        self.obs_std = self.obs.std(dim=0) + 1e-6
        self.action_mean = self.actions.mean(dim=0)
        self.action_std = self.actions.std(dim=0) + 1e-6
        
        self.obs_norm = (self.obs - self.obs_mean) / self.obs_std
        self.actions_norm = (self.actions - self.action_mean) / self.action_std
        
        # Add 0
        self.pad_action = torch.zeros(context_len, 3) 
        self.padded_actions = torch.cat([self.pad_action, self.actions_norm], dim=0)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        state = self.obs_norm[idx]
        history = self.padded_actions[idx : idx + self.context_len]
        target_action = self.actions_norm[idx]
        
        return state.to(self.device), history.to(self.device), target_action.to(self.device)