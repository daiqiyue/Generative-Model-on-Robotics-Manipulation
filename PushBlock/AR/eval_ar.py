import torch
import numpy as np
import pybullet as p
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_ar import ARPolicy
from collect_data import ManualCollectEnv 

# relative observation
def get_relative_obs(raw_obs):
    ee_pos = raw_obs[0:3]
    block_pos = raw_obs[3:6]
    target_pos = raw_obs[6:8]
    rel_ee_block = block_pos - ee_pos
    rel_block_target = target_pos - block_pos[:2]
    ee_z = [ee_pos[2]]
    return np.concatenate([rel_ee_block, rel_block_target, ee_z])

def evaluate():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "ar_policy_clean_human.pth"
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE)
    obs_mean = checkpoint["obs_mean"].to(DEVICE)
    obs_std = checkpoint["obs_std"].to(DEVICE)
    action_mean = checkpoint["action_mean"].to(DEVICE)
    action_std = checkpoint["action_std"].to(DEVICE)
    state_dim = checkpoint["state_dim"]
    num_bins = checkpoint["num_bins"]
    model = ARPolicy(state_dim=state_dim, action_dim=3, num_bins=num_bins, hidden_dim=512).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    print(f"AR Policy Loaded. Bins: {num_bins}")
    
    env = ManualCollectEnv()
    n_test = 10
    
    for i in range(n_test):
        obs = env.reset() 
        print(f"Test {i+1} Start...")
        time.sleep(0.5)
            
        done = False
        step = 0
        last_action = np.zeros(3)
        smoothing = 0.4
        
        while not done and step < 1200:
            obs_tensor = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
            # normalize
            obs_norm = (obs_tensor - obs_mean) / obs_std
            with torch.no_grad():
                action_norm = model.sample(obs_norm, temperature = 0.8) # [1, 3]
            # de-normalize
            raw_action = action_norm * action_std + action_mean
            raw_action = raw_action.cpu().numpy()[0]
            
            action = smoothing * raw_action + (1 - smoothing) * last_action
            last_action = action
            
            obs, _, done, _ = env.step(action)
            step += 1
            
        if done: print("Success!")
        else: print("Failed")
            
    env.close()

if __name__ == "__main__":
    evaluate()