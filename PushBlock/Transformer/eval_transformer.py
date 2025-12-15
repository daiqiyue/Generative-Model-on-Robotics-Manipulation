import torch
import numpy as np
import pybullet as p
import time
import os
from model_transformer import TransformerARPolicy
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
    model_path = "transformer_policy_human_best.pth"
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE)
    obs_mean = checkpoint["obs_mean"].to(DEVICE)
    obs_std = checkpoint["obs_std"].to(DEVICE)
    action_mean = checkpoint["action_mean"].to(DEVICE)
    action_std = checkpoint["action_std"].to(DEVICE)
    state_dim = checkpoint["state_dim"]
    context_len = checkpoint["context_len"]
    
    model = TransformerARPolicy(state_dim=state_dim, action_dim=3, context_len=context_len).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    print(f"Transformer Policy Loaded. Context Length: {context_len}")
    
    env = ManualCollectEnv()
    n_test = 10
    
    for i in range(n_test):
        obs = env.reset()
        print(f"Test {i+1} Start...")
        time.sleep(0.5)
            
        done = False
        step = 0
        
        # Initialize action history with all zero
        action_history = torch.zeros(1, context_len, 3).to(DEVICE)
        
        last_action = np.zeros(3)
        smoothing = 0.4
        
        while not done and step < 1200:
            obs_rel = get_relative_obs(obs)
            obs_tensor = torch.from_numpy(obs_rel).float().to(DEVICE).unsqueeze(0)
            obs_norm = (obs_tensor - obs_mean) / obs_std
            
            with torch.no_grad():
                # temperature 0.8
                action_norm = model.sample(obs_norm, action_history, temperature=0.8)
            
            # update
            new_action_tensor = action_norm.unsqueeze(1)
            action_history = torch.cat([action_history[:, 1:, :], new_action_tensor], dim=1)
            # de-normalization
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