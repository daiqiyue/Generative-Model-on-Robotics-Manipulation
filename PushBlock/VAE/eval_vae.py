import torch
import numpy as np
import pybullet as p
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_cvae import VAE
from collect_data import ManualCollectEnv 

def evaluate():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "vae_policy_clean_human_best.pth"
    
    if not os.path.exists(model_path):
        print("Model file not found.")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE)
    obs_mean = checkpoint["obs_mean"].to(DEVICE)
    obs_std = checkpoint["obs_std"].to(DEVICE)
    action_mean = checkpoint["action_mean"].to(DEVICE)
    action_std = checkpoint["action_std"].to(DEVICE)
    
    model = VAE(state_dim=8, action_dim=3, latent_dim=32, hidden_dim=512).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    print("VAE Policy loaded (Absolute Coords).")
    
    env = ManualCollectEnv()
    n_test = 10
    
    for i in range(n_test):
        obs = env.reset() 
        print(f"Test {i+1} Start...")
        
        time.sleep(0.01)
            
        done = False
        step = 0
        last_action = np.zeros(3)
        smoothing = 0.4
        
        while not done and step < 1200:
            obs_tensor = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
            obs_norm = (obs_tensor - obs_mean) / obs_std
            
            with torch.no_grad():
                action_norm = model.inference(obs_norm)
            
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