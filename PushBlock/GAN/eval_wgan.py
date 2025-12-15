import torch
import numpy as np
import pybullet as p
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_gan import Generator
from collect_data import ManualCollectEnv 

############
# Wassertein GAN



def evaluate():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "../model/wgan_policy_final.pth"
    
    if not os.path.exists(model_path):
        print("Model file not found.")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE)
    obs_mean = checkpoint["obs_mean"].to(DEVICE)
    obs_std = checkpoint["obs_std"].to(DEVICE)
    action_mean = checkpoint["action_mean"].to(DEVICE)
    action_std = checkpoint["action_std"].to(DEVICE)
    latent_dim = checkpoint["latent_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    
    # Generator Initialization
    model = Generator(state_dim=5, action_dim=3, latent_dim=latent_dim, hidden_dim=hidden_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    print("GAN Generator loaded.")

    # Env initialize
    env = ManualCollectEnv()
    n_test = 10
    success_count = 0
    
    for i in range(n_test):
        obs = env.reset()
        print(f"Test {i+1} - Press Space to start")
        
        time.sleep(0.5)
        done = False
        step = 0
        while not done and step < 1200:
            obs_tensor = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
            obs_norm = (obs_tensor - obs_mean) / obs_std
            
            fixed_z = torch.randn(1, latent_dim).to(DEVICE)
            with torch.no_grad():
                action_norm = model(obs_norm, fixed_z)
            
            action = action_norm * action_std + action_mean
            action = action.cpu().numpy()[0]
            for _ in range(3): 
                obs, _, done, _ = env.step(action)
                if done: break
            step += 1
        if done:
            print("Success!")
            success_count += 1
            time.sleep(0.5)
        else:
            print("Failed")
            
    print(f"Success Rate: {success_count}/{n_test}")
    env.close()

if __name__ == "__main__":
    evaluate()