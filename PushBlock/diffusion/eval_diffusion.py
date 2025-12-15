import torch
import numpy as np
import pybullet as p
import time
import math
import os  
import sys  
from model_diffusion import DiffusionModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collect_data import ManualCollectEnv 

def evaluate():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    model_path = "diffusion_policy_clean_best.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    obs_mean = checkpoint["obs_mean"].to(DEVICE)
    obs_std = checkpoint["obs_std"].to(DEVICE)
    action_mean = checkpoint["action_mean"].to(DEVICE)
    action_std = checkpoint["action_std"].to(DEVICE)
    
    # Initialize model (State dim = 8)
    model = DiffusionModel(state_dim=8, action_dim=3, hidden_dim=512, n_timesteps=100, device=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval() 
    
    print("Model loaded successfully.")
    
    # initialize environment
    env = ManualCollectEnv() 
    
    n_test = 10
    success_count = 0
    
    for i in range(n_test):
        obs = env.reset()
        print(f"\n=== Test Episode {i+1}/{n_test} ===")
        time.sleep(0.5) 
        done = False
        step = 0
        max_steps = 400 
        
        while not done and step < max_steps:
            obs_tensor = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
            obs_norm = (obs_tensor - obs_mean) / obs_std
            
            # Inference time record
            t0 = time.time()
            with torch.no_grad():
                action_norm = model.sample(obs_norm) 
            dt = time.time() - t0
            
            action = action_norm * action_std + action_mean
            action = action.cpu().numpy()[0]
            
            # Action repeat for smoother execution
            repeat_steps = 10  
            
            for _ in range(repeat_steps):
                obs, _, done, _ = env.step(action)
                if done: break 
            step += 1
            # Print inference time
            if step % 10 == 0:  
                print(f"Step {step}, Inference Time: {dt:.3f}s") 
            
            if step % 50 == 0:
                print(f"Step {step}")
        
        if done:
            print("SUCCESS!")
            success_count += 1
            for _ in range(50): p.stepSimulation(); time.sleep(0.01)
        else:
            print("FAILED")
            
    print(f"\nFinal Result: {success_count}/{n_test} Success Rate")
    env.close()

if __name__ == "__main__":
    evaluate()