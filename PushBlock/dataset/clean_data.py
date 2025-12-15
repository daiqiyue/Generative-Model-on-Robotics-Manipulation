import numpy as np

def clean_data(path="human_data.npz", threshold=0.02):
    print(f"Loading {path}...")
    data = np.load(path)
    obs = data['obs']
    actions = data['actions']
    
    # 计算动作的幅度 (Velocity Norm)
    # actions shape: [N, 3]
    vel_norm = np.linalg.norm(actions, axis=1)
    
    # 筛选：只保留速度大于阈值的帧
    # threshold 设为 0.02 左右，过滤掉极微小的抖动和静止
    mask = vel_norm > threshold
    
    clean_obs = obs[mask]
    clean_actions = actions[mask]
    
    print(f"Original frames: {len(obs)}")
    print(f"Cleaned frames:  {len(clean_obs)}")
    print(f"Removed: {len(obs) - len(clean_obs)} ({100*(1 - len(clean_obs)/len(obs)):.1f}%) stationary frames.")
    
    np.savez_compressed("clean_human_data.npz", obs=clean_obs, actions=clean_actions)
    print("Saved to clean_data.npz")

if __name__ == "__main__":
    clean_data("final_data.npz")