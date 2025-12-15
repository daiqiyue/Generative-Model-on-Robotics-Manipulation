import numpy as np
import os

def merge_datasets(file_list, output_name="merged_data.npz"):
    all_obs = []
    all_actions = []
    
    print(f"Start merging {len(file_list)} files...")
    
    for filename in file_list:
        if not os.path.exists(filename):
            print(f"⚠️ Warning: {filename} not found, skipping.")
            continue
            
        data = np.load(filename)
        obs = data['obs']
        actions = data['actions']
        
        print(f" -> Loaded {filename}: {len(obs)} frames")
        
        all_obs.append(obs)
        all_actions.append(actions)
    
    if len(all_obs) == 0:
        print("No data loaded.")
        return

    # concatenate data along axis 0 (time dimension)
    final_obs = np.concatenate(all_obs, axis=0)
    final_actions = np.concatenate(all_actions, axis=0)
    
    print("-" * 30)
    print(f"Merge Complete!")
    print(f"Total Frames: {final_obs.shape[0]}")
    np.savez_compressed(output_name, obs=final_obs, actions=final_actions)
    print(f"Saved to: {output_name}")

if __name__ == "__main__":
    # dataset to be merged
    files_to_merge = [
        "./dataset\human_data.npz",       
        "./dataset/human_data1.npz", 
        "./dataset/human_data80.npz"
    ]
    
    merge_datasets(files_to_merge, output_name="final_data.npz")