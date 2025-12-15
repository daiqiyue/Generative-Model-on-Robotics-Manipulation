import pybullet as p
import pybullet_data
import time
import numpy as np
import math

class ManualCollectEnv:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.5, 0, 0])
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.63])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.ee_index = 11 
        
        # Load block
        self.block_start_pos = [0.5, -0.2, 0.05]
        self.block_id = p.loadURDF("cube_small.urdf", basePosition=self.block_start_pos)

        self.home_joints = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4]
        
        # target zone
        self.target_pos = np.array([0.5, 0.2, 0.0])
        self.target_radius = 0.15
        
        # draw target zone
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=self.target_radius, length=0.002, rgbaColor=[0, 1, 0, 0.3], specularColor=[0,0,0])
        p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=[self.target_pos[0], self.target_pos[1], 0.001])
        p.addUserDebugText("TARGET ZONE", [self.target_pos[0], self.target_pos[1], 0.2], textColorRGB=[0, 1, 0], textSize=1.5)

    def reset(self):
        # reset robot to home position
        for i in range(7):
            p.resetJointState(self.robot_id, i, self.home_joints[i])
            p.resetJointState(self.robot_id, 9, 0.0)
            p.resetJointState(self.robot_id, 10, 0.0)
        # reset block to random position within a defined area
        rand_x = np.random.uniform(0.4, 0.6)
        rand_y = np.random.uniform(-0.25, -0.15)
        p.resetBasePositionAndOrientation(self.block_id, [rand_x, rand_y, 0.05], [0, 0, 0, 1])
        
        for _ in range(50): 
            p.stepSimulation()
            
        return self._get_obs()

    def _get_obs(self):
        ee_state = p.getLinkState(self.robot_id, self.ee_index)         # the end-effector state
        ee_pos = np.array(ee_state[0])                                 # end-effector position
        block_pos, _ = p.getBasePositionAndOrientation(self.block_id)  # block position
        block_pos = np.array(block_pos)                                 
        return np.concatenate([ee_pos, block_pos, self.target_pos[:2]])

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)              
        ee_state = p.getLinkState(self.robot_id, self.ee_index)
        current_pos = np.array(ee_state[0])

        new_pos = current_pos + action * 0.08
        if new_pos[2] < 0.0: new_pos[2] = 0.0
        
        orn = p.getQuaternionFromEuler([math.pi, 0, math.pi / 2])
        joint_poses = p.calculateInverseKinematics(self.robot_id, self.ee_index, new_pos, targetOrientation=orn)
        
        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=joint_poses[i])
            
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=0.0, force=10)
        p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, targetPosition=0.0, force=10)
        p.stepSimulation()
        time.sleep(1./240.)
        
        obs = self._get_obs()
        block_pos = obs[3:6]
        
        dist = np.linalg.norm(block_pos[:2] - self.target_pos[:2])
        done = False
        if dist < self.target_radius:
            done = True
            
        return obs, 0, done, {}
    
    def close(self):
        p.disconnect()

# Using Keyboard to control the robot
def get_keyboard_action():
    keys = p.getKeyboardEvents()
    vx, vy, vz = 0, 0, 0
    speed = 0.8
    
    if ord('j') in keys and keys[ord('j')] & p.KEY_IS_DOWN: vx = -speed
    if ord('l') in keys and keys[ord('l')] & p.KEY_IS_DOWN: vx = speed
    if ord('i') in keys and keys[ord('i')] & p.KEY_IS_DOWN: vy = speed
    if ord('k') in keys and keys[ord('k')] & p.KEY_IS_DOWN: vy = -speed
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN: vz = speed
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN: vz = -speed
        
    reset = False
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED: reset = True
    quit_sim = False
    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED: quit_sim = True
    
    # Type space to start recording
    start_key = False
    if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
        start_key = True

    return np.array([vx, vy, vz]), reset, quit_sim, start_key

def collect_human_demo(save_path="human_data1.npz", target_episodes=80):
    env = ManualCollectEnv()
    
    all_obs = []
    all_actions = []
    success_count = 0
    
    print("\n" + "="*50)
    print(" 🎮 HUMAN DATA COLLECTION (LAG FREE) ")
    print("="*50)
    print(" 1. Controls: [I/K/J/L] Move, [R] Reset, [Q] Quit")
    print(" 2. IMPORTANT: Press [SPACE] to start recording each episode!")
    print("="*50)
    
    while success_count < target_episodes:
        obs = env.reset()
        
        # waiting for press space
        print(f"\n[Episode {success_count + 1}] Ready. Waiting for SPACE to start...")
        
        waiting_for_start = True
        while waiting_for_start:
            p.stepSimulation() 
            keys = p.getKeyboardEvents()
            if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
                waiting_for_start = False
            time.sleep(0.01)
            
        print(">>> GO! Recording...")
        
        # start recording
        episode_obs = []
        episode_actions = []
        done = False
        discard = False
        
        while not done:
            action, reset_req, quit_req, _ = get_keyboard_action()
            
            if quit_req:
                env.close()
                save_data(all_obs, all_actions, save_path)
                return

            if reset_req:
                print("Discarding episode.")
                discard = True
                break
            
            episode_obs.append(obs)
            episode_actions.append(action)
            obs, _, done, _ = env.step(action)
            
        if not discard and done:
            print("Success!")
            all_obs.extend(episode_obs)
            all_actions.extend(episode_actions)
            success_count += 1
            time.sleep(0.5)
            
    env.close()
    save_data(all_obs, all_actions, save_path)

def save_data(obs_list, action_list, path):
    if len(obs_list) == 0: return
    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(action_list, dtype=np.float32)
    print(f"\nCollection Complete! Total Frames: {obs_arr.shape[0]}")
    np.savez_compressed(path, obs=obs_arr, actions=act_arr)
    print(f"Data saved to {path}")

if __name__ == "__main__":
    collect_human_demo()