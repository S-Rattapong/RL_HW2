"""Script to play a trained RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import random

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with a trained agent."""

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # ------------------------------------------------------------------ #
    # Device & naming
    # ------------------------------------------------------------------ #
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print("device:", device)

    task_name      = str(args_cli.task).split("-")[0]   # "Stabilize" or "SwingUp"
    
    # === เปลี่ยนชื่อ Algorithm ตรงนี้ได้เลย ===
    Algorithm_name = "PPO"                              
    n_episodes     = 10
    n_observations = 4

    # ------------------------------------------------------------------ #
    # Agent construction & loading
    # ------------------------------------------------------------------ #
    if Algorithm_name == "Linear_Q":
        from RL_Algorithm.Function_based.Linear_Q import Linear_QN as Algorithm
        agent = Algorithm(num_of_action=11, action_range=[-1.0, 1.0])
    elif Algorithm_name == "DQN":
        from RL_Algorithm.Function_based.DQN import DQN as Algorithm
        agent = Algorithm(device=device, num_of_action=11, action_range=[-1.0, 1.0], n_observations=n_observations, hidden_dim=128, dropout=0.0)
    elif Algorithm_name == "MC_REINFORCE":
        from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE as Algorithm
        agent = Algorithm(device=device, num_of_action=1, action_range=[-1.0, 1.0], n_observations=n_observations, hidden_dim=128, dropout=0.0, action_type="continuous")
    elif Algorithm_name == "AC":
        from RL_Algorithm.Function_based.AC import AC as Algorithm
        agent = Algorithm(device=device, num_of_action=1, action_range=[-1.0, 1.0], n_observations=n_observations, hidden_dims=[128, 128], activation="relu", action_type="continuous")
    elif Algorithm_name == "PPO":
        from RL_Algorithm.Function_based.PPO import PPO as Algorithm
        agent = Algorithm(device=device, num_of_action=1, action_range=[-1.0, 1.0], n_observations=n_observations, hidden_dims=[128, 128], activation="relu", action_type="continuous")

    # ปิดโหมดสุ่มสำหรับเล่นจริง
    agent.epsilon = 0.0

    model_dir      = os.path.join("model", task_name, Algorithm_name)
    model_filename = f"{Algorithm_name}_final.pth"
    if Algorithm_name == "Linear_Q": 
        model_filename = f"{Algorithm_name}_final.npy"

    agent.load_model(model_dir, model_filename)
    print(f"Loaded: {os.path.join(model_dir, model_filename)}")

    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():
        with torch.inference_mode():

            for episode in range(n_episodes):
                
                # ========= put your code here ========= #
                obs, _ = env.reset()
                if isinstance(obs, dict):
                    obs = obs["policy"]
                
                done = False
                steps = 0
                episode_reward = 0.0
                
                while not done and steps < 2000:
                    current_state = obs[0] if obs.dim() > 1 else obs
                    current_state = current_state.unsqueeze(0) if current_state.dim() == 1 else current_state

                    action, _ = agent.select_action(current_state)
                    env_action = action.unsqueeze(0) if action.dim() == 1 else action
                    
                    next_obs, reward, terminated, truncated, _ = env.step(env_action)
                    if isinstance(next_obs, dict):
                        next_obs = next_obs["policy"]
                        
                    done = terminated.item() or truncated.item()
                    obs = next_obs
                    steps += 1
                    episode_reward += reward.item()
                    
                    if args_cli.video:
                        timestep += 1
                        if timestep >= args_cli.video_length:
                            break
                            
                print(f"Play Episode {episode+1}: Survived {steps} steps | Total Reward: {episode_reward:.2f}")
                # ====================================== # 

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()