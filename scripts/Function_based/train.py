"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
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
import matplotlib
import matplotlib.pyplot as plt

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
    """Train with stable-baselines agent."""

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
    n_episodes     = 1000
    n_observations = 4

    # ------------------------------------------------------------------ #
    # Agent construction (Auto Selector)
    # ------------------------------------------------------------------ #
    if Algorithm_name == "Linear_Q":
        from RL_Algorithm.Function_based.Linear_Q import Linear_QN as Algorithm
        agent = Algorithm(num_of_action=11, action_range=[-1.0, 1.0], learning_rate=0.005, discount_factor=0.99, initial_epsilon=1.0, epsilon_decay=0.995, final_epsilon=0.01)

    elif Algorithm_name == "DQN":
        from RL_Algorithm.Function_based.DQN import DQN as Algorithm
        agent = Algorithm(device=device, num_of_action=11, action_range=[-1.0, 1.0], n_observations=n_observations, hidden_dim=128, dropout=0.0, learning_rate=1e-3, tau=0.005, initial_epsilon=1.0, epsilon_decay=0.995, final_epsilon=0.01, discount_factor=0.99, buffer_size=10000, batch_size=128)

    elif Algorithm_name == "MC_REINFORCE":
        from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE as Algorithm
        agent = Algorithm(device=device, num_of_action=1, action_range=[-1.0, 1.0], n_observations=n_observations, hidden_dim=128, dropout=0.0, action_type="continuous", learning_rate=1e-3, discount_factor=0.99)

    elif Algorithm_name == "AC":
        from RL_Algorithm.Function_based.AC import AC as Algorithm
        agent = Algorithm(device=device, num_of_action=1, action_range=[-1.0, 1.0], n_observations=n_observations, hidden_dims=[128, 128], activation="relu", action_type="continuous", init_noise_std=1.0, learning_rate=1e-3, discount_factor=0.99, value_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5)

    elif Algorithm_name == "PPO":
        from RL_Algorithm.Function_based.PPO import PPO as Algorithm
        agent = Algorithm(device=device, num_of_action=1, action_range=[-1.0, 1.0], n_observations=n_observations, hidden_dims=[128, 128], activation="relu", action_type="continuous", init_noise_std=1.0, num_learning_epochs=4, num_mini_batches=4, clip_param=0.2, gamma=0.99, lam=0.95, value_loss_coef=0.5, entropy_coef=0.01, learning_rate=1e-3, max_grad_norm=0.5, desired_kl=0.01)

    # ------------------------------------------------------------------ #
    # Save path — checkpoints saved every save_interval episodes
    # ------------------------------------------------------------------ #
    save_interval = 250
    model_dir     = os.path.join("model", task_name, Algorithm_name)
    os.makedirs(model_dir, exist_ok=True)

    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():

        for episode in tqdm(range(n_episodes)):

            # ========= put your code here ========= #
            if Algorithm_name == "Linear_Q":
                ep_return, steps = agent.learn(env, max_steps=1000)
            elif Algorithm_name == "DQN":
                ep_return, steps = agent.learn(env, num_agents=args_cli.num_envs, max_steps=1000)
            elif Algorithm_name == "MC_REINFORCE":
                ep_return, loss, steps = agent.learn(env, num_agents=args_cli.num_envs)
            elif Algorithm_name == "AC":
                ep_return, loss, steps = agent.learn(env, max_steps=1000, num_agents=args_cli.num_envs)
            elif Algorithm_name == "PPO":
                agent.learn(env, num_envs=args_cli.num_envs, num_transitions_per_env=500, max_episodes=1)
                steps = 500  

            agent.plot_durations(timestep=steps, show_result=False)
            # ====================================== #

            # Logging & checkpointing
            if episode % 100 == 0:
                print(f"[{Algorithm_name}] episode {episode}")

            if episode % save_interval == 0 and episode > 0:
                agent.save_model(model_dir, f"{Algorithm_name}_{episode}.pth")

        # Save final model and display training curve
        agent.save_model(model_dir, f"{Algorithm_name}_final.pth")
        print("Training complete.")

        agent.plot_durations(show_result=True)
        plt.ioff()
        plt.show()

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