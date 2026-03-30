"""Script to play RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
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
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import numpy as np
import matplotlib.pyplot as plt

def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,  #use_fabric=not args_cli.disable_fabric
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # 1. ตั้งค่าพารามิเตอร์ให้ "ตรงเป๊ะ" กับตอน Train
    num_of_action = 5
    action_range = [-2.0, 2.0]
    discretize_state_weight = [10, 100, 10, 100]

    task_name = str(args_cli.task).split('-')[0]  # จะได้คำว่า Stabilize
    Algorithm_name = "Q_Learning"

    # 2. สร้าง Agent (จุดสำคัญ: ตอนเล่นจริงเราต้องตั้ง epsilon=0 เพื่อไม่ให้มันสุ่มมั่วซั่ว และใช้ความรู้ 100%)
    agent = Q_Learning(
        num_of_action=num_of_action,
        action_range=action_range,
        discretize_state_weight=discretize_state_weight,
        learning_rate=0.1,
        initial_epsilon=0.0,   # <--- ตั้งเป็น 0 เพื่อบังคับ Exploitation
        epsilon_decay=0.0,
        final_epsilon=0.0,
        discount_factor=0.99
    )

    # 3. กำหนดชื่อไฟล์สมอง (Q-Table) ที่จะโหลด
    # ปกติลูป 5000 รอบ เซฟทุก 100 รอบ ไฟล์สุดท้ายจะอยู่ที่รอบ 4900
    # **ข้อควรระวัง:** ให้ลองเช็คในโฟลเดอร์ q_value/Stabilize/Q_Learning อีกครั้งเพื่อความชัวร์ว่ามีไฟล์ชื่อนี้อยู่
    target_episode = 4900 
    q_value_file = f"{Algorithm_name}_{target_episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
    full_path = os.path.join(f"q_value/{task_name}", Algorithm_name)

    # โหลดไฟล์ Q-table
    print(f"[INFO] กำลังโหลดสมองจาก: {full_path}/{q_value_file}")
    try:
        agent.load_q_value(full_path, q_value_file)
        print("[INFO] โหลด Q-Table สำเร็จ! พร้อมลุย!")
    except FileNotFoundError:
        print("[ERROR] ไม่พบไฟล์ Q-Table รบกวนตรวจสอบชื่อไฟล์ในโฟลเดอร์ q_value ครับ")

    # 4. ลูปรันแสดงผล
    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # ให้ Agent เลือก Action จาก State ปัจจุบัน
            action, _ = agent.get_action(obs)
            
            # ส่ง Action ไปให้ Environment รัน
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # ถ้าไม้ล้ม หรือรถวิ่งชนขอบ (จบเกม) ให้รีเซ็ตเริ่มรอบใหม่ทันที
            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs
                
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
                
    print("!!! ปิดระบบ !!!")
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()