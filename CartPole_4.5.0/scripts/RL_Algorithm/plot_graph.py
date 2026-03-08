import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# รับค่าจาก Terminal
parser = argparse.ArgumentParser()
parser.add_argument("--alg", type=str, default="all", help="ใส่ชื่อ Algorithm (เช่น Q_Learning) หรือ all เพื่อพล็อตรวม")
args = parser.parse_args()

task = "Stabilize"

if args.alg == "all":
    # ================= 1. พล็อตกราฟรวม 4 Algorithm =================
    algorithms_all = ["Q_Learning", "SARSA", "Double_Q_Learning", "MC"]
    colors = ['blue', 'green', 'red', 'purple']
    plt.figure(figsize=(12, 6))

    for alg, color in zip(algorithms_all, colors):
        file_path = os.path.join(f"q_value/{task}", alg, f"{alg}_rewards.npy")
        try:
            rewards = np.load(file_path)
            plt.plot(moving_average(rewards, n=100), label=f"{alg} (MA 100)", color=color, linewidth=2, alpha=0.85)
        except FileNotFoundError:
            print(f"[Warning] ไม่พบไฟล์ข้อมูลของ {alg}")
    
    plt.title("Performance Comparison of All RL Algorithms (CartPole)")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward (Moving Average)")
    plt.legend()
    plt.grid(True)
    save_path = "Compare_All_Algorithms.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ บันทึกรูปกราฟเปรียบเทียบเสร็จสิ้น! (ไฟล์: {save_path})")

else:
    # ================= 2. พล็อตกราฟทีละ Algorithm =================
    alg = args.alg
    path = os.path.join(f"q_value/{task}", alg)
    
    # 2.1 พล็อต Reward
    try:
        rewards = np.load(os.path.join(path, f"{alg}_rewards.npy"))
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.3, color='blue', label='Reward per Episode')
        plt.plot(moving_average(rewards, n=100), color='red', linewidth=2, label='Moving Average (100)')
        plt.title(f"Learning Curve: {alg}")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{alg}_Reward_Plot.png", dpi=300)
        print(f"✅ บันทึกกราฟ Reward ของ {alg} สำเร็จ!")
    except FileNotFoundError:
        print(f"[Error] ไม่พบไฟล์ Reward ของ {alg}")

    # 2.2 พล็อต Error
    try:
        errors = np.load(os.path.join(path, f"{alg}_errors.npy"))
        plt.figure(figsize=(10, 5))
        plt.plot(errors[::100], alpha=0.6, color='orange') # ข้ามพล็อตทีละ 100 step
        plt.title(f"TD Errors over Time: {alg}")
        plt.xlabel("Steps (x100)")
        plt.ylabel("TD Error")
        plt.grid(True)
        plt.savefig(f"{alg}_Error_Plot.png", dpi=300)
        print(f"✅ บันทึกกราฟ Error ของ {alg} สำเร็จ!")
    except FileNotFoundError:
        pass