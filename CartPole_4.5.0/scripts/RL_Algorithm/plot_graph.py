import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# 1. โหลดข้อมูลที่เซฟไว้
alg_name = "Q_Learning"
task = "Stabilize"
path = f"q_value/{task}/{alg_name}"

try:
    rewards = np.load(os.path.join(path, f"{alg_name}_rewards.npy"))
    errors = np.load(os.path.join(path, f"{alg_name}_errors.npy"))
except FileNotFoundError:
    print("ไม่พบไฟล์ข้อมูล กรุณาเช็คว่ารัน train.py เสร็จสมบูรณ์แล้ว")
    exit()

# 2. พล็อตกราฟ Cumulative Rewards (ผลตอบแทนรวม)
plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, color='blue', label='Reward per Episode')
plt.plot(moving_average(rewards, n=100), color='red', linewidth=2, label='Moving Average (100 Episodes)')
plt.title(f"{alg_name} - Learning Curve (Rewards)")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.savefig(f"{alg_name}_Reward_Plot.png")
print("บันทึกรูปกราฟ Reward เสร็จสิ้น!")

# 3. พล็อตกราฟ TD Errors (ความคลาดเคลื่อน)
# (พล็อตแค่บางส่วนเพราะข้อมูล error จะมีจำนวนเท่ากับ step ทั้งหมด ซึ่งเยอะมาก)
plt.figure(figsize=(10, 5))
plt.plot(errors[::100], alpha=0.6, color='orange') # ข้ามพล็อตทีละ 100 step เพื่อลดภาระเครื่อง
plt.title(f"{alg_name} - TD Errors over Time")
plt.xlabel("Steps (x100)")
plt.ylabel("TD Error")
plt.grid(True)
plt.savefig(f"{alg_name}_Error_Plot.png")
print("บันทึกรูปกราฟ Error เสร็จสิ้น!")