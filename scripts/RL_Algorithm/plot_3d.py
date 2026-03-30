import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast

def plot_3d_q_values(json_file_path):
    print(f"[INFO] กำลังโหลด Q-Table จาก: {json_file_path}")
    
    # 1. โหลดข้อมูลจากไฟล์ JSON
    with open(json_file_path, 'r') as f:
        q_table = json.load(f)

    parsed_data = {}
    cart_positions = set()
    pole_angles = set()

    # 2. คัดแยกเฉพาะ State ที่ความเร็วเป็น 0
    for state_str, q_values in q_table.items():
        try:
            # แปลง string ยึกยือใน JSON ให้กลายเป็น Tuple
            # โค้ดเก่าอาจจะเซฟเป็น "(x, y, z, w)" หรือ "[x, y, z, w]"
            state = ast.literal_eval(state_str)
        except:
            state = tuple(map(float, state_str.strip('()[]').split(',')))
        
        cart_pos, pole_angle, cart_vel, pole_vel = state
        
        # *** จุดสำคัญ: กรองเอาเฉพาะที่ความเร็ว = 0 ***
        if cart_vel == 0 and pole_vel == 0:
            max_q = np.max(q_values)  # ดึงค่า Q-Value ที่ดีที่สุดใน State นั้น
            parsed_data[(cart_pos, pole_angle)] = max_q
            cart_positions.add(cart_pos)
            pole_angles.add(pole_angle)

    if not parsed_data:
        print("[ERROR] ไม่พบข้อมูล State ที่ความเร็วเป็น 0 เลย!")
        return

    # 3. เตรียมแกน X (ตำแหน่งรถ) และ Y (มุมเสา)
    cart_positions = sorted(list(cart_positions))
    pole_angles = sorted(list(pole_angles))
    
    # คุณสามารถหารค่า Weight กลับตรงนี้ได้ เพื่อให้แกนโชว์ตัวเลขจริง (เช่น /1.5 หรือ /10)
    # แต่พล็อตค่า Index ดิบๆ ไปก่อนก็เห็นรูปทรงภูเขาชัดเจนครับ
    X, Y = np.meshgrid(cart_positions, pole_angles)
    Z = np.zeros_like(X, dtype=float)

    # 4. เติมความสูง (Z) หรือ Max Q-Value ลงในแผนที่
    for i in range(len(pole_angles)):
        for j in range(len(cart_positions)):
            state_key = (cart_positions[j], pole_angles[i])
            # ถ้าช่องไหนหุ่นยนต์ยังไม่เคยเดินไปเหยียบ ให้ตั้งค่าความสูงเป็น 0
            Z[i, j] = parsed_data.get(state_key, 0.0)

    # 5. พล็อตกราฟ 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # สร้าง Surface สีสวยๆ แบบรุ้ง (viridis)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_title("3D Surface Plot of Q-Values (Velocity = 0)")
    ax.set_xlabel("Cart Position (Discretized)")
    ax.set_ylabel("Pole Angle (Discretized)")
    ax.set_zlabel("Max Q-Value")

    # ใส่แถบสีบอกระดับคะแนนด้านข้าง
    fig.colorbar(surf, shrink=0.5, aspect=5, label="Max Q-Value")

    plt.show()

if __name__ == "__main__":
    # ใส่พาธไฟล์ JSON ที่คุณเซฟไว้ล่าสุด (แก้ชื่อไฟล์ให้ตรงกับของคุณ)
    # ตัวอย่าง: "q_value/Stabilize/Q_Learning/Q_Learning_4900_11_2.0_1.5_10.0.json"
    
    FILE_PATH = "q_value/Stabilize/Q_Learning/Q_Learning_5000_11_2.0_1.5_10.json" 