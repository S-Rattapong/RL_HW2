from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType


class Q_Learning(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Q-Learning algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.Q_LEARNING,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(self, obs, action, reward, next_obs, terminated):
            """
            Update the Q-table using the Q-learning algorithm.
            """
            # 1. แปลง State ปัจจุบันและ State ถัดไปให้เป็นแบบ Discretize (จำนวนเต็ม)
            current_state = self.discretize_state(obs)
            next_state = self.discretize_state(next_obs)
            
            # 2. ดึงค่า Q ปัจจุบันจากตาราง
            current_q = self.q_values[current_state][action]
            
            # 3. หาค่า Q ที่ดีที่สุดของ State ถัดไป (จุดเด่นของ Q-Learning คือการใช้ Max)
            if terminated:
                # ถ้าไม้ล้มหรือรถออกนอกขอบ (จบเกม) State ถัดไปจะไม่มีค่า
                max_next_q = 0.0
            else:
                # ถ้ายังไม่จบเกม ให้ดูว่าในอนาคตมี Action ไหนให้ค่า Q สูงสุด
                max_next_q = np.max(self.q_values[next_state])
                
            # 4. คำนวณเป้าหมาย (TD Target) และความคลาดเคลื่อน (TD Error)
            td_target = reward + (self.discount_factor * max_next_q)
            td_error = td_target - current_q
            
            # 5. อัปเดตค่าลงในตาราง Q-Table ตามสมการ Q-Learning
            self.q_values[current_state][action] = current_q + (self.lr * td_error)
            
            # เก็บค่า Error ไว้สำหรับวิเคราะห์ผล (Part 3)
            self.training_error.append(float(td_error))
