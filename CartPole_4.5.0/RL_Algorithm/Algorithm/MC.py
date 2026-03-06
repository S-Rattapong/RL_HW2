from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
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
        Initialize the Monte Carlo algorithm.

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
            control_type=ControlType.MONTE_CARLO,
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
        Update Q-values using Monte Carlo.

        This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        """
        current_state = self.discretize_state(obs)
        
        # 1. เก็บประวัติ (History) ของ Step ปัจจุบัน
        self.obs_hist.append(current_state)
        self.action_hist.append(action)
        self.reward_hist.append(reward)
        
        # 2. จะทำการอัปเดตตารางก็ต่อเมื่อ Episode จบลงแล้วเท่านั้น
        if terminated:
            G = 0 # ตัวแปรเก็บ Cumulative Reward (ผลตอบแทนรวมสะสม)
            
            # ย้อนเวลาจากท้ายสุดของ Episode กลับมาจุดเริ่มต้น
            for i in reversed(range(len(self.reward_hist))):
                state = self.obs_hist[i]
                act = self.action_hist[i]
                r = self.reward_hist[i]
                
                # คำนวณ G = Reward + (Discount * G_ถัดไป)
                G = r + self.discount_factor * G
                
                # นับว่าเคยมา state-action นี้ (เพื่อใช้ในอนาคตถ้าอยากหาแบบ 1/N)
                self.n_values[state][act] += 1
                
                # อัปเดตตาราง Q ด้วยสมการ Q = Q + lr * (G - Q)
                error = G - self.q_values[state][act]
                self.q_values[state][act] += self.lr * error
                self.training_error.append(float(error))
                
            # เคลียร์ประวัติความจำ เพื่อเริ่มรอบฝึก (Episode) ใหม่
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []