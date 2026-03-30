from __future__ import annotations
import os
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    """
    Linear Q-Learning with function approximation.

    Args:
        num_of_action (int): Number of discrete actions.
        action_range (list): [min, max] continuous action range.
        learning_rate (float): TD weight-update step size.
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step epsilon decay.
        final_epsilon (float): Minimum exploration rate.
        discount_factor (float): Discount factor γ.
    """

    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
    ) -> None:

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        # ===== Linear weight matrix ===== #
        # Shape: (obs_feature_dim, num_of_action)
        self.w = np.zeros((4, num_of_action))

    # ------------------------------------------------------------------ #
    # Linear Q-value estimation                                           #
    # ------------------------------------------------------------------ #

    def q(self, obs, a=None):
        """
        Return the linearly-estimated Q-value(s) for a given observation.

        Args:
            obs: State feature vector φ(s), shape (obs_dim,).
            a (int | None): Action index. If None, returns Q for all actions
                            as a 1-D array of shape (num_of_action,).

        Returns:
            float | np.ndarray: Q(s, a) scalar, or Q(s, :) array.
        """
        # ========= put your code here ========= #
        # 1. แปลง Tensor จาก Isaac Lab ให้กลายเป็น Numpy Array แบนๆ (1D)
        obs_np = obs.cpu().numpy().flatten() if isinstance(obs, torch.Tensor) else np.array(obs).flatten()
        
        # 2. ถ้าระบุ Action (a) ให้คูณกับ Weight เฉพาะคอลัมน์ของ Action นั้น
        if a is not None:
            return obs_np @ self.w[:, a]
        # 3. ถ้าไม่ระบุ ให้คูณกับ Weight ทั้งหมดเพื่อหาค่า Q-Value ของทุก Action
        else:
            return obs_np @ self.w
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        next_action: int,
        terminated: bool,
    ):
        """
        Update the weight vector using the TD error.

        Args:
            obs: Current state feature vector φ(s).
            action (int): Action index taken in state s.
            reward (float): Reward received.
            next_obs: Next state feature vector φ(s').
            next_action (int): Next action taken (for SARSA-style update).
            terminated (bool): True if the episode ended.
        """
        # ========= put your code here ========= #
        # 1. หาค่า Q-Value ปัจจุบัน
        q_current = self.q(obs, action)
        
        # 2. หาค่า Q-Target (เป้าหมาย)
        if terminated:
            q_target = reward  # ถ้าจบเกมแล้ว เป้าหมายคือ Reward ก้อนสุดท้าย
        else:
            q_target = reward + self.discount_factor * np.max(self.q(next_obs))
            
        # 3. หาค่าความคลาดเคลื่อน (TD Error)
        td_error = q_target - q_current
        
        # 4. อัปเดตน้ำหนัก w = w + lr * Error * State
        obs_np = obs.cpu().numpy().flatten() if isinstance(obs, torch.Tensor) else np.array(obs).flatten()
        self.w[:, action] += self.lr * td_error * obs_np
        # ====================================== #

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy over Q(s, :).

        Args:
            state: Current state feature vector φ(s).

        Returns:
            Tuple[Tensor, int]: Scaled continuous action tensor and action index.
        """
        # ========= put your code here ========= #
        # สุ่มตัวเลข 0-1 ถ้าต่ำกว่า Epsilon ให้ "สุ่ม Action มั่วๆ"
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.num_of_action)
        else:
            # ถ้าไม่สุ่ม ให้เลือก Action ที่ได้ค่า Q-Value สูงสุด
            q_values = self.q(state)
            action_idx = int(np.argmax(q_values))
            
        # แปลง Index (เช่น 0, 1, 2) เป็นค่าแรงผลักจริง ผ่านฟังก์ชันของคลาสแม่
        scaled_action = self.scale_action(action_idx)
        return scaled_action, action_idx
        # ====================================== #

    def learn(self, env, max_steps: int):
        """
        Train the agent for one episode.

        Args:
            env: The environment.
            max_steps (int): Maximum steps per episode.

        Returns:
            Tuple[float, int]: (episode_return, timestep)
        """
        # ========= put your code here ========= #
        # 1. รีเซ็ตสภาพแวดล้อม
        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = obs["policy"]
            
        episode_return = 0.0
        
        for step in range(max_steps):
            # ดึง State ของ environment ตัวที่ 1 ออกมา
            current_state = obs[0] if obs.dim() > 1 else obs
            
            # 2. เลือก Action
            scaled_action, action_idx = self.select_action(current_state)
            
            # ปรับ Shape ให้เข้ากับ Isaac Lab (num_envs, action_dim)
            env_action = scaled_action.unsqueeze(0) if scaled_action.dim() == 1 else scaled_action
            
            # 3. สั่งให้หุ่นยนต์ขยับ
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            
            if isinstance(next_obs, dict):
                next_obs = next_obs["policy"]
                
            next_state = next_obs[0] if next_obs.dim() > 1 else next_obs
            reward_val = reward.item()
            done_bool = terminated.item() or truncated.item()
            
            # 4. อัปเดตสมอง (Weight)
            self.update(current_state, action_idx, reward_val, next_state, None, done_bool)
            
            obs = next_obs
            episode_return += reward_val
            
            # 5. ลดค่าสุ่ม (Epsilon) ลงทีละนิด
            self.decay_epsilon()
            
            if done_bool:
                step += 1
                break
                
        return episode_return, step
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence — linear weights only                                    #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save the weight matrix self.w to disk as a .npy file.

        Args:
            path (str): Directory to save the file.
            filename (str): File name (e.g., 'linear_q_cartpole.npy').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        np.save(filepath, self.w)
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load the weight matrix self.w from a .npy file.

        Args:
            path (str): Directory containing the file.
            filename (str): File name (e.g., 'linear_q_cartpole.npy').
        """
        # ========= put your code here ========= #
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            self.w = np.load(filepath)
            print(f"[INFO] Successfully loaded model from {filepath}")
        else:
            print(f"[WARNING] Model file not found at {filepath}")
        # ====================================== #