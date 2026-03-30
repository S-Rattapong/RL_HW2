from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from storage.off_policy import OffPolicyAlgorithm


class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.

    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input state tensor.

        Returns:
            Tensor: Q-value estimates for each action.
        """
        # ========= put your code here ========= #
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        # ====================================== #


class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN) — off-policy, value-based.

    Args:
        device: Torch device.
        num_of_action (int): Number of discrete actions.
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        dropout (float): Dropout rate.
        learning_rate (float): Adam learning rate.
        tau (float): Polyak soft-update coefficient for target network.
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step epsilon decay.
        final_epsilon (float): Minimum exploration rate.
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            dropout: float = None,
            learning_rate: float = None,
            tau: float = None,
            initial_epsilon: float = None,
            epsilon_decay: float = None,
            final_epsilon: float = None,
            discount_factor: float = None,
            buffer_size: int = None,
            batch_size: int = None,
    ) -> None:

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.device        = device
        self.steps_done    = 0
        self.num_of_action = num_of_action
        self.tau           = tau

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        pass
        # ====================================== #

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state (Tensor): Current state.

        Returns:
            Tuple[Tensor, int]: Scaled action tensor and action index.
        """
        # ========= put your code here ========= #
        import random
        # สุ่มตัวเลข 0-1 ถ้าต่ำกว่า Epsilon ให้สุ่ม Action
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.num_of_action)
        else:
            # ถ้าไม่สุ่ม ให้ใช้ Policy Network ประเมินค่า Q ที่ดีที่สุด
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_idx = q_values.max(1)[1].item()
                
        # แปลง Index เป็นแรงผลักจริง
        scaled_action = self.scale_action(action_idx)
        return scaled_action, action_idx
        # ====================================== #

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        """
        Compute the Bellman loss for a sampled mini-batch.

        Args:
            non_final_mask (Tensor): True where next state is not terminal.
            non_final_next_states (Tensor): Non-terminal next states.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of action indices.
            reward_batch (Tensor): Batch of rewards.

        Returns:
            Tensor: Scalar Huber / MSE loss.
        """
        # ========= put your code here ========= #
        # 1. หา Q-Value ปัจจุบันของ Action ที่เราเคยเลือก
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 2. หา Q-Value ของ Next State จาก Target Network
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            
        # 3. คำนวณ Q-Target 
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.discount_factor) + reward_batch

        # 4. หาค่า Loss โดยใช้ Huber Loss (Smooth L1) ซึ่งทนทานต่อ Outlier กว่า MSE
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        return loss
        # ====================================== #

    def generate_sample(self, batch_size=None):
        """
        Sample a mini-batch and unpack it into DQN-ready tensors.

        Returns:
            Tuple or None:
                - non_final_mask (Tensor)
                - non_final_next_states (Tensor)
                - state_batch (Tensor)
                - action_batch (Tensor)
                - reward_batch (Tensor)
            Returns None if the buffer is not ready.
        """
        # ========= put your code here ========= #
        batch = super().generate_sample()
        if batch is None:
            return None
        # ====================================== #

        # Unpack and prepare tensors from the Transition namedtuples
        # ========= put your code here ========= #
        # แยกร่าง (Unzip) ข้อมูลที่สุ่มมาได้ 
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # จัดการกับ Next State (เลือกเฉพาะ State ที่ยังไม่จบเกม)
        non_final_mask = torch.tensor([not d for d in dones], device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s, d in zip(next_states, dones) if not d]).to(self.device)
        
        # แปลงเป็น Tensor เพื่อเข้าสมการ
        state_batch = torch.cat(states).to(self.device)
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(1)
        
        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch
        # ====================================== #

    def update_policy(self):
        """Perform one gradient step on the policy network."""
        sample = self.generate_sample()
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        # ========= put your code here ========= #
        # เคลียร์ Gradient เก่า -> คำนวณ Gradient ใหม่ -> อัปเดตน้ำหนัก
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping (ป้องกัน Gradient ระเบิด)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        # ====================================== #

    def update_target_networks(self):
        # ========= put your code here ========= #
        # Polyak Soft Update (ค่อยๆ ซึมซับความรู้ไปทีละนิด)
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            
        self.target_net.load_state_dict(target_net_state_dict)
        # ====================================== #

    def learn(self, env, num_agents: int = 1, max_steps: int = 1000):
        """
        Train the agent for one episode (single env) or one fixed-length
        run (parallel envs).

        Args:
            env: The Isaac Lab environment.
            num_agents (int): Number of parallel environments.
            max_steps (int): Steps per episode (single) or total env steps (parallel).

        Returns:
            Tuple[float, int]: (episode_return, timestep)
        """

        # ========= put your code here ========= #
        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = obs["policy"]
            
        episode_return = 0.0
        
        for step in range(max_steps):
            # จัด Shape ให้เป็น (1, state_dim)
            current_state = obs[0] if obs.dim() > 1 else obs
            current_state = current_state.unsqueeze(0) if current_state.dim() == 1 else current_state

            scaled_action, action_idx = self.select_action(current_state)
            env_action = scaled_action.unsqueeze(0) if scaled_action.dim() == 1 else scaled_action

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            
            if isinstance(next_obs, dict):
                next_obs = next_obs["policy"]
                
            next_state = next_obs[0] if next_obs.dim() > 1 else next_obs
            next_state = next_state.unsqueeze(0) if next_state.dim() == 1 else next_state
            
            reward_val = reward.item()
            done_bool = terminated.item() or truncated.item()

            # โยนประสบการณ์เข้า Replay Buffer
            self.store_transition(current_state, action_idx, reward_val, next_state, done_bool)
            
            # อัปเดตสมอง 2 เครือข่าย
            self.update_policy()
            self.update_target_networks()

            obs = next_obs
            episode_return += reward_val
            
            self.decay_epsilon()
            
            if done_bool:
                step += 1
                break
                
        return episode_return, step
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save policy network weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'dqn_cartpole.pth').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load policy network weights and sync to target network.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'dqn_cartpole.pth').
        """
        # ========= put your code here ========= #
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
            # Sync ไปที่ Target ด้วยเลย
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"[INFO] Successfully loaded DQN model from {filepath}")
        else:
            print(f"[WARNING] Model file not found at {filepath}")
        # ====================================== #