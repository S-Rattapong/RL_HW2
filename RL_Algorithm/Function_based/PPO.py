from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from RL_Algorithm.storage.on_policy import OnPolicyAlgorithm
from RL_Algorithm.storage.buffers import RolloutBuffer
from RL_Algorithm.Function_based.AC import ActorCritic


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization (PPO) — on-policy, clipped surrogate.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
        num_learning_epochs (int): Epochs per PPO update.
        num_mini_batches (int): Mini-batches per epoch.
        clip_param (float): PPO clipping ε.
        gamma (float): Discount factor γ.
        lam (float): GAE lambda λ.
        value_loss_coef (float): Coefficient for value loss.
        entropy_coef (float): Coefficient for entropy bonus.
        learning_rate (float): Adam learning rate.
        max_grad_norm (float): Gradient clipping norm.
        desired_kl (float): KL target for adaptive LR (0 to disable; use 0 for discrete).
        normalize_advantage_per_mini_batch (bool): Normalise advantages per mini-batch.
        use_clipped_value_loss (bool): Apply clipped value loss.
    """

    def __init__(
        self,
        device=None,
        num_of_action: int = None,
        action_range: list = [None, None],
        n_observations: int = None,
        hidden_dims: list[int] = [None],
        activation: str = None,
        action_type: str = None,
        init_noise_std: float = None,
        num_learning_epochs: int = None,
        num_mini_batches: int = None,
        clip_param: float = None,
        gamma: float = None,
        lam: float = None,
        value_loss_coef: float = None,
        entropy_coef: float = None,
        learning_rate: float = None,
        max_grad_norm: float = None,
        desired_kl: float = None,
        normalize_advantage_per_mini_batch: bool = False,
        use_clipped_value_loss: bool = True,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ===== Build ActorCritic network (imported from AC.py) ===== #
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.policy = ActorCritic(
            state_dim=n_observations,
            action_dim=num_of_action,
            hidden_dims=hidden_dims,
            activation=activation,
            action_type=action_type,
            init_noise_std=init_noise_std,
        ).to(self.device)
        # ====================================== #

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # ===== PPO hyperparameters ===== #
        self.action_type                        = action_type
        self.clip_param                         = clip_param
        self.num_learning_epochs                = num_learning_epochs
        self.num_mini_batches                   = num_mini_batches
        self.value_loss_coef                    = value_loss_coef
        self.entropy_coef                       = entropy_coef
        self.gamma                              = gamma
        self.lam                                = lam
        self.max_grad_norm                      = max_grad_norm
        self.desired_kl                         = desired_kl
        self.learning_rate                      = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.use_clipped_value_loss             = use_clipped_value_loss

        super(PPO, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
        )

    # ------------------------------------------------------------------ #
    # Rollout collection                                                   #
    # ------------------------------------------------------------------ #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample actions for all parallel envs and populate self.transition.

        Continuous: actions shape (num_envs, action_dim).
        Discrete  : actions shape (num_envs, 1).

        Args:
            obs (Tensor): shape (num_envs, obs_dim).

        Returns:
            Tensor: Sampled actions.
        """
        # ========= put your code here ========= #
        with torch.no_grad():
            self.policy._update_distribution(obs)
            actions = self.policy.distribution.sample()
            values = self.policy.evaluate(obs)
            log_probs = self.policy.get_actions_log_prob(actions)

        # บันทึกข้อมูลที่หุ่นยนต์คิดได้ลงในกระดาษทด (Transition)
        self.transition.observations = obs.clone()
        self.transition.actions = actions.clone()
        self.transition.values = values.clone()
        self.transition.actions_log_prob = log_probs.clone()
        self.transition.action_mean = self.policy.action_mean.clone()
        self.transition.action_sigma = self.policy.action_std.clone()
        # ====================================== #

        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """
        Write rewards and dones into self.transition, then flush to storage.

        Args:
            rewards (Tensor): shape (num_envs,) or (num_envs, 1).
            dones (Tensor): shape (num_envs,) or (num_envs, 1).
        """
        # ========= put your code here ========= #
        # จัด Shape ให้เป็น Column Vector (num_envs, 1) เสมอ
        self.transition.rewards = rewards.clone().view(-1, 1).float()
        self.transition.dones = dones.clone().view(-1, 1).byte()
        # ====================================== #

        # Flush transition into RolloutBuffer via inherited add_transition()
        self.add_transition()

    # ------------------------------------------------------------------ #
    # Return & Advantage Computation                                       #
    # ------------------------------------------------------------------ #

    def compute_returns(self, last_obs: torch.Tensor) -> None:
        """
        Compute GAE returns and advantages over the collected rollout.

        Args:
            last_obs (Tensor): Observation after the final rollout step.
                               Shape: (num_envs, obs_dim).
        """
        # ========= put your code here ========= #
        # คำนวณ GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            next_value = self.policy.evaluate(last_obs)
            
        adv = 0
        for step in reversed(range(self.storage.num_transitions_per_env)):
            if step == self.storage.num_transitions_per_env - 1:
                next_non_terminal = 1.0 - self.storage.dones[step]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.storage.dones[step + 1]
                next_values = self.storage.values[step + 1]

            delta = self.storage.rewards[step] + self.gamma * next_values * next_non_terminal - self.storage.values[step]
            adv = delta + self.gamma * self.lam * next_non_terminal * adv
            self.storage.advantages[step] = adv
            self.storage.returns[step] = self.storage.advantages[step] + self.storage.values[step]
            
        if self.normalize_advantage_per_mini_batch:
            self.storage.advantages = (self.storage.advantages - self.storage.advantages.mean()) / (self.storage.advantages.std() + 1e-8)
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Policy Update                                                        #
    # ------------------------------------------------------------------ #

    def update(self) -> dict:
        """
        Perform PPO updates over the collected rollout.

        Calls ``self.storage.mini_batch_generator()`` which now lives in
        ``RolloutBuffer`` (storage/buffers.py) and yields 8-tuples.

        Returns:
            dict: Mean losses {'value', 'surrogate', 'entropy'}.
        """
        mean_value_loss     = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy        = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            # ========= put your code here ========= #
            # ดึง Distribution เพื่อหา log_prob ใหม่
            self.policy._update_distribution(obs_batch)
            log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            entropy_batch = self.policy.entropy

            # PPO Clipped Surrogate Loss
            ratio = torch.exp(log_prob_batch - old_actions_log_prob_batch)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value Loss
            if self.use_clipped_value_loss:
                value_pred_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                critic_loss = 0.5 * (returns_batch - value_batch).pow(2).mean()

            entropy_loss = entropy_batch.mean()
            loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += critic_loss.item()
            mean_surrogate_loss += actor_loss.item()
            mean_entropy += entropy_loss.item()
            # ====================================== #

        num_updates          = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy        /= num_updates

        self.storage.clear()   # on-policy: discard rollout after update

        return {
            "value":     mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy":   mean_entropy,
        }

    # ------------------------------------------------------------------ #
    # Main Training Loop                                                   #
    # ------------------------------------------------------------------ #

    def learn(
        self,
        env,
        num_envs: int,
        num_transitions_per_env: int,
        max_episodes: int = 10000,
    ) -> None:
        """
        Main PPO parallel training loop.

        Calls ``_init_storage()`` (from OnPolicyAlgorithm) to create the buffer.

        Continuous: actions_shape = (num_of_action,)
        Discrete  : actions_shape = (1,)

        Args:
            env: Isaac Lab vectorised environment.
            num_envs (int): Number of parallel environments.
            num_transitions_per_env (int): Rollout horizon per env.
            max_episodes (int): Total number of training rollouts.
        """
        # ========= put your code here ========= #
        # 1. ดึง Shape จาก observation_space["policy"] (ใช้ [] เท่านั้น)
        if hasattr(env.observation_space, 'keys'):
            obs_space = env.observation_space["policy"]
        else:
            obs_space = env.observation_space
            
        obs_shape = obs_space.shape
        actions_shape = (self.num_of_action,) if self.action_type == "continuous" else (1,)
        
        # สร้าง Buffer สำหรับเก็บข้อมูล
        self._init_storage(num_envs, num_transitions_per_env, obs_shape, actions_shape, self.device)

        for episode in range(max_episodes):
            # ต้อง reset และเก็บข้อมูลภายใต้ inference_mode เพื่อป้องกัน RuntimeError
            with torch.inference_mode():
                obs, _ = env.reset()
                if isinstance(obs, dict):
                    obs = obs["policy"]
                    
                # ลูปเก็บประสบการณ์ (Rollout)
                for _ in range(num_transitions_per_env):
                    actions = self.act(obs)
                    
                    if self.action_type == "discrete":
                        # สำหรับ Discrete: แปลง action index เป็นแรงผลักจริง
                        env_actions = torch.stack([self.scale_action(a.item()) for a in actions]).to(self.device)
                        if env_actions.dim() == 1: env_actions = env_actions.unsqueeze(1)
                    else:
                        # สำหรับ Continuous: จำกัดช่วงแรงผลัก
                        env_actions = torch.clamp(actions, self.action_range[0], self.action_range[1])

                    next_obs, rewards, terminated, truncated, _ = env.step(env_actions)
                    if isinstance(next_obs, dict):
                        next_obs = next_obs["policy"]
                        
                    dones = terminated | truncated
                    self.process_env_step(rewards, dones)
                    obs = next_obs

                # คำนวณ GAE (ความได้เปรียบ)
                self.compute_returns(obs)

            # อัปเดตสมอง (อยู่นอก Inference Mode เพราะต้องใช้ Gradient)
            self.update()
            
            if episode % 10 == 0:
                print(f"Episode {episode} | Completed PPO Update")
        # ====================================== #


    # ------------------------------------------------------------------ #
    # Inference & Persistence                                              #
    # ------------------------------------------------------------------ #

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for evaluation.

        Continuous: actor mean. Discrete: argmax of logits.

        Args:
            obs (Tensor): shape (1, obs_dim) or (obs_dim,).
        """
        # ========= put your code here ========= #
        with torch.no_grad():
            action = self.policy.act_inference(obs.unsqueeze(0) if obs.dim() == 1 else obs)
            
        if self.action_type == "discrete":
            action_idx = action.item()
            return self.scale_action(action_idx), action_idx
        else:
            return torch.clamp(action, self.action_range[0], self.action_range[1]), action
        # ====================================== #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor-critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'ppo_cartpole.pth').
        """
        # ========= put your code here ========= #
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor-critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'ppo_cartpole.pth').
        """
        # ========= put your code here ========= #
        import os
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
            print(f"[INFO] Successfully loaded PPO model from {filepath}")
        else:
            print(f"[WARNING] Model file not found at {filepath}")
        # ====================================== #