import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import numpy as np
from typing import Optional, Type, Union, Callable, Dict, Any

# Custom environment class
class DummyEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = gym.spaces.Discrete(2)
        
    def reset(self):
        return np.zeros(4)
        
    def step(self, action):
        obs = np.random.uniform(-1, 1, 4)
        reward = float(action)
        done = False
        return obs, reward, done, {}

class GRPO(BaseAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env,
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Callable] = 0.2,
        kl_coeff: float = 0.01,
        verbose: int = 0,
        device: Union[torch.device, str] = "auto",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        group_size: int = 5,  # 新增组内采样次数参数
        _init_setup_model: bool = True,
    ):
        self.verbose = verbose
        self.use_sde = False  # 兼容 SB3 保存/加载机制
        self.group_size = group_size
        # Assign the policy to self.policy_class
        self.policy_class = policy
        
        # Assign environment and related attributes
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.policy_kwargs = policy_kwargs or {}
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = get_schedule_fn(clip_range)
        self.kl_coeff = kl_coeff

        self.lr_schedule = get_schedule_fn(learning_rate)
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr_schedule(1))
        self.rollout_buffer = RolloutBuffer(
            self.n_steps, self.observation_space, self.action_space, self.device, gamma=self.gamma, gae_lambda=self.gae_lambda
        )

    def train(self) -> None:
        self.policy.train()

        for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):
            actions = rollout_data.actions
            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

            # 这里使用组内相对优势（已在 collect_rollouts 中计算并存储在 rollout_buffer.advantages）
            advantages = rollout_data.advantages
            # 可选：归一化组内优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO 风格的策略损失
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range(1), 1 + self.clip_range(1))
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # KL 散度正则项
            kl_div = torch.mean(log_prob - rollout_data.old_log_prob)
            kl_loss = self.kl_coeff * kl_div

            # 注意：这里可能存在 shape 不匹配问题（例如 values 的 shape 可能为 [batch_size]），可考虑 unsqueeze
            value_loss = nn.functional.mse_loss(rollout_data.returns, values.unsqueeze(-1))
            loss = policy_loss + 0.5 * value_loss + kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def collect_rollouts(self, env, rollout_buffer, n_rollout_steps: int):
        rollout_buffer.reset()  # 重置缓冲区
        obs = env.reset()[0]  # 取出单个环境的初始状态
        group_advantages = []  # 用于存储每一步计算的组内相对优势

        for step in range(n_rollout_steps):
            # 对当前状态做 batch 化处理
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(self.device)
            group_outputs = []
            # 组内采样：对同一状态采样 group_size 次
            for _ in range(self.group_size):
                with torch.no_grad():
                    action_i, value_i, log_prob_i = self.policy(obs_tensor)
                # 提取 batch 中的第一个元素，保持 value 和 log_prob 为 tensor
                action_i = action_i[0].cpu().numpy() if torch.is_tensor(action_i) else action_i[0]
                value_i = value_i[0]
                log_prob_i = log_prob_i[0]
                group_outputs.append((action_i, value_i, log_prob_i))
            # 选择一个采样用于实际与环境交互，例如选取第一组结果
            chosen_action, chosen_value, chosen_log_prob = group_outputs[0]
            # 与环境交互，注意动作需要保持 batch 维度
            next_obs, reward, done, info = env.step([chosen_action])
            reward = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            done = done[0] if isinstance(done, (list, np.ndarray)) else done

            # 使用下一状态计算 next_value（这里为统一的评估，适用于所有采样）
            next_obs_tensor = torch.as_tensor(next_obs[0]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value, _ = self.policy(next_obs_tensor)
            next_value = next_value[0]  # 保持为 tensor

            # 计算组内所有样本的 TD 误差 delta_i = r + γ·V(s') - V_i(s)
            deltas = []
            for (_, value_i, _) in group_outputs:
                delta_i = reward + (0 if done else self.gamma * next_value) - value_i
                deltas.append(delta_i)
            # 计算组内 TD 误差均值
            group_delta = sum(deltas) / len(deltas)
            # 对选中的样本，计算其 TD 误差（即 r + γ·V(s') - chosen_value ）
            chosen_delta = reward + (0 if done else self.gamma * next_value) - chosen_value
            # 组内相对优势 A^{grp} 定义为选中样本与组内均值之差
            A_grp = chosen_delta - group_delta
            group_advantages.append(A_grp)

            # 将选中的样本数据存入 RolloutBuffer
            rollout_buffer.add(obs, chosen_action, reward, done, chosen_value, chosen_log_prob)
            obs = next_obs[0]  # 更新状态

        # 将组内优势覆盖原有的 advantage（注意：此处转换为 tensor）
        rollout_buffer.advantages = torch.tensor(group_advantages, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 可选：如果需要计算 returns，可调用 compute_returns_and_advantage()（此处我们主要替换优势）
        # rollout_buffer.compute_returns_and_advantage(last_values=chosen_value, dones=done)

        # 构造 dummy 对象返回采样步数
        class DummyRollout:
            pass
        dummy = DummyRollout()
        dummy.episode_timesteps = n_rollout_steps
        return dummy

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 10):
        timesteps_done = 0

        while timesteps_done < total_timesteps:
            rollout = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps)
            timesteps_done += rollout.episode_timesteps

            self.train()

            if timesteps_done % log_interval == 0:
                explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
                print(f"Timestep: {timesteps_done}, Explained variance: {explained_var}")

        return self

# Test example
if __name__ == "__main__":
    # 创建一个简单的向量化环境
    env = DummyVecEnv([lambda: DummyEnv(render_mode="human")])
    
    # 初始化 GRPO 模型，设置 group_size 参数
    model = GRPO(
        ActorCriticPolicy, 
        env, 
        learning_rate=3e-4, 
        n_steps=512, 
        kl_coeff=0.01,
        group_size=5  # 例如，每个状态采样 5 次
    )

    print("Starting training...")
    model.learn(total_timesteps=5000)
    print("Training completed. Saving model...")
    model.save("grpo_model")
    print("Model saved. Loading and testing model...")

    loaded_model = GRPO.load("grpo_model", env=env)
    obs = env.reset()
    for step in range(10):
        print(f"Step {step + 1}")
        action, _states = loaded_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(f"Action: {action}, Reward: {rewards}, Done: {dones}")
        env.render()

    print("Testing completed.")
