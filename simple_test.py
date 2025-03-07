import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def train():
    # GPU优化设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 环境设置
    num_envs = 16
    env_id = "CartPole-v1"
    envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    
    # 创建模型
    model = PPO(
        "MlpPolicy", 
        envs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64 * num_envs,
        n_epochs=10,
        device="cuda",
        policy_kwargs={
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
            "activation_fn": torch.nn.ReLU
        },
        verbose=1
    )
    
    # 训练
    try:
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
        model.learn(
            total_timesteps=1000000,
            callback=CheckpointCallback(
                save_freq=10000,
                save_path="./logs/",
                name_prefix="ppo_model"
            )
        )
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        envs.close()
        print(f"Final GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

if __name__ == "__main__":
    train()