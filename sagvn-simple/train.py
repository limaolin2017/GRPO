from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from env import SimpleVehicularEnv
from config import CONFIG
import torch
import wandb
import numpy as np

class WandbCallback(BaseCallback):
    """用于WandB记录的回调函数"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self):

        # 记录训练指标
        metrics = {
            "timesteps": self.num_timesteps,
            "episode_reward": self.model.ep_info_buffer[-1]["r"] if self.model.ep_info_buffer else 0,
            "policy_loss": self.model.logger.name_to_value.get("train/policy_loss", 0),
            "value_loss": self.model.logger.name_to_value.get("train/value_loss", 0)
        }
            
        # GPU指标
        if torch.cuda.is_available():
                metrics.update({
                    "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2,
                    "gpu_utilization": torch.cuda.utilization()
            })
            
        wandb.log(metrics)

        return True

def train():
    # 初始化wandb
    wandb.init(
        project="sagvn",
        name="ppo_experiment",
        config={**CONFIG, "algorithm": "PPO"}
    )
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device.upper()}")
    
    # 创建环境和模型
    env = SimpleVehicularEnv(CONFIG)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=CONFIG['learning_rate'],
        batch_size=CONFIG['batch_size'] * 4,
        n_steps=2048,
        n_epochs=10,
        device=device,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
            activation_fn=torch.nn.ReLU
        )
    )
    
    try:
        # 训练模型
        model.learn(
            total_timesteps=CONFIG['total_timesteps'],
            callback=[
                CheckpointCallback(
                    save_freq=10000,
                    save_path="./logs/",
                    name_prefix="vehicular_model"
                ),
                WandbCallback()
            ],
            progress_bar=True
        )
        
        # 保存模型
        model.save("vehicular_final")
        wandb.save("vehicular_final.zip")
        
    except KeyboardInterrupt:
        print("Training interrupted, saving model...")
        model.save("vehicular_interrupted")
    
    except Exception as e:
        print(f"Training error: {e}")
        raise e
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    train()