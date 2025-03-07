# Simple SAGVN (Space-Air-Ground Vehicular Networks)

一个基于深度强化学习的简化版车载网络任务卸载与资源分配系统。

A simplified Space-Air-Ground Vehicular Network system for task offloading and resource allocation based on deep reinforcement learning.

## 项目描述 (Project Description)

本项目实现了一个简化版的车载网络环境，用于研究任务卸载决策和资源分配问题。使用PPO (Proximal Policy Optimization) 算法和DeepSeek提出的GRPO (Group Relative Policy Optimization) 算法进行训练。

This project implements a simplified vehicular network environment for studying task offloading decisions and resource allocation problems. It uses both the PPO (Proximal Policy Optimization) algorithm and the GRPO (Group Relative Policy Optimization) algorithm proposed by DeepSeek, which is similar to PPO but with improved efficiency and robustness.

## 快速开始 (Quick Start)

### 环境要求 (Requirements)
```bash
Python >= 3.8
PyTorch >= 2.0.0
Stable-Baselines3 >= 2.0.0
Gymnasium >= 0.28.1
```

### 安装 (Installation)
```bash
# 创建虚拟环境 (Create virtual environment)
conda create -n sagvn python=3.8
conda activate sagvn

# 安装依赖 (Install dependencies)
pip install torch
pip install stable-baselines3
pip install gymnasium
```

### 训练模型 (Train the model)
```bash
python train.py
```

## 项目结构 (Project Structure)
```
sagvn-simple/
├── env.py          # 环境实现 (Environment implementation)
├── train.py        # 训练脚本 (Training script)
├── config.py       # 配置文件 (Configuration file)
├── GRPO.py         # GRPO算法实现 (GRPO algorithm implementation)
└── README.md       # 项目文档 (Project documentation)
```

## 核心功能 (Core Features)

- 任务卸载决策 (Task offloading decisions)
- 计算资源分配 (Computational resource allocation)
- 奖励机制：基于任务完成时间 (Reward mechanism: Based on task completion time)

## 配置参数 (Configuration Parameters)

主要参数可在 `config.py` 中修改：
Main parameters can be modified in `config.py`:
- num_vehicles: 车辆数量 (Number of vehicles)
- num_edges: 边缘服务器数量 (Number of edge servers)
- task_size_min/max: 任务大小范围 (Task size range)
- total_timesteps: 训练步数 (Training steps)

## TODO

- [ ] 添加UAV和卫星层 (Add UAV and satellite layers)
- [ ] 完善奖励函数 (Improve reward function)
- [ ] 添加更多评估指标 (Add more evaluation metrics)
- [ ] 实现可视化功能 (Implement visualization features)
- [ ] 优化GRPO算法 (Optimize GRPO algorithm)

## 参考 (References)

- Stable-Baselines3 文档 (Documentation): https://stable-baselines3.readthedocs.io/
- Gymnasium 文档 (Documentation): https://gymnasium.farama.org/
- GRPO Paper: [Group Relative Policy Optimization (DeepSeekMath paper)](https://arxiv.org/abs/2402.03300) (DeepSeek's algorithm similar to PPO)