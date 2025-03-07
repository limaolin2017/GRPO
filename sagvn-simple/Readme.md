# Simple SAGVN (Space-Air-Ground Vehicular Networks)

一个基于深度强化学习的简化版车载网络任务卸载与资源分配系统。

## 项目描述

本项目实现了一个简化版的车载网络环境，用于研究任务卸载决策和资源分配问题。使用PPO (Proximal Policy Optimization) 算法进行训练。

## 快速开始

### 环境要求
```bash
Python >= 3.8
PyTorch >= 2.0.0
Stable-Baselines3 >= 2.0.0
Gymnasium >= 0.28.1
```

### 安装
```bash
# 创建虚拟环境
conda create -n sagvn python=3.8
conda activate sagvn

# 安装依赖
pip install torch
pip install stable-baselines3
pip install gymnasium
```

### 训练模型
```bash
python train.py
```

## 项目结构
```
sagvn-simple/
├── env.py          # 环境实现
├── train.py        # 训练脚本
├── config.py       # 配置文件
└── README.md       # 项目文档
```

## 核心功能

- 任务卸载决策
- 计算资源分配
- 奖励机制：基于任务完成时间

## 配置参数

主要参数可在 `config.py` 中修改：
- num_vehicles: 车辆数量
- num_edges: 边缘服务器数量
- task_size_min/max: 任务大小范围
- total_timesteps: 训练步数

## TODO

- [ ] 添加UAV和卫星层
- [ ] 完善奖励函数
- [ ] 添加更多评估指标
- [ ] 实现可视化功能

## 参考

- Stable-Baselines3 文档: https://stable-baselines3.readthedocs.io/
- Gymnasium 文档: https://gymnasium.farama.org/