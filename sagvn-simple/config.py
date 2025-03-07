# config.py
CONFIG = {
    # 系统参数
    'num_vehicles': 5,
    'num_edges': 2,
    # 新增参数
    'min_data': 1,
    'max_data': 10,
    'min_compute': 1,
    'max_compute': 10,
    'min_delay': 1,
    'max_delay': 10,
    'min_distance': 0.1,
    'max_distance': 10,
    
    # 任务参数
    'task_size_min': 1,
    'task_size_max': 10,
    
    # 资源参数
    'computing_resources': [100, 100],
    'bandwidth': [50, 50],
    
    # 训练参数 - 调整以更好利用GPU
    'total_timesteps': 1_000_000,  # 增加训练步数
    'learning_rate': 3e-4,
    'batch_size': 256,  # 增大批量大小
    
    # GPU优化参数
    'gpu_params': {
        'use_amp': True,  # 使用自动混合精度
        'optimizer_kwargs': {
            'eps': 1e-5,
            'weight_decay': 1e-5
        }
    },

    # 计算能力参数
    'local_computing_power': 1.0,    # 本地计算能力
    'an_computing_power': 3.0,       # 空中节点计算能力
    'gn_computing_power': 4.0,       # 地面节点计算能力

    # 传输速率参数
    'an_transmission_rate': 3.0,     # 空中节点传输速率
    'gn_transmission_rate': 4.0,     # 地面节点传输速率
    
    # 新增：softmax温度参数
    'softmax_temperature': 1.0,
    
    'wandb': {
        'project': 'sagvn',
        'name': 'ppo_baseline',
        'group': 'experiments'
    }
}