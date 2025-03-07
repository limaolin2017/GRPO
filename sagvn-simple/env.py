import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SimpleVehicularEnv(gym.Env): 
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_vehicles = config['num_vehicles']
        
        # 1. 状态空间保持不变
        state_dim = self.num_vehicles * (
            1 +  # 任务数据量
            1 +  # 计算需求
            1 +  # 时延约束
            2    # 与两类节点的距离
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # 2. 动作空间
        total_actions = (
            self.num_vehicles * 3 +  # 每个车辆3个概率值(本地,AN,GN)
            self.num_vehicles * 2    # 资源分配比例(AN,GN)
        )
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(total_actions,),
            dtype=np.float32
        )
    
    def _softmax(self, x):
        """计算softmax值"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def reset(self, seed=None, options=None):  # 这是reset方法
        """重置环境到初始状态"""
        super().reset(seed=seed)  # 必须调用父类的reset
        
        # 初始化任务状态
        self.task_data = np.random.uniform(
            self.config['min_data'],
            self.config['max_data'],
            self.num_vehicles
        )
        self.compute_required = np.random.uniform(
            self.config['min_compute'],
            self.config['max_compute'],
            self.num_vehicles
        )
        self.delay_constraints = np.random.uniform(
            self.config['min_delay'],
            self.config['max_delay'],
            self.num_vehicles
        )
        self.distances = np.random.uniform(
            self.config['min_distance'],
            self.config['max_distance'],
            (self.num_vehicles, 2)  # 改为2个节点的距离
        )
        
        # 返回初始状态和信息
        return self._get_state(), {}
    
    def step(self, action):
        # 1. 解析动作
        offload_probs = []
        for i in range(self.num_vehicles):
            # 获取每个车辆的三个概率值
            start_idx = i * 3
            vehicle_probs = action[start_idx:start_idx + 3]
            # 应用softmax得到概率分布
            probs = self._softmax(vehicle_probs)
            offload_probs.append(probs)
        
        # 2. 根据概率选择卸载决策
        offload_decisions = []
        for probs in offload_probs:
            # 根据概率选择动作(0:本地, 1:AN, 2:GN)
            decision = np.random.choice(3, p=probs)
            offload_decisions.append(decision)
        
        offload_decisions = np.array(offload_decisions)
        
        # 3. 处理资源分配
        resource_start_idx = self.num_vehicles * 3
        resource_alloc = action[resource_start_idx:]
        resource_alloc = self._normalize_resources(resource_alloc)

        # 4. 计算完成时间和存储
        completion_times = self._calculate_completion_time(
            offload_decisions, 
            resource_alloc
        )
        storage_allocated = self._calculate_storage(
            offload_decisions, 
            resource_alloc
        )
        
        # 5. 计算奖励
        reward = self._calculate_reward(
            completion_times,
            storage_allocated
        )
        
        # 6. 更新状态
        self.task_data = np.random.uniform(
            self.config['min_data'],
            self.config['max_data'],
            self.num_vehicles
        )
        self.compute_required = np.random.uniform(
            self.config['min_compute'],
            self.config['max_compute'],
            self.num_vehicles
        )
        self.delay_constraints = np.random.uniform(
            self.config['min_delay'],
            self.config['max_delay'],
            self.num_vehicles
        )
        self.distances = np.random.uniform(
            self.config['min_distance'],
            self.config['max_distance'],
            (self.num_vehicles, 2)
        )
        
        return self._get_state(), reward, False, False, {}

    # 添加新方法
    def _normalize_resources(self, resource_alloc):
        normalized = np.zeros_like(resource_alloc)
        for i in range(2):  # 改为2种节点类型
            start_idx = i * self.num_vehicles
            end_idx = (i + 1) * self.num_vehicles
            resources = resource_alloc[start_idx:end_idx]
            sum_resources = np.sum(resources) + 1e-10
            normalized[start_idx:end_idx] = resources / sum_resources
        return normalized
    
    def _get_state(self):
        return np.concatenate([
            self.task_data,           # d_i^t
            self.compute_required,     # c_i^t
            self.delay_constraints,    # t_i^t
            self.distances.flatten()   # ϵ_i^t,κ
        ])
    
    def _calculate_reward(self, completion_times, storage_allocated):
        reward = 0
        for i in range(self.num_vehicles):
            # 时延约束项
            time_penalty = min(0, completion_times[i] - self.delay_constraints[i])
            # 存储约束项
            storage_penalty = min(0, storage_allocated[i] - self.task_data[i])
            reward += time_penalty + storage_penalty
        return reward
    
    def _local_compute_time(self, vehicle_idx):
        """计算本地处理时间"""
        # 简化计算: 任务数据量 * 计算需求 / 本地计算能力
        local_computing_power = self.config['local_computing_power']
        return (self.task_data[vehicle_idx] * 
                self.compute_required[vehicle_idx] / 
                local_computing_power)

    def _offload_compute_time(self, vehicle_idx, offload_target, resource_alloc):
        """计算卸载处理时间
        Args:
            vehicle_idx: 车辆索引
            offload_target: 1(SN), 2(AN), 3(GN)
            resource_alloc: 资源分配比例
        """
        # 1. 确定目标节点类型
        if offload_target == 1:    # Aerial Node
            computing_power = self.config['an_computing_power']
            transmission_rate = self.config['an_transmission_rate']
            node_idx = 0
        else:                      # Ground Node
            computing_power = self.config['gn_computing_power']
            transmission_rate = self.config['gn_transmission_rate']
            node_idx = 1

        # 2. 计算传输时间：数据量/传输速率
        transmission_time = (self.task_data[vehicle_idx] / 
                            transmission_rate * 
                            (1 + self.distances[vehicle_idx][node_idx]))

        # 3. 计算处理时间：计算需求/(计算能力*资源分配比例)
        resource_idx = vehicle_idx + node_idx * self.num_vehicles
        processing_time = (self.compute_required[vehicle_idx] / 
                        (computing_power * resource_alloc[resource_idx]))

        # 4. 总时间 = 传输时间 + 处理时间
        return transmission_time + processing_time
    
    def _calculate_completion_time(self, offload_decisions, resource_alloc):
        # 计算每个任务的完成时间
        completion_times = np.zeros(self.num_vehicles)
        for i in range(self.num_vehicles):
            if offload_decisions[i] == 0:  # 本地计算
                completion_times[i] = self._local_compute_time(i)
            else:  # 卸载计算
                completion_times[i] = self._offload_compute_time(
                    i, 
                    offload_decisions[i], 
                    resource_alloc
                )
        return completion_times
    
    def _calculate_storage(self, offload_decisions, resource_alloc):
        # 计算分配的存储资源
        storage = np.zeros(self.num_vehicles)
        # 实现存储资源分配逻辑
        return storage