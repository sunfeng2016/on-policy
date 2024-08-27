# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-17
# @Description: Implementation of Agent Class

import os
import random
import pygame
import numpy as np

try:
    from onpolicy.envs.swarm_Confrontation.baseEnv import BaseEnv
except:
    from baseEnv import BaseEnv

from scipy.spatial import distance

os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

class ScoutEnv(BaseEnv):
    def __init__(self, args):
        super(ScoutEnv, self).__init__(args)

        # 蓝方待侦察区域配置
        self.scout_width = 6000
        self.scout_height = 4000
        self.scout_pos = np.array([-self.scout_width / 2, self.scout_height / 2]) # 侦察区域起始位置
        
        # 蓝方候选高价值区域配置
        self.candidate_core = self._initialize_candidate_cores()

        # 随机选择4个高价值区域
        self.core_ranges_num = 4

        # 蓝方的威胁区域配置
        self.threat_ranges_num = 3
        self.threat_ranges = self._initialize_threat_ranges()
        self.threat_center = np.array([d['center'] for d in self.threat_ranges])
        self.threat_radius = np.array([d['radius'] for d in self.threat_ranges])
        
        # 红方初始分布的区域配置
        self.red_group_num = 4
        self.red_init_range = self._initialize_red_ranges()

        # 划分网格配置
        self.grid_size = 50     # 每个小格子的尺寸
        self.num_grids_x = self.size_x // self.grid_size
        self.num_grids_y = self.size_y // self.grid_size
        self.num_grids = self.num_grids_x * self.num_grids_y
        # 所有网格的中心坐标和左上角坐标
        self.grid_centers, self.grid_left_tops = self._generate_grid_centers()
        # 计算扁平化的格子中心坐标
        self.grid_centers_flat = self.grid_centers.reshape(-1, 2)
        
        # 其它重要参数
        self.scout_dist = 25    # 飞机位置与格子的中心距离阈值
        self.guard_ratio = 0.3  # 蓝方防守在高价值区域的飞机的比例
        self.guard_dist = 100   # 防守的距离

        # 状态空间配置
        self._initialize_state_space()
        
        # 奖励参数配置
        self._initialize_rewards()
        
        # 威胁区域时间配置
        self.max_in_threat_time = 10
        
        # 侦察边界配置
        self._initialize_scout_bounds()

    def _initialize_candidate_cores(self):
        """
        初始化候选高价值区域配置
        """
        return [
            # {'center': np.array([-2250.0,  1250.0]), 'radius': 250.0},
            # {'center': np.array([-2250.0, -1250.0]), 'radius': 250.0}, 
            {'center': np.array([-1700.0,   700.0]), 'radius': 300.0}, 
            {'center': np.array([-1750.0, -1050.0]), 'radius': 250.0}, 
            {'center': np.array([ -700.0,  -100.0]), 'radius': 300.0}, 
            {'center': np.array([  300.0,  -800.0]), 'radius': 300.0}, 
            # {'center': np.array([ 2450.0,  1650.0]), 'radius': 250.0}, 
            # {'center': np.array([ 2250.0, -1250.0]), 'radius': 250.0}, 
        ]
    
    def _initialize_threat_ranges(self):
        """
        初始化威胁区域配置
        """
        return [
            {'center': np.array([-1250.0,   250.0]), 'radius': 250.0},
            {'center': np.array([-1100.0,  -700.0]), 'radius': 300.0},
            {'center': np.array([ 1000.0,  -800.0]), 'radius': 200.0},
        ]
        
    def _initialize_red_ranges(self):
        """
        初始化红方初始分布区域配置
        """
        return [
            {
                'x': [-self.scout_width/2, self.scout_width/2], 
                'y': [self.scout_height/2, self.size_y/2],
                'dir': -np.pi/2  
            },  # 上方
            {
                'x': [-self.scout_width/2, self.scout_width/2],
                'y': [-self.size_y/2, -self.scout_height/2],
                'dir': np.pi/2
            },  # 下方
            {
                'x': [-self.size_x/2, -self.scout_width/2],
                'y': [-self.scout_height/2, self.scout_height/2],
                'dir': 0
            }, # 左侧
            {
                'x': [self.scout_width/2, self.size_x/2],
                'y': [-self.scout_height/2, self.scout_height/2],
                'dir': np.pi
            }  # 右侧
        ]
        
    def _generate_grid_centers(self):
        """
        生成网格中心点和左上角顶点坐标
        """
        x_centers = np.linspace(-self.size_x/2 + self.grid_size/2,
                                self.size_x/2 - self.grid_size/2,
                                self.num_grids_x)
        y_centers = np.linspace(-self.size_y/2 + self.grid_size/2,
                                self.size_y/2 - self.grid_size/2,
                                self.num_grids_y)
        
        X, Y = np.meshgrid(x_centers, y_centers)
        grid_centers = np.stack((X, Y), axis=-1)
        grid_left_tops = grid_centers + np.array([-self.grid_size / 2, self.grid_size / 2])

        return grid_centers, grid_left_tops
    
    def _initialize_state_space(self):
        """
        初始化状态空间
        """
        base_state_size = self.get_state_size()
        state_size = [
            base_state_size[0] + self.num_grids, 
            [base_state_size[0]], 
            [1, self.num_grids_y, self.num_grids_x]
        ]
        self.share_observation_space = [state_size] * self.n_reds
        
    
    def _initialize_rewards(self):
        """
        初始化奖励参数配置
        """
        self.time_reward = 0.1
        self.scout_core_reward = 10
        self.scout_comm_reward = 1
        self.kill_reward = 0.5
        self.be_killed_penalty = -1
        self.out_scout_penalty = -0.05
        self.in_threat_penalty = -0.5  # 红方进入威胁区域的惩罚
        self.repeated_scouted_penalty = -0.1  # 重复侦查的惩罚
        self.reward_near_area = 0.1  # 靠近目标区域的奖励
        self.reward_away_area = -0.1
        self.reward_win = 1000
        self.reward_defeat = 0
        
    def _initialize_scout_bounds(self):
        """
        初始化侦察边界和相关配置
        """
        self.half_scout_size_x = self.scout_width / 2
        self.half_scout_size_y = self.scout_height / 2
        
        self.scout_bounds = np.array([
            [[ 1,  1], [-1,  1]],  # 上边界
            [[-1,  1], [-1, -1]],  # 左边界
            [[-1, -1], [ 1, -1]],  # 下边界
            [[ 1, -1], [ 1,  1]]   # 右边界
        ]) * np.array([self.half_scout_size_x, self.half_scout_size_y])
        
        # 计算侦查边界向量、长度以及单位向量
        self.scout_bounds_vec = self.scout_bounds[:, 1, :] - self.scout_bounds[:, 0, :]
        self.scout_bounds_len = np.linalg.norm(self.scout_bounds_vec, axis=1)
        self.scout_bounds_unitvec = self.scout_bounds_vec / self.scout_bounds_len[:, np.newaxis]
        
    def reset(self):
        super().reset()

        # 初始化格子状态：scouted_grids 表示每个格子的扫描情况，初始值为 False（未扫描）
        self.scouted_grids = np.zeros((self.num_grids_y, self.num_grids_x), dtype=bool)

        # 每个格子的类型 1为普通区域，其它类型在后续更新(2：高价值区域，3：威胁区域， 4: 非侦察区域)
        self.grids_type = np.ones((self.num_grids_y, self.num_grids_x), dtype=int)

        # 观测到格子信息观测情况
        self.grids_info = np.zeros_like(self.grids_type)
        
        # 重置并标记核心侦察区域
        self.core_grids = self._reset_circular_grids(self.core_ranges)
        self.core_grids_num = np.sum(self.core_grids)
        self.grids_type[self.core_grids] = 2
        
        # 重置并标记威胁区域
        self.threat_grids = self._reset_circular_grids(self.threat_ranges)
        self.threat_grids_num = np.sum(self.threat_grids)
        self.grids_type[self.threat_grids] = 3
        
        # 重置并标记非侦察区域
        self.out_grids = self._reset_out_grids()
        self.out_grids_num = np.sum(self.out_grids)
        self.grids_type[self.out_grids] = 4
        
        # 普通区域的格子(为标记为高价值,威胁或非侦察的区域)
        self.comm_grids = (self.grids_type == 1)
        self.comm_grids_num = np.sum(self.comm_grids)
        
        # 检查格子总数是否正确分类
        assert self.num_grids == (self.comm_grids_num + self.core_grids_num + self.threat_grids_num + self.out_grids_num)

        # 初始化各类统计信息
        self._reset_counters()
        
        # 红方智能体与目标区域的距离 
        self.dist2area = None               

        # 获取观测,状态和可用动作
        local_obs = self.get_obs()
        global_state = [self.get_state()] * self.n_reds
        available_actions = self.get_avail_actions()

        return local_obs, global_state, available_actions
    
    def _reset_circular_grids(self, ranges):
        """
        根据给定的圆形区域列表标记格子。
        
        参数:
        - ranges: 包含多个圆形区域的列表，每个区域包含 'center' 和 'radius'。

        返回:
        - grids: 布尔矩阵，表示该区域内的格子（True 为区域内）。
        """
        grids = np.zeros((self.num_grids_y, self.num_grids_x), dtype=bool)
        for r in ranges:
            is_in_range = self.in_circle(self.grid_centers_flat, r['center'], r['radius'])
            grids |= is_in_range.reshape(self.num_grids_y, self.num_grids_x)
        return grids

    def _reset_out_grids(self):
        """
        标记非侦察区域（超出侦察范围的格子）。

        返回:
        - out_grids: 布尔矩阵，表示非侦察区域的格子（True 为区域外）。
        """
        out_scout_x = (self.grid_centers_flat[:, 0] < -self.scout_width/2) | (self.grid_centers_flat[:, 0] > self.scout_width/2)
        out_scout_y = (self.grid_centers_flat[:, 1] < -self.scout_height/2) | (self.grid_centers_flat[:, 1] > self.scout_height/2)
        out_scout = out_scout_x | out_scout_y
        return out_scout.reshape(self.num_grids_y, self.num_grids_x)
    
    def _reset_counters(self):
        """
        重置计数器
        """
        # 核心区被侦察的格子数
        self.scouted_core_num = 0
        self.scouted_core_total = 0

        # 普通区被侦察的格子数
        self.scouted_comm_num = 0
        self.scouted_comm_total = 0

        # 在侦察区域外的智能体数
        self.outofscout_num = 0             # 每个时间步在侦察区域外的智能体数目
        self.outofscout_total = 0

        # 每步重复侦查的格子的数量
        self.repeated_scouted = 0
        
        # 在威胁区的智能体数目
        self.in_threat_red_num = 0
        self.in_threat_red_total = 0

        # 红方智能体进入威胁区的次数
        self.in_threat_time = np.zeros(self.n_reds)

    def get_state(self):
        """
        获取当前环境的全局状态。

        返回:
        - state: 包含智能体状态和已侦察区域信息的全局状态。
        """
        agent_state = super().get_state()
        
        # 预先将 self.scouted_grids 类型转换为浮点数
        scouted_grids_flat = self.scouted_grids.astype(float).flatten()

        # 直接连接智能体状态和侦察区域信息
        state = np.concatenate((agent_state, scouted_grids_flat))

        return state

    def in_threat_area(self):
        """
        判断红方智能体是否进入威胁区域，并更新相关状态。
        """
        in_threat = np.zeros(self.n_reds, dtype=bool)

        # 遍历所有威胁区域并进行向量化计算
        for threat_range in self.threat_ranges:
            in_threat |= self.in_circle(self.red_positions, threat_range['center'], threat_range['radius'])

        # 更新威胁区域内的红方智能体状态
        self.in_threat_red_num = np.sum(in_threat & self.red_alives)
        self.red_alives[in_threat] = False

    def in_circle(self, positions, center, radius, return_distance=False):
        """
        判断 positions 中的坐标是否在给定的圆形区域内。
        
        参数:
        - positions: 待判断的坐标列表，形状为 (n, 2)。
        - center: 圆形区域的中心坐标，形状为 (2,)。
        - radius: 圆形区域的半径。
        - return_distance: 是否返回距离。

        返回:
        - 是否在圆形区域内的布尔值数组，形状为 (n,)。
        - 如果 return_distance 为 True，同时返回距离数组。
        """
        # 计算所有位置与圆心的欧几里得距离
        distances = np.linalg.norm(positions - center, axis=1)
        
        # 判断是否在圆形区域内
        within_radius = distances < radius

        if return_distance:
            return within_radius, distances
        else:
            return within_radius
    
    def step(self, actions):
        """
        执行环境中的一步操作，更新状态，计算奖励，返回观测、状态、奖励和其他信息。
        
        参数:
        actions: 红方智能体的动作集，包含加速、航向和攻击动作。

        返回:
        local_obs: 各个红方智能体的局部观测值。
        global_state: 每个红方智能体的全局状态。
        rewards: 每个红方智能体的奖励值。
        dones: 每个红方智能体的完成标志。
        infos: 各种环境信息的字典。
        available_actions: 每个红方智能体可用的动作集。
        """
        
        # 获取红方的动作
        at = self.acc_actions[actions[:, 0]]
        pt = self.heading_actions[actions[:, 1]]
        attack_t = actions[:, 2]

        # 执行攻击动作
        self._perform_attack_actions(attack_t, pt)
        
        # 更新红方的位置和方向
        self._update_red_position_and_direction(at, pt)
        
        # 如果需要保存仿真数据，则记录红方的动作信息
        if self.save_sim_data:
            self.red_action = np.stack([at, pt * self.max_angular_vel * self.dt_time * 180 / np.pi, attack_t], axis=-1)
        
        # 更新侦察
        self.update_scout()
        
        # 执行蓝方的动作
        self.blue_step()
        
        # 合并状态
        self.merge_state()

        # 更新计步器
        self._total_steps += 1
        self._episode_steps += 1

        # 检查是否终止并计算奖励
        terminated, win, res = self.get_result()
        bad_transition = self._update_result(terminated, win)
        
        # 收集信息
        info = self._collect_info(bad_transition, res, win)

        # 获取观测、状态和奖励、可用动作
        local_obs = self.get_obs()
        global_state = [self.get_state()] * self.n_reds
        rewards = self.get_reward(win)
        dones = np.where(terminated, True, ~self.red_alives)
        infos = [info] * self.n_reds
        available_actions = self.get_avail_actions()
        
        # 存储数据
        if self.save_sim_data:
            self.dump_data()

        return local_obs, global_state, rewards, dones, infos, available_actions
    
    def _collect_info(self, bad_transition, res, win):
        """
        收集当前环境的统计信息
        
        参数:
        bad_transition: 是否为不良转移。
        res: 环境的其他结果信息。
        win: 是否获胜。

        返回:
        info: 包含环境信息的字典。
        """
        return {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            'bad_transition': bad_transition,
            'explode_ratio': self.red_self_destruction_total / self.n_reds, # 红方主动自爆的比例
            'be_exploded_ratio': self.explode_red_total / self.n_reds, # 红方被自爆的比例
            'invalid_explode_ratio': self.invalid_explode_red_total / self.n_reds, # 红方无效自爆的比例
            'collide_ratio': self.collide_blue_total / self.n_reds,   # 红方主动撞击的比例
            'be_collided_ratio': self.collide_red_total / self.n_reds, # 红方被撞击的比例
            'soft_kill_ratio': self.soft_kill_blue_total / self.n_reds,   # 红方软杀伤的比例
            'kill_num': self.explode_blue_total + self.collide_blue_total, # 红方毁伤蓝方的总数
            'hit_core_num': 0, # 高价值区域被打击的次数
            'explode_ratio_blue': self.blue_self_destruction_total / self.n_blues, # 蓝方主动自爆的比例
            'scout_core_ratio': self.scouted_core_total / self.core_grids_num, # 高价值区域被侦察的比例
            'scout_comm_ratio': self.scouted_comm_total / self.comm_grids_num, # 普通区域被侦察的比例
            'episode_length': self._episode_steps, # 轨迹长度
            'won': self.win_counted,
            "other": res
        }
        
    def get_avail_heading_actions(self):
        """
        根据智能体是否在侦察区域以及是否接近威胁区域，调整其有效的航向动作列表。
        
        包含以下两种情况的处理：
        1.如果智能体超出侦察区域，限制其航向动作，使其返回侦察区域。
        2.如果智能体接近威胁区域，限制其航向动作，使其远离威胁区域。
        
        返回：
        available_actions: 每个智能体的有效航向动作布尔矩阵
        """
        # 初始化每个智能体的所有航向动作为有效
        available_actions = np.ones((self.n_reds, self.heading_action_num), dtype=bool)
        
        # 检查智能体是否出界（侦察区域外）
        out_of_bounds_indices = self._get_out_of_bounds_inidces()
        if out_of_bounds_indices.size > 0:
            self._adjust_actions_for_scout_area(available_actions, out_of_bounds_indices)
        
        # 检查智能体是否接近威胁区域
        near_threat_agent_id, nearest_threat = self._get_near_threat_agent_indices()
        if near_threat_agent_id.size > 0:
            self._adjust_actions_for_threat_zone(available_actions, near_threat_agent_id, nearest_threat)
            
        return available_actions
        
    def _get_out_of_bounds_inidces(self):
        """
        获取超出侦察区域的智能体索引
        """
        out_of_bounds = (self.red_positions[:, 0] < -self.half_scout_size_x) | (self.red_positions[:, 0] > self.half_scout_size_x) | \
                        (self.red_positions[:, 1] < -self.half_scout_size_y) | (self.red_positions[:, 1] > self.half_scout_size_y)
        return np.where(out_of_bounds)[0]
    
    def _get_nearest_boundary_points(self, out_of_bounds_indices):
        """
        计算超出侦察区域的智能体到最近边界点的坐标。

        参数:
        - out_of_bounds_indices: 超出侦察区域的智能体索引。

        返回:
        - nearest_target: 每个智能体到最近边界点的坐标。
        """
        # 计算智能体当前位置到每个边界起点的向量
        pos_vec = self.red_positions[out_of_bounds_indices, np.newaxis, :] - self.scout_bounds[:, 0, :]  # (n, 4, 2)

        # 计算点向量的单位向量
        pos_unitvec = pos_vec / self.scout_bounds_len[:, np.newaxis]

        # 计算在边界上的投影比例
        t = np.einsum('nij,ij->ni', pos_unitvec, self.scout_bounds_unitvec)   # (n, 4)
        # 将 t 限制在 [0, 1] 范围内，确保投影点在线段上
        t = np.clip(t, 0.0, 1.0)

        # 计算线段上距离智能体最近的点的坐标 (投影点)
        nearest_points = self.scout_bounds[:, 0, :] + t[:, :, np.newaxis] * self.scout_bounds_vec[np.newaxis, :, :]  # (n, 4, 2)

        # 计算智能体当前位置到最近点的距离
        distances = np.linalg.norm(self.red_positions[out_of_bounds_indices, np.newaxis, :] - nearest_points, axis=2) # (n, 4)

        # 找到每个智能体距离最近的投影点
        nearest_id = np.argmin(distances, axis=1)    # (n)
        nearest_target = nearest_points[np.arange(out_of_bounds_indices.size), nearest_id] # (n, 2)

        return nearest_target
    
    def _adjust_actions_for_scout_area(self, available_actions, out_of_bounds_indices):
        """
        调整超出侦察区域的智能体的有效航向动作，使其返回侦察区域。
        
        参数：
        available_actions: 当前有效的动作列表。
        out_of_bounds_indices: 超出侦察区域的智能体索引。
        """
        # 计算超出侦察区域的智能体到最近边界点的期望方向
        nearest_target = self._get_nearest_boundary_points(out_of_bounds_indices)
        desired_directions = np.arctan2(nearest_target[:, 1] - self.red_positions[out_of_bounds_indices, 1],
                                        nearest_target[:, 0] - self.red_positions[out_of_bounds_indices, 0])    # (n)
        
        # 计算当前方向到期望方向的角度差
        angles_diff = (desired_directions - self.red_directions[out_of_bounds_indices] + np.pi) % (2 * np.pi) - np.pi # (n)
        angles_diff_threshold = self.max_angular_vel * self.dt_time

        # 根据角度差限制有效航向动作
        mask_pos = angles_diff >= angles_diff_threshold
        mask_neg = angles_diff <= -angles_diff_threshold

        # 对角度差大于阈值的智能体，只保留向左转的动作
        available_actions[out_of_bounds_indices[mask_pos], :self.heading_action_mid_id + 1] = 0

        # 对角度差小于负阈值的智能体，只保留向右转的动作
        available_actions[out_of_bounds_indices[mask_neg], self.heading_action_mid_id:] = 0
    
    def _get_near_threat_agent_indices(self):
        """
        获取接近威胁区域的智能体索引
        """  
        # 计算智能体当前位置到每个威胁区域中心的向量
        threat_vec = self.red_positions[:, np.newaxis, :] - self.threat_center[np.newaxis, :, :]
        
        # 计算智能体当前位置到中心点的距离
        threat_dist = np.linalg.norm(threat_vec, axis=-1)
        
        # 计算智能体当前位置到最近威胁区的中心点的索引，半径和距离
        nearest_id = np.argmin(threat_dist, axis=-1)
        nearest_threat_radius = self.threat_radius[nearest_id]
        nearest_dist = np.amin(threat_dist, axis=-1)
        
        # 选择接近威胁区的智能体
        near_threat_agent = (nearest_dist - (100 + nearest_threat_radius)) <= 0
        near_threat_agent_id = np.where(near_threat_agent)[0]
        
        # 接近威胁区的智能体接近威胁区的编号
        nearest_threat = self.threat_center[nearest_id][near_threat_agent_id]
        
        return near_threat_agent_id, nearest_threat
    
    def _adjust_actions_for_threat_zone(self, available_actions, near_threat_agent_id, nearest_threat):
        """
        调整接近威胁区域的智能体的有效航向动作，使其远离威胁区域。
        
        available_actions: 当前有效动作列表
        near_threat_agent_id: 接近威胁区域的智能体索引
        nearest_threat: 最近的威胁区的中心
        """
        # 计算远离威胁区的期望方向 (远离威胁区域中心的方向)
        desired_directions = -np.arctan2(nearest_threat[:, 1] - self.red_positions[near_threat_agent_id, 1],
                                         nearest_threat[:, 0] - self.red_positions[near_threat_agent_id, 0])
        
        # 计算当前方向到期望方向的角度差
        angles_diff = (desired_directions - self.red_directions[near_threat_agent_id] + np.pi) % (2 * np.pi) - np.pi
        angles_diff_threshold = self.max_angular_vel * self.dt_time
        
        # 根据角度差限制有效动作航向
        mask_pos = angles_diff >= angles_diff_threshold
        mask_neg = angles_diff <= -angles_diff_threshold
        
        # 对角度差大于阈值的，只保留左转的动作
        available_actions[near_threat_agent_id[mask_pos], :self.heading_action_mid_id + 1] = 0
        
        # 对于角度差小于负阈值的，只保留右转动作
        available_actions[near_threat_agent_id[mask_neg], self.heading_action_mid_id:] = 0       
    
    def blue_explode(self):
        """
        判断蓝方智能体是否需要自爆，并执行相应的操作。
        自爆规则：
        1. 如果红方存活智能体数量超过70%，且蓝方自爆范围内有超过3个红方智能体，蓝方自爆。
        2. 如果红方存活智能体数量在40%到70%之间，且蓝方自爆范围内有超过2个红方智能体，蓝方自爆。
        3. 如果红方存活智能体数量少于40%，且蓝方自爆范围内有超过1个红方智能体，蓝方自爆。
        """
        # 计算存活的红方智能体的比例
        alive_percentage = np.sum(self.red_alives) / self.n_reds
        
        # 计算蓝方智能体与红方智能体之间的距离
        distances_blue2red = distance.cdist(self.blue_positions, self.red_positions, 'euclidean')
        
        # 判断每个蓝方智能体自爆范围内的红方智能体数量
        red_in_explode_zone = (distances_blue2red < self.explode_radius) & self.red_alives
        red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)
        
        # 根据不同的存活比例选择自爆条件
        if alive_percentage >= 0.7:
            self_destruction_mask = red_counts_in_zone >= 3
        elif 0.4 <= alive_percentage < 0.7:
            self_destruction_mask = red_counts_in_zone >= 2
        else:
            self_destruction_mask = red_counts_in_zone >= 1
            
        # 只对存活的蓝方智能体执行自爆
        self_destruction_mask &= self.blue_alives
        
        # 记录自爆相关信息
        self.blue_self_destruction_mask = self_destruction_mask
        self.blue_self_destruction_num = np.sum(self_destruction_mask)
        self.blue_self_destruction_total += self.blue_self_destruction_num
        self.blue_action[self_destruction_mask, 2] = 1  # 标记自爆动作
        
        # 标记自爆范围内的红方智能体为死亡
        red_explode_mask = np.any(red_in_explode_zone[self_destruction_mask], axis=0)
        self.be_exploded_flag = red_explode_mask & self.red_alives
        self.red_explode_mask = self.be_exploded_flag   # 用于渲染
        self.explode_red_num = np.sum(self.red_explode_mask)
        self.explode_blue_total += self.explode_red_num
        self.red_alives[self.red_explode_mask] = False
        
        # 更新蓝方存活状态
        self.blue_alives[self_destruction_mask] = False
        
    def blue_collide(self, pt):
        """
        判断并处理蓝方智能体的撞击行为：
        规则如下：
        1. 如果蓝方智能体的攻击范围（扇形区域）内存在红方智能体，则撞击最近的红方智能体。
        2. 撞击成功后，蓝方和目标红方智能体都被标记为死亡。

        返回:
        - pt: 更新后的智能体航向改变比例。
        """
        # 计算蓝方智能体到红方智能体之间的方向向量和角度
        delta_blue2red = self.red_positions[np.newaxis, :, :] - self.blue_positions[:, np.newaxis, :]
        angles_blue2red = np.arctan2(delta_blue2red[:, :, 1], delta_blue2red[:, :, 0])

        # 计算蓝方智能体当前方向与到红方智能体的方向的角度差
        angles_diff_blue2red = (angles_blue2red - self.blue_directions[:, np.newaxis] + np.pi) % (2 * np.pi) - np.pi

        # 计算蓝方智能体到红方智能体之间的距离
        distances_blue2red = self.get_dist().T  # 过滤掉了死亡智能体

        # 蓝方智能体攻击范围内的红方智能体
        red_in_attack_zone = (distances_blue2red < self.attack_radius) & (
            np.abs(angles_diff_blue2red) < self.attack_angle / 2
        )

        # 对不在攻击范围内的红方，距离设置为无限大
        distances_blue2red[~red_in_attack_zone] = np.inf

        # 找个每个蓝方智能体最近的红方目标
        nearest_red_id = np.argmin(distances_blue2red, axis=1)

        # 如果攻击范围内没有红方智能体，则设置目标设为-1
        nearest_red_id[np.all(np.isinf(distances_blue2red), axis=1)] = -1

        # 生成蓝方的目标选择掩码，排除没有目标的蓝方智能体
        collide_mask = (nearest_red_id != -1) & self.blue_alives

        # 获取有效的目标红方 ID 和对应的蓝方 ID
        target_ids = nearest_red_id[collide_mask]
        agent_ids = np.where(collide_mask)[0]

        # 获取蓝方智能体与其目标之间的距离
        distances = distances_blue2red[collide_mask, target_ids]

        # 判断是否撞击成功
        collide_success_mask = distances < self.collide_distance
        
        # 获取撞击成功的蓝方和目标红方id
        success_agent_ids = agent_ids[collide_success_mask]
        success_target_ids = target_ids[collide_success_mask]

        # 存储撞击动作数据
        self.blue_action[success_agent_ids, 2] = 2

        # 记录蓝方撞击成功的智能体和目标，用作渲染
        self.blue_collide_agent_ids = success_agent_ids
        self.blue_collide_target_ids = success_target_ids

        # 更新红方的被撞击状态
        self.be_collided_flag = np.zeros(self.n_reds, dtype=bool)
        self.be_collided_flag[success_target_ids] = True

        # 记录撞击成功的数量
        self.collide_red_num = success_target_ids.size
        self.collide_red_total += self.collide_red_num

        # 更新蓝方和红方的存活状态
        self.blue_alives[success_agent_ids] = False
        self.red_alives[success_target_ids] = False

        # 更新蓝方智能体的方向并停止其移动
        self.blue_directions[collide_mask] = angles_blue2red[collide_mask, target_ids]
        pt[collide_mask] = 0

        return pt

    def blue_guard(self, pt):
        """
        控制蓝方智能体防守高价值区域。智能体会调整方向，向最近的高价值区域中心移动。
        
        参数:
        - pt: 蓝方智能体的当前航向改变比例。

        返回:
        - pt: 更新后的智能体航向改变比例。
        """
        # 初始化存储距离的数组和布尔掩码，判断是否在防守范围内
        in_core_range = np.zeros(self.n_blues, dtype=bool)
        distances_to_core = []
        
        # 遍历核心区域，判断蓝方智能体是否在防守范围内并计算与核心中心的距离
        for core in self.core_ranges:
            in_core_range_i, distances_i = self.in_circle(self.blue_positions, 
                core['center'], core['radius'] + self.guard_dist, return_distance=True)
            in_core_range |= in_core_range_i
            distances_to_core.append(distances_i)

        # 过滤掉游曳和死亡的智能体
        in_core_range[:self.free_blue_num] = True
        in_core_range |= ~self.blue_alives

        # 如果所有智能体都在防守范围内，则不进行任何操作
        if np.sum(in_core_range) == self.n_blues:
            return pt

        # 将距离数组合并为矩阵，并找到每个智能体最近的核心区域
        distances_to_core = np.column_stack(distances_to_core)
        nearest_core_indices = np.argmin(distances_to_core, axis=1)

        # 获取目标核心区域的中心坐标
        target_core_centers = self.core_centers[nearest_core_indices]

        # 计算期望方向
        desired_directions = np.arctan2(target_core_centers[:, 1] - self.blue_positions[:, 1],
                                        target_core_centers[:, 0] - self.blue_positions[:, 0])

        # 计算当前方向到期望方向的角度差，并将其归一化到[-pi,pi]区间
        angles_diff = (desired_directions - self.blue_directions + np.pi) % (2 * np.pi) - np.pi

        # 限制转向角度在最大角速度范围内
        angles_diff = np.clip(angles_diff, -self.max_angular_vel * self.dt_time, self.max_angular_vel * self.dt_time)

        # 更新蓝方智能体的方向，并限制未在防守范围内的智能体的移动
        self.blue_directions[~in_core_range] += angles_diff[~in_core_range]
        pt[~in_core_range] = 0

        return pt

    def blue_return(self, pt):
        """
        让超出缓冲区域边界的蓝方智能体返回有效区域。
        
        参数:
        - pt: 当前智能体航向改变比例。

        返回:
        - pt: 更新后的智能体航向改变比例。
        """
        # 判断智能体是否出缓冲边界
        out_of_bounds_x = (self.blue_positions[:, 0] < -self.half_size_x) | (self.blue_positions[:, 0] > self.half_size_x)
        out_of_bounds_y = (self.blue_positions[:, 1] < -self.half_size_y) | (self.blue_positions[:, 1] > self.half_size_y)
        out_of_bounds = out_of_bounds_x | out_of_bounds_y

        # 找到出界的智能体索引
        out_of_bounds_indices = np.where(out_of_bounds)[0]

        # 如果没有出界，直接返回
        if out_of_bounds_indices.size == 0:
            return pt

        # 计算智能体到每条边界线段的投影点
        pos_vec = self.blue_positions[out_of_bounds_indices, np.newaxis, :] - self.cache_bounds[:, 0, :]  # (n, 4, 2)
        pos_unitvec = pos_vec / self.bounds_len[:, np.newaxis]
        t = np.einsum('nij,ij->ni', pos_unitvec, self.bounds_unitvec)   # 计算投影比例 (n, 4)
        t = np.clip(t, 0.0, 1.0)    # 将投影比例规范化到 [0,1] 范围内

        # 计算最近的边界点坐标
        nearest = self.cache_bounds[:, 0, :] + t[:, :, np.newaxis] * self.bounds_vec[np.newaxis, :, :]  # (n, 4, 2)

        # 计算到投影点的距离
        nearest_dist = np.linalg.norm(self.blue_positions[out_of_bounds_indices, np.newaxis, :] - nearest, axis=2) # (n, 4)

        # 找到每个智能体最近的边界点
        nearest_id = np.argmin(nearest_dist, axis=1)    # (n)
        nearest_target = nearest[np.arange(out_of_bounds_indices.size), nearest_id] # (n, 2)

        # 计算智能体的期望方向
        desired_directions = np.arctan2(nearest_target[:, 1] - self.blue_positions[out_of_bounds_indices, 1],
                                        nearest_target[:, 0] - self.blue_positions[out_of_bounds_indices, 0])    # (n)
        
        # 计算当前方向到期望方向的角度差，并将其归一化到 [-pi, pi]
        angles_diff = (desired_directions - self.blue_directions[out_of_bounds_indices] + np.pi) % (2 * np.pi) - np.pi # (n)

        # 确保转向角度不超过最大角速度
        angles_diff = np.clip(angles_diff, -self.max_angular_vel * self.dt_time, self.max_angular_vel * self.dt_time)

        # 根性出界智能体的方向
        self.blue_directions[out_of_bounds_indices] += angles_diff
        pt[out_of_bounds] = 0

        return pt
    
    def blue_step(self):
        """
        处理蓝方智能体在每个时间步的行为，包括自爆、碰撞、防守、出界返回和软杀伤等操作。
        """
        # 1.初始化随机航向调整
        pt = np.random.uniform(-1.0, 1.0, size=self.n_blues)
        
        # 2.处理自爆行为
        self.blue_explode()
        
        # 3.处理蓝方与红方的自爆行为
        pt = self.blue_collide(pt)
        
        # 4.处理蓝方防守高价值区域的行为
        pt = self.blue_guard(pt)
        
        # 5.处理蓝方智能体出界后的返回行为
        pt = self.blue_return(pt)
        
        # 6.处理蓝方智能体受到软杀伤时的行为，航向保持不变
        # pt = self._apply_soft_kill(pt)
        
        # 7.更新蓝方智能体的方向和位置
        self._update_blue_position_and_direction(pt)
        
    def _apply_soft_kill(self, pt):
        """
        如果蓝方智能体在某个红方智能体的软杀伤范围内，则航向不变。
        """
        blue_soft_kill_mask = np.any(self.blue_in_soft_kill_time > 0, axis=0)
        pt[blue_soft_kill_mask] = 0
        
        return pt
        
    def _update_blue_position_and_direction(self, pt):
        """
        更新蓝方智能体的方向和位置
        仅更新存活的智能体
        """
        alive_mask = self.blue_alives
        
        # 更新方向，确保方向在 [-pi, pi] 范围内
        self.blue_directions[alive_mask] = (
            (self.blue_directions[alive_mask] + pt[alive_mask] * self.max_angular_vel * self.dt_time + np.pi) \
                % (2 * np.pi) - np.pi
        )
        
        # 更新位置
        dx = self.blue_velocities[alive_mask] * np.cos(self.blue_directions[alive_mask]) * self.dt_time
        dy = self.blue_velocities[alive_mask] * np.sin(self.blue_directions[alive_mask]) * self.dt_time
        self.blue_positions[alive_mask] += np.column_stack((dx, dy))
        
        # 存储数据
        self.blue_action[:, 1] = pt * self.max_angular_vel * self.dt_time
        
    def get_result(self):
        """
        判断当前仿真环境是否满足终止条件，并返回终止状态、胜利标志和相关信息。
        
        返回：
        - terminated: 布尔值，表示是否满足终止条件。
        - win: 布尔值，表示是否取得胜利。
        - info: 字符串，包含终止原因的描述信息。
        """
        # 计算存活的红方智能体数量
        n_red_alive = np.sum(self.red_alives)

        # 计算核心区域和普通区域的侦察比例
        core_percentage = self.scouted_core_total / self.core_grids_num
        comm_percentage = self.scouted_comm_total / self.comm_grids_num

        # 胜利条件判断：核心区域 90% 被侦查，且普通区域 70% 被侦查
        if core_percentage >= 0.9 and comm_percentage >= 0.7:
            return True, True, '[Win] Finish Scout.'
            
        # 失败条件判断：红方智能体全部死亡
        if n_red_alive == 0:
            return True, False, '[Defeat] All Dead.'
            
        # 失败条件判断：达到最大时间限制
        if self._episode_steps >= self.episode_limit:
            return True, False, '[Defeat] Time out.'
        
        return False, False, ""
    
    def get_dist_reward(self):
        """
        计算红方智能体距离目标区域的奖励。如果智能体不在目标区域内，且仍然存活，
        根据其距离边界的远近给予奖励或惩罚。
        """
        # 目标区域边界
        x_min, x_max = -self.half_scout_size_x, self.half_scout_size_x
        y_min, y_max = -self.half_scout_size_y, self.half_scout_size_y
        
        # 初始化奖励数组
        rewards = np.zeros(self.n_reds)

        # 获取红方智能体的当前位置
        red_x, red_y = self.red_positions[:, 0], self.red_positions[:, 1]

        # 判断智能体是否在目标区域内
        in_area = (x_min <= red_x) & (red_x <= x_max) & (y_min <= red_y) & (y_min <= y_max)
        
        # 筛选出不在目标区域但仍然存活的智能体
        out_area = ~in_area & self.red_alives 

        # 计算离区域边界最短的距离
        dx = np.maximum(x_min - red_x, 0, red_x - x_max)
        dy = np.maximum(y_min - red_y, 0, red_y - y_max)
        distance2area = dx + dy

        # 计算奖励: 靠近边界给奖励，远离边界给惩罚
        if self.dist2area is not None:
            improved_dist = distance2area[out_area] < self.dist2area[out_area]
            rewards[out_area] = np.where(improved_dist, self.reward_near_area, self.reward_away_area)
            
            # 更新智能体与目标区域的距离
            self.dist2area[out_area] = distance2area[out_area]
            self.dist2area[~out_area] = 0

        return rewards
    
    def get_reward(self, win=False):
        """
        计算当前时间步的整体奖励，结合多个因素，包括存活时间、扫描进度、毁伤敌方智能体等。
        """
        # 初始化每个智能体的奖励
        rewards = np.zeros(self.n_reds, dtype=float)

        # 计算各类奖励系数
        episode_progress = 1 + self._episode_steps / self.episode_limit
        dead_red_ratio = 1  + (1 - np.sum(self.red_alives) / self.n_reds)
        dead_blue_ratio = 1 + (1 - np.sum(self.blue_alives) / self.n_blues)
        scouted_core_ratio = 1 + self.scouted_core_total / self.core_grids_num
        scouted_comm_ratio = 1 + self.scouted_comm_total / self.comm_grids_num
        in_threat_progress = 1 + self.in_threat_time / self.max_in_threat_time

        # 时间奖励：存活时间越久，奖励系数越大
        rewards += self.time_reward * self.red_alives * episode_progress
        
        # 扫描奖励：高价值区域和低价值区域的扫描奖励
        rewards += self.scout_core_reward * self.scout_core_grids * scouted_core_ratio
        rewards += self.scout_comm_reward * self.scout_comm_grids * scouted_comm_ratio

        # 毁伤蓝方智能体的奖励，毁伤的敌方智能体数量越多，奖励系数越大
        kill_blue_agents = self.explode_flag | self.collide_flag
        rewards += self.kill_reward * kill_blue_agents * dead_blue_ratio

        # 获胜奖励或失败惩罚
        rewards += self.reward_win if win else self.reward_defeat

        # 被毁伤惩罚：被毁伤的智能体越多，惩罚系数越大
        kill_red_agents = self.be_exploded_flag | self.be_collided_flag | self.dead_in_threat
        rewards += self.be_killed_penalty * kill_red_agents * dead_red_ratio

        # 在威胁区内的惩罚
        rewards += self.in_threat_penalty * self.scout_threat_grids * in_threat_progress

        # 重复侦查区域惩罚
        rewards += self.repeated_scouted_penalty * self.repeated_scouted

        # 在扫描区外的智能体，靠近侦查区域给奖励，远离给惩罚
        rewards += self.get_dist_reward()

        return rewards.reshape(-1, 1).tolist()
    
    def update_scout(self):
        """
        更新红方智能体的侦察状态，包括格子被侦察情况、重复侦察的格子、
        以及根据格子类型判断属于普通区、高价值区、威胁区或非侦察区。
        """
        
        # 1.将中心点为 (0,0) 的坐标转换为左下角为原点的坐标
        shifted_positions = (self.red_positions + np.array([self.size_x/2, self.size_y/2])) # (100, 2)

        # 2.计算智能体所在格子的索引
        grid_indices_x = (shifted_positions[:, 0] // self.grid_size).astype(int)    # (100, )
        grid_indices_y = (shifted_positions[:, 1] // self.grid_size).astype(int)    # (100, )

        # 3.判断智能体是否出界
        out_of_bound = (
            (grid_indices_x >= self.num_grids_x) | (grid_indices_y >= self.num_grids_y) | 
            (grid_indices_x < 0) | (grid_indices_y < 0)
        )

        # 处理出界情况，将索引设置为无效值(0, 0)
        grid_indices_x[out_of_bound] = 0
        grid_indices_y[out_of_bound] = 0

        # 4.判定哪些智能体正在进行侦察
        scouted = ~out_of_bound & self.red_alives
        
        # 5.获取已经被侦察过的格子，并更新重复侦察的格子
        already_scouted = self.scouted_grids[grid_indices_y, grid_indices_x] # (100, )
        new_scouted = scouted & ~already_scouted # (100, )
        
        self.repeated_scouted = scouted & already_scouted # (100,)

        # 6.更新格子的扫描状态
        self.scouted_grids[grid_indices_y[new_scouted], grid_indices_x[new_scouted]] = True

        # 7.更新观测到的格子信息
        self.grids_info[self.scouted_grids] = self.grids_type[self.scouted_grids]

        # 8.更新低价值区的扫描情况
        self.scout_comm_grids = np.zeros(self.n_reds, dtype=bool)
        self.scout_comm_grids[new_scouted] = self.comm_grids[grid_indices_y[new_scouted], grid_indices_x[new_scouted]]
        self.scouted_comm_total = np.sum(self.comm_grids & self.scouted_grids)

        # 9.更新高价值区的扫描情况
        self.scout_core_grids = np.zeros(self.n_reds, dtype=bool)
        self.scout_core_grids[new_scouted] = self.core_grids[grid_indices_y[new_scouted], grid_indices_x[new_scouted]]
        self.scouted_core_total = np.sum(self.core_grids & self.scouted_grids)

        # 10.更新威胁区的扫描情况
        self.scout_threat_grids = self.threat_grids[grid_indices_y, grid_indices_x] & self.red_alives
        self.in_threat_time[self.scout_threat_grids] += 1
        self.in_threat_time[~self.scout_threat_grids] = 0
        self.dead_in_threat = (self.in_threat_time >= self.max_in_threat_time)
        self.red_alives[self.dead_in_threat] = False
        
        # 11. 更新非侦察区域的扫描情况
        self.scout_out_grids = self.out_grids[grid_indices_y, grid_indices_x] & self.red_alives

    def init_in_rect(self, x_range, y_range, angle, num):
        """
        在给定的矩形区域内随机生成智能体的位置，并统一设置其方向。
        
        参数:
        - x_range: x 方向的范围 (min, max)。
        - y_range: y 方向的范围 (min, max)。
        - angle: 统一的初始方向角度。
        - num: 生成的智能体数量。
        
        返回:
        - positions: 智能体的位置数组 (num, 2)。
        - directions: 智能体的方向数组 (num,)。
        """
        
        x = np.random.uniform(x_range[0] + 100, x_range[1] - 100, num)
        y = np.random.uniform(y_range[0] + 100, y_range[1] - 100, num)

        positions = np.column_stack((x, y))
        directions = np.full(num, angle)

        return positions, directions
    
    def init_in_circle(self, center, radius, num):
        """
        在给定的圆形区域内随机生成智能体的位置。
    
        参数:
        - center: 圆心坐标 (x, y)。
        - radius: 圆的半径。
        - num: 生成的智能体数量。
        
        返回:
        - positions: 智能体的位置数组 (num, 2)。
        """
        angles = np.random.uniform(0, 2 * np.pi, num)
        radii = radius * np.sqrt(np.random.uniform(0, 1, num))

        # 计算智能体的位置
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)

        positions = np.column_stack((x, y))

        return positions
    
    def generate_blue_positions(self):
        """
        随机生成蓝方智能体的位置和方向。
        
        蓝方智能体分为高价值区域防守和随机游曳两类。
        
        返回:
        - positions: 蓝方智能体的位置数组 (n_blues, 2)。
        - directions: 蓝方智能体的方向数组 (n_blues,)。
        """
        # 随机选取高价值区域
        self.core_ranges = random.sample(self.candidate_core, self.core_ranges_num)
        self.core_centers = np.vstack([core['center'] for core in self.core_ranges])
        
        # 计算每个高价值区域的面积和概率分布
        areas = np.array([core['radius'] ** 2 for core in self.core_ranges])
        ratios = areas / np.sum(areas)
        probs = np.array([1-self.guard_ratio] + list(self.guard_ratio * ratios))

        # 根据概率分布生成各组智能体的数量
        group_sizes = np.random.multinomial(self.n_blues, probs)
        in_rect_size = group_sizes[0]
        in_core_sizes = group_sizes[1:]
        
        # 生成在矩形区域内随机游曳的智能体位置
        in_rect_positions = (np.random.rand(in_rect_size, 2) - 0.5) * (np.array([self.scout_width, self.scout_height]) -100)
        
        # 生成在高价值区域内的智能体位置
        in_core_positions = np.vstack([
            self.init_in_circle(core['center'], core['radius'], size)
            for core, size in zip(self.core_ranges, in_core_sizes)
        ]) 

        positions = np.vstack([in_rect_positions, in_core_positions])
        directions = np.random.uniform(-np.pi, np.pi, self.n_blues)

        # 随机游曳的蓝方智能体数量
        self.free_blue_num = in_rect_size

        return positions, directions

    def generate_red_positions(self):
        """
        随机生成红方智能体的位置和方向。
        
        红方智能体在不同的矩形区域内生成，每个区域有统一的初始方向。
        
        返回:
        - positions: 红方智能体的位置数组 (n_reds, 2)。
        - directions: 红方智能体的方向数组 (n_reds,)。
        """
        
        # 根据红方分组数量生成每组智能体的数量
        groups_size = np.random.multinomial(self.n_reds, np.ones(self.red_group_num) / self.red_group_num)
        init_ranges = random.sample(self.red_init_range, self.red_group_num)

        positions_directions = [
            self.init_in_rect(init_range['x'], init_range['y'], init_range['dir'], size)
            for init_range, size in zip(init_ranges, groups_size)
        ]
        
        positions, directions = zip(*positions_directions)
        positions = np.vstack(positions)
        directions = np.hstack(directions)

        return positions, directions
   
    def init_positions(self):
        """
        初始化红方和蓝方智能体的位置、方向和速度。
    
        返回:
        - positions: 所有智能体的位置数组 (n_reds + n_blues, 2)。
        - directions: 所有智能体的方向数组 (n_reds + n_blues,)。
        - velocities: 所有智能体的速度数组 (n_reds + n_blues,)。
        """
        red_positions, red_directions = self.generate_red_positions()
        blue_positions, blue_directions = self.generate_blue_positions()

        positions = np.vstack([red_positions, blue_positions])
        directions = np.hstack([red_directions, blue_directions])
        velocities = np.hstack([np.full(self.n_reds, self.red_max_vel), np.full(self.n_blues, self.blue_max_vel)])

        return positions, directions, velocities

    def transform_circles(self):
        """
        将核心区域和威胁区域的圆心和半径转换为屏幕坐标，并设置可视化属性
        """
        circles = self.core_ranges + self.threat_ranges
        
        # 初始化转换后的圆心和半径列表
        self.transformed_circles_center = [
            self.transform_position(circle['center']) for circle in circles
        ]
        
        self.transformed_circles_radius = [
            circle['radius'] * self.scale_factor for circle in circles
        ]
        
        # 设置可视化属性：圆的线宽和颜色
        self.num_circles = self.core_ranges_num + self.threat_ranges_num
        self.circles_width = [5] * self.num_circles
        self.circles_color = [(0, 0, 255)] * self.core_ranges_num + [(255, 0, 0)] * self.threat_ranges_num
        
    def transform_grids(self):
        """
        将网格区域的左上角坐标转换为屏幕坐标
        """
        # 计算转换中心和方向
        new_center = np.array([-self.size_x / 2, self.size_y / 2])
        new_dir = np.array([1, -1])

        # 计算屏幕上的网格大小和左上角位置
        self.screen_grid_size = int(self.grid_size * self.scale_factor)
        self.screen_grid_left_tops = ((self.grid_left_tops - new_center) * new_dir * self.scale_factor).astype(int)

    def transform_rect(self):
        """
        将矩形区域的位置和大小转换到屏幕坐标系中
        """
        transformed_rect_center = self.transform_position(self.scout_pos)
        self.transformed_rect = (
            transformed_rect_center[0],
            transformed_rect_center[1],
            self.scout_width * self.scale_factor,
            self.scout_height * self.scale_factor
        )

    def _transformer_coordinate(self):
        """
        将场景中的相关坐标转换到屏幕坐标
        """
        # 转换圆形和网格的位置
        self.transform_circles()
        self.transform_grids()
        
        # 转换矩形位置
        self.transform_rect()
        
    def _render_scenario(self):
        """
        渲染跟场景相关的元素
        """
        # 渲染侦察区域
        self._render_rect()
        
        # 渲染高价值区域和威胁区域
        self._render_circles()
        
        # 渲染网格
        self._render_grids()
    
    def _render_rect(self):
        """
        渲染矩形
        """
        pygame.draw.rect(self.screen, (0, 0, 0), self.transformed_rect, 3)
    
    def _render_circles(self):
        """
        渲染圆形
        """
        for i in range(self.num_circles):
            pygame.draw.circle(
                self.screen,
                self.circles_color[i],
                self.transformed_circles_center[i],
                self.transformed_circles_radius[i],
                width=self.circles_width[i]
            )
    
    def _render_grids(self):
        """
        渲染网格
        """
        for i in range(self.num_grids_y):
            for j in range(self.num_grids_x):
                if not (self.out_grids[i, j] or not self.scouted_grids[i, j]):
                    x, y = self.screen_grid_left_tops[i, j, :]
                    pygame.draw.rect(self.screen, (0, 0, 0), (x, y, self.screen_grid_size, self.screen_grid_size), 1)
    
    def _render_text(self):
        """
        渲染屏幕上的文本信息
        """
        red_alive = sum(self.red_alives)
        blue_alive = sum(self.blue_alives)
        scout_core_ratio = self.scouted_core_total / self.core_grids_num
        scout_comm_ratio = self.scouted_comm_total / self.comm_grids_num

        time_text = self.font.render(
            f'Episode: {self._episode_count} Time Step: {self._episode_steps} Win count: {self.battles_won}',
            True, (0, 0, 0)
        )
        red_text = self.font.render(f'Red Alive: {red_alive}', True, (255, 0, 0))
        blue_text = self.font.render(f'Blue Alive: {blue_alive}', True, (0, 0, 255))
        scout_text = self.font.render(f'Scout Core: {round(scout_core_ratio, 2)} Scout Comm: {round(scout_comm_ratio, 2)}', True, (255, 0, 0))

        self.screen.blit(red_text, (10, 10))
        self.screen.blit(blue_text, (10, 50))
        self.screen.blit(scout_text, (10, 90))
        self.screen.blit(time_text, (10, 130))
    
class Arg(object):
    def __init__(self) -> None:
        self.map_name = '100_vs_100'
        self.scenario_name = 'scout'
        self.episode_length = 400
        self.use_script = True
        self.save_sim_data = True
        self.plane_name = "plane_scout"

if __name__ == "__main__":
    args = Arg()

    env = ScoutEnv(args)

    env.reset()
    
    import time
    for i in range(50):
        start = time.time()
        env.render()
        actions = env.random_policy_red()
        env.step(actions)
        env.render()
        print(f'[frame: {i}]---[Time: {time.time() - start}]')

    env.close()
    # indices, distances = env.find_nearest_grid()

