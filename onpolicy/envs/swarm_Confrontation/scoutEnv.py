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

from scipy.spatial import distance, cKDTree

os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

class ScoutEnv(BaseEnv):
    def __init__(self, args):
        super(ScoutEnv, self).__init__(args)

        # 蓝方待侦察区域配置
        self.scout_width = 6000
        self.scout_height = 4000
        self.scout_pos = np.array([-self.scout_width / 2, self.scout_height / 2]) # 侦察区域起始位置(左上角)
        
        # 蓝方候选高价值区域配置
        self.candidate_core = self._initialize_candidate_cores()

        # 随机选择4个高价值区域
        self.core_ranges_num = 4

        # 蓝方的威胁区域配置
        self.threat_ranges_num = 3
        self.threat_ranges = self._initialize_threat_ranges()
        self.threat_ranges_center = np.array([d['center'] for d in self.threat_ranges])
        self.threat_ranges_radius = np.array([d['radius'] for d in self.threat_ranges])
        
        # 红方初始分布的区域配置
        self.red_group_num = 4
        self.red_init_range = self._initialize_red_ranges()

        # 划分网格配置
        self._init_grids()
        
        # 其它重要参数
        self.scout_dist = 25    # 飞机位置与格子的中心距离阈值
        self.guard_ratio = 0.3  # 蓝方防守在高价值区域的飞机的比例
        self.guard_dist = 100   # 防守的距离

        # 状态空间配置
        self._initialize_state_space()
        
        # 奖励参数配置
        # self._initialize_rewards()
        
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
        # 列：从左到右
        x_centers = np.linspace(-self.size_x/2 + self.grid_size/2,
                                self.size_x/2 - self.grid_size/2,
                                self.grids_col_num)
        # 行：从上到下
        y_centers = np.linspace(self.size_y/2 - self.grid_size/2,
                                -self.size_y/2 + self.grid_size/2,
                                self.grids_row_num)
        
        X, Y = np.meshgrid(x_centers, y_centers)
        grid_centers = np.stack((X, Y), axis=-1)
        grid_left_tops = grid_centers + np.array([-self.grid_size / 2, self.grid_size / 2])

        return grid_centers, grid_left_tops
    
    def _init_grids(self):
        """
        配置网格信息
        """
        self.grid_size = 50     # 每个小格子的尺寸
        
        self.grids_row_num = self.size_y // self.grid_size # 网格的行数
        self.grids_col_num = self.size_x // self.grid_size # 网格的列数
        self.grids_num = self.grids_row_num * self.grids_col_num
        
        # 所有网格的中心坐标和左上角坐标
        self.grid_centers, self.grid_left_tops = self._generate_grid_centers()
        # 计算扁平化的格子中心坐标
        self.grid_centers_flat = self.grid_centers.reshape(-1, 2)
    
    def _initialize_state_space(self):
        """
        初始化状态空间
        """
        self.red_state_size = 4             # x, y, v, phi
        self.blue_state_size = 4            # x, y, v, phi
        
        base_state_size = self.get_state_size()
        state_size = [
            base_state_size[0] + self.grids_num, 
            [base_state_size[0]], 
            [1, self.grids_row_num, self.grids_col_num]
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
        """
        重置环境
        """
        super().reset()

        # 重置格子的状态和类型
        self._reset_grids_type_and_state()

        # 红方智能体进入威胁区的次数
        self.in_threat_zone_time = np.zeros(self.n_reds)
        
        # 红方智能体与目标区域的距离 
        self.dist2area = None               

        # 获取观测,状态和可用动作
        local_obs = self.get_obs()
        global_state = [self.get_state()] * self.n_reds
        available_actions = self.get_avail_actions()

        return local_obs, global_state, available_actions
    
    def _reset_core_ranges(self):
        """
        重置高价值区域
        """
        # 随机选取高价值区域
        self.core_ranges = random.sample(self.candidate_core, self.core_ranges_num)
        self.core_ranges_center = np.array([d['center'] for d in self.core_ranges])
        self.core_ranges_radius = np.array([d['radius'] for d in self.core_ranges])
    
    def _reset_grids_type_and_state(self):
        """
        重置格子的类似和扫描状态
        """
        # 初始化格子状态：scouted_grids 表示每个格子的扫描情况，初始值为 False（未扫描）
        self.grids_scout_mask = np.zeros((self.grids_row_num, self.grids_col_num), dtype=bool)

        # 每个格子的类型 1为普通区域，其它类型在后续更新(2：高价值区域，3：威胁区域， 4: 非侦察区域)
        self.grids_type = np.ones((self.grids_row_num, self.grids_col_num), dtype=int)

        # 观测到格子信息
        self.grids_info = np.zeros_like(self.grids_type)
        
        # 重置并标记核心侦察区域
        self.core_grids_mask = self._reset_circular_grids(self.core_ranges)
        self.core_grids_num = np.sum(self.core_grids_mask)
        self.grids_type[self.core_grids_mask] = 2
        
        # 重置并标记威胁区域
        self.threat_grids_mask = self._reset_circular_grids(self.threat_ranges)
        self.threat_grids_num = np.sum(self.threat_grids_mask)
        self.grids_type[self.threat_grids_mask] = 3
        
        # 重置并标记非侦察区域
        self.out_grids_mask = self._reset_out_grids()
        self.out_grids_num = np.sum(self.out_grids_mask)
        self.grids_type[self.out_grids_mask] = 4
        
        # 普通区域的格子(为标记为高价值,威胁或非侦察的区域)
        self.comm_grids_mask = (self.grids_type == 1)
        self.comm_grids_num = np.sum(self.comm_grids_mask)
        
        # 检查格子总数是否正确分类
        assert self.grids_num == (self.comm_grids_num + self.core_grids_num + self.threat_grids_num + self.out_grids_num)
    
    def _reset_circular_grids(self, ranges):
        """
        根据给定的圆形区域列表标记格子。
        
        参数:
        - ranges: 包含多个圆形区域的列表，每个区域包含 'center' 和 'radius'。

        返回:
        - grids: 布尔矩阵，表示该区域内的格子（True 为区域内）。
        """
        grids = np.zeros((self.grids_row_num, self.grids_col_num), dtype=bool)
        for r in ranges:
            is_in_range = self.in_circle(self.grid_centers_flat, r['center'], r['radius'])
            grids |= is_in_range.reshape(self.grids_row_num, self.grids_col_num)
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
        
        return out_scout.reshape(self.grids_row_num, self.grids_col_num)    

    def get_state(self):
        """
        获取当前环境的全局状态。

        返回:
        - state: 包含智能体状态和已侦察区域信息的全局状态。
        """
        agents_state = super().get_state()
        
        # 预先将 self.scouted_grids 类型转换为浮点数
        grids_state = self.grids_scout_mask.astype(float).flatten()

        # 直接连接智能体状态和侦察区域信息
        state = np.concatenate((agents_state, grids_state))

        return state

    def is_red_in_threat_zone(self):
        """
        判断红方智能体是否进入威胁区域，并更新相关状态。
        """
        in_threat = np.zeros(self.n_reds, dtype=bool)

        # 遍历所有威胁区域并进行向量化计算
        for threat_range in self.threat_ranges:
            in_threat |= self.in_circle(self.red_positions, threat_range['center'], threat_range['radius'])

        # 更新在威胁区域内的时间
        self.in_threat_zone_time[in_threat] += 1
        self.in_threat_zone_time[~in_threat] = 0
        
        # 毁伤掩码：在威胁区域内停留超过最大时间的智能体将被标记为死亡
        kill_mask = self.in_threat_zone_time > self.max_in_threat_time
        self.red_alives[kill_mask] = False

        self.red_threat_damage_mask = kill_mask
        self.red_threat_damage_count = np.sum(kill_mask)
        self.red_threat_damage_total += self.red_collide_damage_count

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
        - actions: 红方智能体的动作集，包含加速、航向和攻击动作。

        返回:
        - local_obs: 各个红方智能体的局部观测值。
        - global_state: 每个红方智能体的全局状态。
        - rewards: 每个红方智能体的奖励值。
        - dones: 每个红方智能体的完成标志。
        - infos: 各种环境信息的字典。
        - available_actions: 每个红方智能体可用的动作集。
        """
        # 重置动作掩码和数量
        self._reset_action_mask_and_counter()
        
        # 执行红方的动作
        self.red_step(actions)
        
        # 更新侦察
        self.update_scout()
        
        # 执行蓝方的动作
        self.blue_step()

        # 更新计步器
        self._total_steps += 1
        self._episode_steps += 1

        # 检查是否终止并计算奖励
        terminated, win, res = self.get_result()
        bad_transition = self._update_result(terminated, win)
        
        # 收集信息
        info = self._collect_info(bad_transition, res, terminated)

        # 获取观测、状态和奖励、可用动作
        local_obs = self.get_obs()
        global_state = [self.get_state()] * self.n_reds
        # rewards = self.get_reward(win)
        rewards = [[self.get_reward(win)]] * self.n_reds
        dones = np.where(terminated, True, ~self.red_alives)
        infos = [info] * self.n_reds
        available_actions = self.get_avail_actions()
        
        # 存储数据
        if self.save_sim_data:
            self.dump_data()

        return local_obs, global_state, rewards, dones, infos, available_actions
    
    def _init_red_target_positions_old(self):
        """
        初始化红方智能体的目标航迹点
        1. 给每个智能体找一个距离最近的未侦察过的高价值区域或普通区域的格子中心
        """
        # 筛选有效的格子索引：未侦察过的高价值和普通格子
        valid_grids_mask = (self.comm_grids_mask | self.core_grids_mask) & ~self.grids_scout_mask
        
        # 获取有效格子中心位置，并计算智能体与这些中心的距离
        valid_grids_center = self.grid_centers[valid_grids_mask].reshape(-1, 2)
        distances_red2center = distance.cdist(self.red_positions, valid_grids_center, 'euclidean')
        
        # 选取距离每个智能体最近的格子中心作为目标位置
        nearest_grids_id = np.argmin(distances_red2center, axis=1)
        target_positions = valid_grids_center[nearest_grids_id]
        
        return target_positions
    
    def _init_red_target_positions(self):
        """
        初始化红方智能体的目标航迹点
        优先给每个智能体分配未侦察的高价值区域作为目标
        若无高价值区域则分配普通区域
        避免多个智能体目标冲突
        """
        # 筛选未侦察过的高价值区域和普通区域的格子
        high_value_mask = self.core_grids_mask & ~self.grids_scout_mask # 高价值格子
        normal_value_mask = self.comm_grids_mask & ~self.grids_scout_mask # 普通格子
        
        # 获取高价值和普通格子的中心位置
        high_value_centers = self.grid_centers[high_value_mask].reshape(-1, 2)
        normal_value_centers = self.grid_centers[normal_value_mask].reshape(-1, 2)
        
        # 构建高效的空间查询树
        high_value_tree = cKDTree(high_value_centers) if high_value_centers.size > 0 else None
        normal_value_tree = cKDTree(normal_value_centers) if normal_value_centers.size > 0 else None

        # 初始化目标位置数组
        target_positions = np.full((self.n_reds, 2), np.nan)
        assigned_targets = set()  # 用于记录已分配的目标，防止冲突
        
        # 先为每个智能体尝试分配最近的高价值格子
        for i, position in enumerate(self.red_positions):
            if high_value_tree:
                # 查询离当前智能体最近的高价值格子
                dist, nearest_idx = high_value_tree.query(position)
                nearest_target = tuple(high_value_centers[nearest_idx])

                # 如果目标未被分配，则使用该目标
                if nearest_target not in assigned_targets:
                    target_positions[i] = nearest_target
                    assigned_targets.add(nearest_target)
                else:
                    # 如果目标冲突，选择下一个最近的普通格子
                    if normal_value_tree:
                        dist, nearest_idx = normal_value_tree.query(position)
                        nearest_target = tuple(normal_value_centers[nearest_idx])
                        if nearest_target not in assigned_targets:
                            target_positions[i] = nearest_target
                            assigned_targets.add(nearest_target)

        # 对于未分配到高价值格子的智能体，再分配普通格子
        for i, position in enumerate(self.red_positions):
            if np.isnan(target_positions[i, 0]) and normal_value_tree:
                dist, nearest_idx = normal_value_tree.query(position)
                nearest_target = tuple(normal_value_centers[nearest_idx])
                
                if nearest_target not in assigned_targets:
                    target_positions[i] = nearest_target
                    assigned_targets.add(nearest_target)
        
        return target_positions
    
    def _correct_out_of_bounds_positions(self, out_of_bounds_indices, target_positions):
        """
        修正超出边界的智能体目标位置
        
        参数：
        - out_of_bounds_indices: 出界的智能体索引
        - target_positions: 智能体的目标位置
        
        返回：
        - target_positions: 修正后的目标位置
        """
        # 计算超出边界的智能体到每个边界线段的向量和投影点
        pos_vec = self.red_positions[out_of_bounds_indices, np.newaxis, :] - self.scout_bounds[:, 0, :]  # (n, 4, 2)
        pos_unitvec = pos_vec / self.scout_bounds_len[:, np.newaxis]
        t = np.clip(np.einsum('nij,ij->ni', pos_unitvec, self.scout_bounds_unitvec), 0.0, 1.0)    # 投影比例
        nearest = self.scout_bounds[:, 0, :] + t[:, :, np.newaxis] * self.scout_bounds_vec[np.newaxis, :, :]  # (n, 4, 2)
        
        # 计算每个智能体到边界最近点的距离，选择最近的投影点修正目标位置
        nearest_dist = np.linalg.norm(self.red_positions[out_of_bounds_indices, np.newaxis, :] - nearest, axis=2)
        nearest_id = np.argmin(nearest_dist, axis=1)
        target_positions[out_of_bounds_indices] = nearest[np.arange(out_of_bounds_indices.size), nearest_id]
        
        return target_positions
    
    def _correct_near_threat_positions(self, near_threat_indices, nearest_threat_zone_id, target_positions):
        """
        修正靠近威胁区的智能体目标位置
        
        参数：
        - near_threat_indices: 靠近威胁区的智能体索引
        - nearest_threat_zone_id: 最近的威胁区id
        - target_positions: 智能体的目标位置
        
        返回：
        - target_positions: 修正后的目标位置
        """
        # 获取最近的威胁区中心和半径
        nearest_threat_center = self.threat_ranges_center[nearest_threat_zone_id] # (n, 2)
        
        # 计算远离威胁区的期望方向 (智能体->威胁区相反方向)
        delta = self.red_positions[near_threat_indices] - nearest_threat_center
        unit_vec = delta / np.linalg.norm(delta, axis=1)[:, np.newaxis]

        # 计算偏移量，目标位置设为远离威胁中心的方向，并稍微远离威胁区
        offsets = unit_vec * 100
        target_positions[near_threat_indices] = self.red_positions[near_threat_indices] + offsets
        
        return target_positions
    
    def _get_near_threat_agent_indices(self):
        """
        获取接近威胁区域的智能体索引
        
        返回：
        - near_threat_indices: 靠近威胁区域的智能体索引
        - nearest_threat_id: 最近的威胁区域索引
        """  
        # 计算智能体当前位置到中心点的距离
        threat_dist = distance.cdist(self.red_positions, self.threat_ranges_center, 'euclidean')
        
        # 计算智能体当前位置到最近威胁区的中心点的索引，半径和距离
        nearest_threat_id = np.argmin(threat_dist, axis=-1)
        nearest_threat_radius = self.threat_ranges_radius[nearest_threat_id]
        nearest_dist = np.amin(threat_dist, axis=-1)
        
        # 选择接近威胁区的智能体
        # near_threat_agent = (nearest_dist - nearest_threat_radius) <= 100
        near_threat_agent = nearest_dist <= 100
        near_threat_indices = np.where(near_threat_agent)[0]
        
        return near_threat_indices, nearest_threat_id[near_threat_indices]
    
    def get_avail_heading_actions(self):
        """
        根据智能体是否在侦察区域以及是否接近威胁区域，调整其有效的航向动作列表。
        
        包含以下两种情况的处理：
        1. 如果智能体超出侦察区域，限制其航向动作，使其返回侦察区域。
        2. 如果智能体接近威胁区域，限制其航向动作，使其远离威胁区域。
        
        返回：
        available_actions: 每个智能体的有效航向动作布尔矩阵
        """
        # 初始化所有航向动作为可用
        available_actions = np.ones((self.n_reds, self.heading_action_num), dtype=bool)
        
        # 初始化目标点
        target_positions = self._init_red_target_positions() 
        
        # 判断那些智能体的位置超出侦察边界
        out_of_bounds = (
            (self.red_positions[:, 0] < -self.half_scout_size_x) | 
            (self.red_positions[:, 0] > self.half_scout_size_x) |
            (self.red_positions[:, 1] < -self.half_scout_size_y) | 
            (self.red_positions[:, 1] > self.half_scout_size_y)
        )
        
        # 如果有智能体超出边界，修正目标位置
        out_of_bounds_indices = np.where(out_of_bounds)[0]
        if out_of_bounds_indices.size > 0:
            target_positions = self._correct_out_of_bounds_positions(out_of_bounds_indices, target_positions)
        
        # 如果有智能体接近威胁区域，修正目标位置
        near_threat_indices, nearest_threat_zone_id = self._get_near_threat_agent_indices()
        if near_threat_indices.size > 0:
            target_positions = self._correct_near_threat_positions(near_threat_indices, nearest_threat_zone_id, target_positions)
        
        # 计算智能体的期望方向和角度差
        desired_directions = np.arctan2(target_positions[:, 1] - self.red_positions[:, 1],
                                        target_positions[:, 0] - self.red_positions[:, 0])
        angles_diff = (desired_directions - self.red_directions + np.pi) % (2 * np.pi) - np.pi

        # 计算动作限制限，限制智能体的左转和右转
        # 1.如果角度差大于阈值，限制只能选择右转动作（负号表示逆时针）
        mask_pos = angles_diff >= self.max_turn
        available_actions[mask_pos, :self.heading_action_mid_id + 1] = False
        # available_actions[mask_pos, :self.heading_action_mid_id + 2] = False
        

        # 2.如果角度差小于负的阈值，限制只能选择左转动作（正号表示顺时针）
        mask_neg = angles_diff <= -self.max_turn
        available_actions[mask_neg, self.heading_action_mid_id:] = False
        # available_actions[mask_neg, self.heading_action_mid_id-1:] = False
        
        # 对于被干扰的智能体，只能保持航向 (self.heading_action_mid_id 号动作可用)
        interfere_mask = self.red_interfere_damage_mask
        available_actions[interfere_mask] = False
        available_actions[interfere_mask, self.heading_action_mid_id] = True

        return available_actions
    
    def get_avail_acc_actions(self):
        """
        获取红方智能体的可用加速动作。

        返回:
        available_actions: 布尔数组，形状为 (n_reds, acc_action_num)，
                        表示每个红方智能体的各个加速动作是否可用。
        """
        # 初始化所有动作为不可用
        available_actions = np.ones((self.n_reds, self.acc_action_num), dtype=bool)

        available_actions[:, self.acc_action_mid_id] = True

        return available_actions
    
    def blue_explode(self):
        """
        判断蓝方智能体是否需要自爆，并执行相应的操作。
        自爆规则：
        1. 如果红方存活智能体数量超过70%，且蓝方智能体自爆范围内的红方智能体数量不少于3个，则自爆。
        2. 如果红方存活智能体数量在40%到70%之间，且蓝方智能体自爆范围内的红方智能体数量不少于2个，则自爆。
        3. 如果红方存活智能体数量少于40%，且蓝方智能体自爆范围内的红方智能体数量不少于1个，则自爆。
        """
        # 初始化蓝方自爆掩码，仅考虑存活且携带自爆载荷的智能体
        blue_explode_mask = self.blue_explode_mode_mask & self.blue_alives &~self.blue_interfere_damage_mask
        
        if not np.any(blue_explode_mask):
            return
        
        # 计算当前红方智能体的存活比例
        alive_percentage = np.sum(self.red_alives) / self.n_reds
        
        # 判断红方智能体是否在每个蓝方智能体自爆范围内
        red_in_explode_zone = (self.distances_blue2red < self.explode_radius) & self.red_alives
        
        # 计算每个蓝方自爆范围内的红方智能体数量
        red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)
        
        # 判断是否满足自爆条件并更新掩码
        if alive_percentage >= 0.7:
            blue_explode_mask &= (red_counts_in_zone >= 3)
        elif 0.4 <= alive_percentage < 0.7:
            blue_explode_mask &= (red_counts_in_zone >= 2)
        else:
            blue_explode_mask &= (red_counts_in_zone >= 1)

        # 更新蓝方自爆统计和状态
        self.blue_explode_mask = blue_explode_mask
        self.blue_explode_count = np.sum(blue_explode_mask)
        self.blue_explode_total += self.blue_explode_count
        self.blue_alives[blue_explode_mask] = False
        
        # 更新红方毁伤统计和状态
        self.red_explode_damage_mask = np.any(red_in_explode_zone[blue_explode_mask], axis=0)
        self.red_explode_damage_count = np.sum(self.red_explode_damage_mask)
        self.red_explode_damage_total += self.red_explode_damage_count
        self.red_alives[self.red_explode_damage_mask] = False
        
        # 存储自爆动作
        self.blue_action[blue_explode_mask, 2] = 1
        
    def blue_softkill(self):
        """
        判断蓝方智能体是否需要开启软杀伤（软杀伤智能体开启一次），并执行相应的操作。
        软杀伤规则：
        1. 如果红方存活智能体数量超过70%，且蓝方智能体软杀伤范围的红方智能体数量不少于3个，则开启软杀伤。
        2. 如果红方存活智能体数量在40%到70%之间，且蓝方智能体软杀伤范围内的红方智能体数量不少于2个，则开启软杀伤。
        3. 如果红方存活智能体数量少于40%，且蓝方智能体软杀伤范围内的红方智能体数量不少于1个，则开启软杀伤。
        """
        # 初始化蓝方软杀伤掩码，仅考虑存活且携带软杀伤载荷的智能体
        blue_softkill_mask = self.blue_softkill_mode_mask & (self.blue_softkill_time < self.softkill_time) & self.blue_alives &~self.blue_interfere_damage_mask
        
        if not np.any(blue_softkill_mask):
            return
        
        # 计算当前红方智能体的存活比例
        alive_percentage = np.sum(self.red_alives) / self.n_reds
        
        # 判断红方智能体是否在每个蓝方智能体软杀伤范围内
        red_in_softkill_zone = (self.distances_blue2red < self.softkill_radius) & self.red_alives
        
        # 计算每个蓝方软杀伤范围内的红方智能体数量
        red_counts_in_zone = np.sum(red_in_softkill_zone, axis=1)
        
        # 判断是否满足软杀伤条件并更新掩码
        if alive_percentage >= 0.7:
            blue_softkill_mask &= (red_counts_in_zone >= 3)
        elif 0.4 <= alive_percentage < 0.7:
            blue_softkill_mask &= (red_counts_in_zone >= 2)
        else:
            blue_softkill_mask &= (red_counts_in_zone >= 1)
            
        # 更新蓝方软毁伤统计和状态
        self.blue_softkill_mask = blue_softkill_mask
        self.blue_softkill_count = np.sum(blue_softkill_mask)
        self.blue_softkill_total += self.blue_softkill_count
        self.blue_softkill_time[blue_softkill_mask] += 1

        # 更新红方被软毁伤统计和状态
        random_prob = np.random.rand(self.n_reds)
        self.red_softkill_damage_mask = np.any(red_in_softkill_zone[blue_softkill_mask], axis=0) & (random_prob < self.softkill_prob) # 以60%的概率被软毁伤
        self.red_softkill_damage_count = np.sum(self.red_softkill_damage_mask)
        self.red_softkill_damage_total += self.red_softkill_damage_count
        self.red_alives[self.red_softkill_damage_mask] = False
        
        # 记录软杀伤的位置
        self.update_transformed_positions()
        mask = np.hstack((self.red_softkill_damage_mask, self.blue_softkill_mask))
        self.softkill_positions[mask] = self.transformed_positions[mask]
        
        # 存储软杀伤动作
        self.blue_action[blue_softkill_mask, 2] = 2
        
    def blue_interfere(self, pt):
        """
        判断蓝方智能体是否需要开启干扰（干扰智能体开启一次）
        规则如下：
        1. 如果视野内某个红方无人机与某个蓝方无人机的距离小于80m，且可以干扰到，则开启干扰。
        2. 如果视野内某个红方无人机与高价值目标的距离小于100m，且可以干扰到，则开启干扰。
        """
        # 初始化蓝方干扰掩码，仅考虑存活且携带干扰载荷的蓝方无人机
        blue_interfere_mask = (
            self.blue_interfere_mode_mask & 
            (self.blue_interfere_duration == 0) & 
            self.blue_alives & 
            ~self.blue_interfere_damage_mask
        )
        
        # 已经开了干扰但未结束的蓝方智能体
        blue_interfering_mask = (
            (self.blue_interfere_duration > 0) & 
            (self.blue_interfere_duration < self.interfere_duration) &
            self.blue_interfere_mode_mask & 
            self.blue_alives
        )
        
        # 如果没有蓝方无人机可以开启干扰或正在干扰，则直接返回
        if not np.any(blue_interfere_mask) and not np.any(blue_interfering_mask):
            return pt
        
        # 计算蓝方干扰范围内的红方智能体
        red_in_interfere_zone = (
            (self.distances_blue2red < self.can_interfere_radius) &
            (np.abs(self.angles_diff_blue2red) < self.can_interfere_angle / 2) &
            self.red_alives
        ) #(nB, nR)
        
        # 计算蓝方干扰范围内的蓝方智能体
        blue_in_interfere_zone = (
            (self.distances_blue2blue < self.can_interfere_radius) &
            (np.abs(self.angles_diff_blue2blue) < self.can_interfere_angle / 2) &
            self.blue_alives
        ) # (nB, nR)
        
        # 计算蓝方智能体与红方智能体的距离是否小于 80m
        closer_blue_red_mask = self.distances_blue2red < 80 # (nB, nR)
        
        # 计算红方智能体是否在某个高价值区域100m的范围内
        dists_to_core_zone = (
            np.linalg.norm(self.red_positions[:, np.newaxis, :] - self.core_ranges_center[np.newaxis, :, :], axis=-1)
            - self.core_ranges_radius[np.newaxis, :]
        ) # (nR, 4)
        red_near_core_mask = np.any(dists_to_core_zone < 100, axis=1) # (nR,)
        
        # 条件1：在干扰范围内且距离小于80m的红方智能体
        condition_1_mask = np.any(red_in_interfere_zone & closer_blue_red_mask, axis=1) # (nB, 1)
        
        # 条件2：在干扰范围内且在高价值区域附近的红方智能体
        condition_2_mask = np.any(red_in_interfere_zone & red_near_core_mask, axis=1)   # (nB, 1)
        
        
        # 更新蓝方干扰掩码
        blue_interfere_mask &= (condition_1_mask | condition_2_mask)
        
        # 如果没有蓝方无人机需要开启干扰，则提前返回
        if not np.any(blue_interfere_mask):
            return pt
        
        # 更新蓝方干扰统计和状态
        self.blue_interfere_mask = blue_interfere_mask
        self.blue_interfere_count = np.sum(blue_interfere_mask[self.blue_interfere_duration == 0])  # 首次开启
        self.blue_interfere_total += self.blue_interfere_count
        self.blue_interfere_duration[self.blue_interfere_mask] += 1
        
        # 更新红方被干扰统计和状态
        red_interfere_damage_mask = (
            (self.distances_blue2red < self.interfere_radius) & 
            (np.abs(self.angles_diff_blue2red) < self.interfere_angle / 2) &
            self.red_alives
        )
        self.red_interfere_damage_mask = np.any(red_interfere_damage_mask[blue_interfere_mask], axis=0)
        self.red_interfere_damage_count = np.sum(self.red_interfere_damage_mask)
        self.red_interfere_damage_total += self.red_interfere_damage_count
        
        # 计算干扰中的蓝方与最近的红方距离
        red_dists = np.where(red_in_interfere_zone, self.distances_blue2red, np.inf)    # 过滤掉不在干扰范围的敌机
        closest_red_indices = np.argmin(red_dists, axis=1)                              # 找到最近的敌机
        
        # 更新干扰中蓝方的方向
        mask = red_in_interfere_zone[np.arange(self.n_blues), closest_red_indices]
        pt[mask] = 0
        self.blue_directions[mask] = self.angles_blue2red[mask, closest_red_indices[mask]]
        
        # 计算干扰中的蓝方与最近的友方距离
        blue_dists = np.where(blue_in_interfere_zone, self.distances_blue2blue, np.inf) # 过滤掉不在干扰范围内的友机
        closest_blue_indices = np.argmin(blue_dists, axis=1)                            # 找到最近的友机
        
        # 更新干扰中蓝方的方向
        mask = blue_in_interfere_zone[np.arange(self.n_blues), closest_blue_indices]
        pt[mask] = 0
        self.blue_directions[mask] = self.angles_blue2blue[mask, closest_blue_indices[mask]]
           
        return pt
        
    def blue_collide(self):
        """
        判断并处理蓝方智能体的撞击行为：
        规则如下：
        1. 如果蓝方智能体的攻击范围（扇形区域）内存在红方智能体，则撞击最近的红方智能体。
        2. 撞击成功后，蓝方和目标红方智能体都被标记为死亡。
        """
        # 判断哪些蓝方智能体已经完成了干扰或软杀伤操作
        blue_done_interfere_mask = self.blue_interfere_duration == self.interfere_duration
        blue_done_softkill_mask = self.blue_softkill_time == self.softkill_time
        blue_collide_mask = (
            (blue_done_interfere_mask | blue_done_softkill_mask & self.blue_explode_mode_mask) & 
            self.blue_alives &
            ~self.blue_interfere_damage_mask
        )
        
        # 判断红方智能体是否在每个蓝方智能体的攻击范围内
        red_in_attack_zone = (
            (self.distances_blue2red < self.collide_radius) &
            (np.abs(self.angles_diff_blue2red) < self.collide_angle / 2) &
            self.red_alives &
            blue_collide_mask[:, np.newaxis]
        )

        if not np.any(red_in_attack_zone):
            return

        # 对不在攻击范围内的红方，距离设置为无限大
        distances_blue2red = self.distances_blue2red.copy()
        distances_blue2red[~red_in_attack_zone] = np.inf

        # 找个每个蓝方智能体最近的红方智能体
        nearest_red_id = np.argmin(distances_blue2red, axis=1)
        
        # 判断是否碰撞成功, 获取有效碰撞的 id
        success_mask = (~np.all(np.isinf(distances_blue2red), axis=1))

        success_agent_ids = np.where(success_mask)[0]
        success_target_ids = nearest_red_id[success_agent_ids]
        
        # 更新统计状态
        self.blue_alives[success_agent_ids] = False
        self.red_alives[success_target_ids] = False
        
        success_count = success_agent_ids.size
        
        self.blue_collide_count = success_count
        self.blue_collide_total += success_count
        
        self.red_collide_damage_count = success_count
        self.red_collide_damage_total += success_count
        
        self.blue_collide_agent_ids = success_agent_ids
        self.blue_collide_target_ids = success_target_ids

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
        target_core_centers = self.core_ranges_center[nearest_core_indices]

        # 计算期望方向
        desired_directions = np.arctan2(target_core_centers[:, 1] - self.blue_positions[:, 1],
                                        target_core_centers[:, 0] - self.blue_positions[:, 0])

        # 计算当前方向到期望方向的角度差，并将其归一化到[-pi,pi]区间
        angles_diff = (desired_directions - self.blue_directions + np.pi) % (2 * np.pi) - np.pi

        # 限制转向角度在最大角速度范围内
        angles_diff = np.clip(angles_diff, -self.max_turn, self.max_turn)

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
        # 判断蓝方智能体是否出界
        out_of_bounds = (
            (self.blue_positions[:, 0] < -self.half_scout_size_x) | 
            (self.blue_positions[:, 0] > self.half_scout_size_x) |
            (self.blue_positions[:, 1] < -self.half_scout_size_y) | 
            (self.blue_positions[:, 1] > self.half_scout_size_y)
        )
        
        # 找到出界的智能体索引
        out_of_bounds_indices = np.where(out_of_bounds)[0]

        # 如果没有出界，直接返回
        if out_of_bounds_indices.size == 0:
            return pt

        # 计算智能体到每条边界线段的投影点
        pos_vec = self.blue_positions[out_of_bounds_indices, np.newaxis, :] - self.scout_bounds[:, 0, :]  # (n, 4, 2)
        pos_unitvec = pos_vec / self.scout_bounds_len[:, np.newaxis]
        t = np.clip(np.einsum('nij,ij->ni', pos_unitvec, self.scout_bounds_unitvec), 0.0, 1.0)   # 计算投影比例 (n, 4)

        # 计算最近的边界点坐标
        nearest = self.scout_bounds[:, 0, :] + t[:, :, np.newaxis] * self.scout_bounds_vec[np.newaxis, :, :]  # (n, 4, 2)

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
        angles_diff = np.clip(angles_diff, -self.max_turn, self.max_turn)

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
        
        # 2.执行蓝方自爆的逻辑
        self.blue_explode()
        
        # 3.执行蓝方软毁伤的逻辑
        self.blue_softkill()
        
        # 4.执行蓝方干扰的逻辑
        pt = self.blue_interfere(pt)
        
        # 3.处理蓝方与红方的自爆行为
        self.blue_collide()
        
        # 4.处理蓝方防守高价值区域的行为
        pt = self.blue_guard(pt)
        
        # 5.处理蓝方智能体出界后的返回行为
        pt = self.blue_return(pt)
        
        # 6.更新蓝方智能体的方向和位置
        self._update_blue_position_and_direction(pt)
        
        # 8.更新距离矩阵和角度矩阵
        self.update_dist_and_angles()
        
    def _update_blue_position_and_direction(self, pt):
        """
        更新蓝方智能体的方向和位置
        仅更新存活的智能体
        """
        alive_mask = self.blue_alives
        
        # 更新方向，确保方向在 [-pi, pi] 范围内
        update_directions_mask = alive_mask & ~self.blue_interfere_damage_mask
        self.blue_directions[update_directions_mask] = (
            (self.blue_directions[update_directions_mask] + pt[update_directions_mask] * self.max_turn + np.pi) \
                % (2 * np.pi) - np.pi
        )
        
        # 更新位置
        dx = self.blue_velocities[alive_mask] * np.cos(self.blue_directions[alive_mask]) * self.dt_time
        dy = self.blue_velocities[alive_mask] * np.sin(self.blue_directions[alive_mask]) * self.dt_time
        self.blue_positions[alive_mask] += np.column_stack((dx, dy))
        
        # 存储数据
        self.blue_action[:, 1] = pt * self.max_turn
        
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
        core_percentage = self.scout_core_total / self.core_grids_num
        comm_percentage = self.scout_comm_total / self.comm_grids_num

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
        计算当前时间步的奖励
        """
        # 初始化奖励
        rewards = 0
        
        # 时间奖励
        time_reward = 1.0
        rewards += time_reward
        
        # 扫描普通区域的奖励
        scout_comm_reward = self.scout_comm_num * 10
        rewards += scout_comm_reward
        
        # 扫描高价值区域的奖励
        scout_core_reward = self.scout_core_num * 50
        rewards += scout_core_reward
        
        # 毁伤蓝方智能体奖励
        kill_num = (
            self.blue_explode_damage_count +
            self.blue_softkill_damage_count +
            self.blue_interfere_damage_count +
            self.blue_collide_damage_count
        )
        kill_reward = kill_num * 5
        rewards += kill_reward
        
        # 发动攻击奖励
        attack_num = (
            self.red_explode_count +
            self.red_softkill_count +
            self.red_interfere_count +
            self.red_collide_count
        )
        attack_reward = attack_num * 1
        rewards += attack_reward
        
        # 被毁伤惩罚
        damage_num = (
            self.red_explode_damage_count +
            self.red_softkill_damage_count +
            self.red_interfere_damage_count +
            self.red_collide_damage_count
        )
        damage_penalty = damage_num * -20
        rewards += damage_penalty
        
        win_reward = 3000 if win else 0
        rewards += win_reward
        
        return rewards 
    
    def get_reward_old(self, win=False):
        """
        计算当前时间步的整体奖励，结合多个因素，包括存活时间、扫描进度、毁伤敌方智能体等。
        """
        # 初始化每个智能体的奖励
        rewards = np.zeros(self.n_reds, dtype=float)

        # 计算各类奖励系数
        episode_progress = 1 + self._episode_steps / self.episode_limit
        dead_red_ratio = 1  + (1 - np.sum(self.red_alives) / self.n_reds)
        dead_blue_ratio = 1 + (1 - np.sum(self.blue_alives) / self.n_blues)
        scouted_core_ratio = 1 + self.scout_core_total / self.core_grids_num
        scouted_comm_ratio = 1 + self.scout_comm_total / self.comm_grids_num
        in_threat_progress = 1 + self.in_threat_zone_time / self.max_in_threat_time

        # 时间奖励：存活时间越久，奖励系数越大
        rewards += self.time_reward * self.red_alives * episode_progress
        
        # 扫描奖励：高价值区域和低价值区域的扫描奖励
        rewards += self.scout_core_reward * self.red_scout_core_mask * scouted_core_ratio
        rewards += self.scout_comm_reward * self.red_scout_comm_mask * scouted_comm_ratio

        # 毁伤蓝方智能体的奖励，毁伤的敌方智能体数量越多，奖励系数越大
        red_collide_mask = np.zeros(self.n_reds, dtype=bool)
        if self.red_collide_agent_ids.size > 0:
            red_collide_mask[self.red_collide_agent_ids] = True
        kill_blue_agents = (self.red_explode_mask | self.red_interfere_mask | self.red_softkill_mask | red_collide_mask)
        rewards += self.kill_reward * kill_blue_agents * dead_blue_ratio

        # 获胜奖励或失败惩罚
        rewards += self.reward_win if win else self.reward_defeat

        # 被毁伤惩罚：被毁伤的智能体越多，惩罚系数越大
        red_collide_damage_mask = np.zeros(self.n_reds, dtype=bool)
        if self.blue_collide_agent_ids.size > 0:
            red_collide_damage_mask[self.blue_collide_target_ids] = True
        kill_red_agents = (self.red_explode_damage_mask | self.red_interfere_damage_mask | self.red_softkill_damage_mask | red_collide_damage_mask)
        rewards += self.be_killed_penalty * kill_red_agents * dead_red_ratio

        # 在威胁区内的惩罚
        rewards += self.in_threat_penalty * self.red_threat_damage_mask * in_threat_progress

        # 重复侦查区域惩罚
        rewards += self.repeated_scouted_penalty * self.red_scout_repeat_mask

        # 在扫描区外的智能体，靠近侦查区域给奖励，远离给惩罚
        rewards += self.get_dist_reward()

        return rewards.reshape(-1, 1).tolist()
    
    def update_scout(self):
        """
        根据红方智能体的位置更新格子的侦察情况。
        """
        # 1. 对红方智能体的位置进行坐标转换，将坐标中心从 (0, 0) 移到 (-self.size_x / 2, self.size_y / 2)
        shifted_positions = (self.red_positions - np.array([-self.size_x / 2, self.size_y / 2])) * np.array([1, -1])
        
        # 2. 计算智能体所在的格子索引
        grid_indices_row = (shifted_positions[:, 1] // self.grid_size).astype(int)
        grid_indices_col = (shifted_positions[:, 0] // self.grid_size).astype(int)
        
        # 3. 判断智能体是否出界
        out_of_bound = (
            (grid_indices_row >= self.grids_row_num) |
            (grid_indices_row < 0) |
            (grid_indices_col >= self.grids_col_num) |
            (grid_indices_col < 0)
        )
        
        # 处理出界情况，将索引设置为无效值
        grid_indices_row[out_of_bound] = 0
        grid_indices_col[out_of_bound] = 0
        
        # 4. 判定那些智能体正在进行侦察
        valid_mask = ~out_of_bound & self.red_alives
        
        if not np.any(valid_mask):
            return
        
        # 5. 判断是否重复侦察
        already_scout_mask = self.grids_scout_mask[grid_indices_row, grid_indices_col]
        self.red_scout_repeat_mask = valid_mask & already_scout_mask
        
        # 6. 更新格子的扫描状态
        valid_scout_mask = valid_mask & ~already_scout_mask
        self.grids_scout_mask[grid_indices_row[valid_scout_mask], grid_indices_col[valid_scout_mask]] = True
        
        # 7. 更新观测到的格子信息
        self.grids_info[self.grids_scout_mask] = self.grids_type[self.grids_scout_mask]
        
        # 8. 更新低价值区域的扫描情况
        self.red_scout_comm_mask[valid_scout_mask] = self.comm_grids_mask[grid_indices_row[valid_scout_mask], grid_indices_col[valid_scout_mask]]
        self.scout_comm_total = np.sum(self.comm_grids_mask & self.grids_scout_mask)
        
        # 9. 更新高价值区域的扫描情况
        self.red_scout_core_mask[valid_scout_mask] = self.core_grids_mask[grid_indices_row[valid_scout_mask], grid_indices_col[valid_scout_mask]]
        self.scout_core_total = np.sum(self.core_grids_mask & self.grids_scout_mask)

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
        # 重置高价值区域
        self._reset_core_ranges()
        
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
        # 计算屏幕上的网格大小和左上角位置
        self.screen_grid_size = int(self.grid_size * self.scale_factor)
        self.screen_grid_left_tops = self.transform_positions(self.grid_left_tops)

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
        for i in range(self.grids_row_num):
            for j in range(self.grids_col_num):
                if not (self.out_grids_mask[i, j] or not self.grids_scout_mask[i, j]):
                    x, y = self.screen_grid_left_tops[i, j, :]
                    pygame.draw.rect(self.screen, (0, 0, 0), (x, y, self.screen_grid_size, self.screen_grid_size), 1)
    
    def _render_text(self):
        """
        渲染屏幕上的文本信息
        """
        n_alive_reds = np.sum(self.red_alives)
        n_alive_blues = np.sum(self.blue_alives)
        
        scout_core_ratio = self.scout_core_total / self.core_grids_num
        scout_comm_ratio = self.scout_comm_total / self.comm_grids_num
        
        black = (0, 0, 0)
        red = (255, 0, 0)
        blue = (0, 0, 255)
        
        # 渲染存活数量和时间步等文本信息
        time_text = self.font.render(
            f'Episode: {self._episode_count} Time Step: {self._episode_steps} Win count: {self.battles_won}', 
            True, 
            black
        )
        
        red_alives_text = self.font.render(
            f'Red alives: [{n_alive_reds} / {np.sum(self.red_explode_mode_mask & self.red_alives)} / {np.sum(self.red_softkill_mode_mask & self.red_alives)} / {np.sum(self.red_interfere_mode_mask & self.red_alives)}]',
            True,
            red
        )
        
        red_explode_text = self.font.render(
            f'Red explode: [{self.red_explode_count} / {self.red_explode_total}] ---- [{self.red_explode_damage_count} / {self.red_explode_damage_total}]',
            True,
            red
        )
        
        red_softkill_text = self.font.render(
            f'Red softkill: [{self.red_softkill_count} / {self.red_softkill_total}] ---- [{self.red_softkill_damage_count} / {self.red_softkill_damage_total}]',
            True,
            red
        )
        
        red_interfere_text = self.font.render(
            f'Red interfere: [{self.red_interfere_count} / {self.red_interfere_total}] ---- [{self.red_interfere_damage_count} / {self.red_interfere_damage_total}]',
            True,
            red
        )
        
        red_collide_text = self.font.render(
            f'Red collide: [{self.red_collide_count} / {self.red_collide_total}] ---- [{self.red_collide_damage_count} / {self.red_collide_damage_total}]',
            True,
            red
        )
        
        red_threat_text = self.font.render(
            f'Red Dead by Threat: [{self.red_threat_damage_count} / {self.red_threat_damage_total}]',
            True,
            red
        )
        
        blue_alives_text = self.font.render(
            f'Blue alives: [{n_alive_blues} / {np.sum(self.blue_explode_mode_mask & self.blue_alives)} / {np.sum(self.blue_softkill_mode_mask & self.blue_alives)} / {np.sum(self.blue_interfere_mode_mask & self.blue_alives)}]',
            True,
            blue
        )
        
        blue_explode_text = self.font.render(
            f'Blue explode: [{self.blue_explode_count} / {self.blue_explode_total}] ---- [{self.blue_explode_damage_count} / {self.blue_explode_damage_total}]',
            True,
            blue
        )
        
        blue_softkill_text = self.font.render(
            f'Blue softkill: [{self.blue_softkill_count} / {self.blue_softkill_total}] ---- [{self.blue_softkill_damage_count} / {self.blue_softkill_damage_total}]',
            True,
            blue
        )
        
        blue_interfere_text = self.font.render(
            f'Blue interfere: [{self.blue_interfere_count} / {self.blue_interfere_total}] ---- [{self.blue_interfere_damage_count} / {self.blue_interfere_damage_total}]',
            True,
            blue
        )
        
        blue_collide_text = self.font.render(
            f'Blue collide: [{self.blue_collide_count} / {self.blue_collide_total}] ---- [{self.blue_collide_damage_count} / {self.blue_collide_damage_total}]',
            True,
            blue
        )
        
        scout_text = self.font.render(
            f'Scout Core: {round(scout_core_ratio, 2)} Scout Comm: {round(scout_comm_ratio, 2)}', 
            True, 
            red
        )
        
        self.screen.blit(time_text, (10, 10))
        self.screen.blit(red_alives_text, (10, 50))
        self.screen.blit(red_explode_text, (10, 90))
        self.screen.blit(red_softkill_text, (10, 130))
        self.screen.blit(red_interfere_text, (10, 170))
        self.screen.blit(red_collide_text, (10, 210))
        self.screen.blit(blue_alives_text, (10, 250))
        self.screen.blit(blue_explode_text, (10, 290))
        self.screen.blit(blue_softkill_text, (10, 330))
        self.screen.blit(blue_interfere_text, (10, 370))
        self.screen.blit(blue_collide_text, (10, 410)) 
        self.screen.blit(red_threat_text, (10, 450))
        self.screen.blit(scout_text, (10, 490))
    
class Arg(object):
    def __init__(self) -> None:
        self.map_name = '100_vs_100'
        self.scenario_name = 'scout'
        self.episode_length = 600
        self.use_script = True
        self.save_sim_data = True
        self.debug = True
        self.plane_name = "plane_scout"

if __name__ == "__main__":
    args = Arg()

    env = ScoutEnv(args)

    env.reset()
    
    for i in range(1):
        local_obs, global_state, available_actions = env.reset()
        
        import time
        done = True
        while done:
            start = time.time()
            actions = env.red_random_policy(available_actions)
            local_obs, global_state, rewards, dones, infos, available_actions = env.step(actions)

            done = ~np.all(dones)

            env.render()

        if env.red_explode_total == 0:
            print(f'Episode: {i}')
            # print(f'[frame: {i}]---[Time: {time.time() - start}]')
    
    env.close()