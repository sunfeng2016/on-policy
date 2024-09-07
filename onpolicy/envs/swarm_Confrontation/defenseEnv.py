# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-17
# @Description: Implementation of Agent Class

import os
import pygame
import numpy as np

try:
    from onpolicy.envs.swarm_Confrontation.baseEnv import BaseEnv, EXPLODE_MODE, INTERFERE_MODE, SOFTKILL_MODE
except:
    from baseEnv import BaseEnv, EXPLODE_MODE, INTERFERE_MODE, SOFTKILL_MODE
    
os.environ["SDL_VIDEODRIVER"] = "dummy"

class DefenseEnv(BaseEnv):
    def __init__(self, args):
        super(DefenseEnv, self).__init__(args)

        # 定义红方核心区域和基地
        self.red_core = self._create_circle([2250.0, 0.0], 25.0)
        self.red_base = self._create_circle([2250.0, 0.0], 1250.0)
        
        self.red_base_center = self.red_base['center']
        self.red_base_radius = self.red_base['radius']

        # 定义红方基地的防线
        self.red_lines = np.array([
            [[1366.0,  884.0], [1750.0,  500.0]],
            [[1750.0,  500.0], [1750.0, -500.0]],
            [[1750.0, -500.0], [1366.0, -884.0]],
            [[3134.0,  884.0], [2750.0,  500.0]],
            [[2750.0,  500.0], [2750.0, -500.0]],
            [[2750.0, -500.0], [3134.0, -884.0]],
        ])
        self._calculate_red_lines_properties()
        
        # 定义红方的目标区域
        self.red_target_area = [1750, 2750, -1000, 1000]

        # 定义红方的威胁区域
        self.red_square_size = 1000.0 / 2
        self._define_threat_zones()
        
        # 定义蓝方基地
        self.blue_bases = self._create_blue_bases()
        
        # 最大威胁区域停留时间
        self.max_in_threat_zone_time = 10
        
        # 蓝方不同批次智能体出发的时间间隔
        self.interval = 5
        
        # 基地的最大攻击次数
        self.max_attack_core_num = 40
        
    def _calculate_red_lines_properties(self):
        """
        计算红方防线的向量、长度和单位向量
        """
        self.red_lines_vec = self.red_lines[:, 1, :] - self.red_lines[:, 0, :]
        self.red_lines_len = np.linalg.norm(self.red_lines_vec, axis=1)
        self.red_lines_unitvec = self.red_lines_vec / self.red_lines_len[:, np.newaxis]
        
    def _define_threat_zones(self):
        """
        定义红方威胁区域的相关属性。
        """
        # 左侧威胁区域的两个扇形边界
        self.left_sector_pos1 = np.array([1366.0, 884.0])
        self.left_sector_pos2 = np.array([1366.0, -884.0])
        
        # 左侧威胁区域的两个扇形边界对应的弧度
        self.left_sector_theta1, self.left_sector_theta2 = calculate_sector_theta(
            self.left_sector_pos1, self.left_sector_pos2, self.red_base['center']
        )
        
        # 左侧威胁区域的右边界
        self.left_threat_x = self.red_base['center'][0] - self.red_square_size

        # 右侧威胁区域的两个扇形边界
        self.right_sector_pos1 = np.array([3134.0, -884.0])
        self.right_sector_pos2 = np.array([3134.0, 884.0])
        
        # 右侧威胁区域的两个扇形边界对应的弧度
        self.right_sector_theta1, self.right_sector_theta2 = calculate_sector_theta(
            self.right_sector_pos1, self.right_sector_pos2, self.red_base['center']
        )
        
        # 右侧威胁区域的左边界
        self.right_threat_x = self.red_base['center'][0] + self.red_square_size
    
    def _create_circle(self, center, radius):
        """
        创建一个圆形区域的字典结构。
        """
        return {'center': np.array(center), 'radius': radius}
        
    def _create_blue_bases(self):
        """
        创建蓝方基地的字典列表。
        """
        # return [
        #     self._create_circle([1500.0,  1500.0], 500.0),  # 上右
        #     self._create_circle([1500.0, -1500.0], 500.0),  # 下右
        #     self._create_circle([ 500.0,  1500.0], 500.0),  # 上左
        #     self._create_circle([ 500.0, -1500.0], 500.0),  # 下左
        # ]
        
        return [
            self._create_circle([1600.0,  1750.0], 600.0),  # 上右
            self._create_circle([1600.0, -1750.0], 600.0),  # 下右
            self._create_circle([ 400.0,  1750.0], 600.0),  # 上左
            self._create_circle([ 400.0, -1750.0], 600.0),  # 下左
        ]
        
    def distribute_red_agents(self):
        """
        分配红方智能体到基地内外的数量。
        
        返回:
        n_in_bases: 分配到基地内部的红方智能体数量。
        n_out_bases: 分配到基地外部的红方智能体数量。
        """
        n_out_bases = int(self.n_reds * np.random.uniform(0.1, 0.2))
        n_in_bases = self.n_reds - n_out_bases
        return n_in_bases, n_out_bases
    
    def generate_red_positions(self):
        """
        生成红方智能体的位置和朝向。
        
        返回:
        positions: 红方智能体的位置数组，形状为 (n_reds, 2)。
        directions: 红方智能体的朝向数组，形状为 (n_reds,)。
        """
        # 分配基地内外的智能体数量
        n_in_bases, n_out_bases = self.distribute_red_agents()
        
        # 基地内部智能体的位置
        in_base_positions = self._generate_positions_in_circle(
            n_in_bases, self.red_base_center, self.red_base_radius
        )

        # 基地外部智能体的位置
        size = np.array([self.size_x, self.size_y]) * 0.5
        out_base_positions = (np.random.rand(n_out_bases, 2) - 0.5) * size

        # 合并所有智能体的位置
        positions = np.empty((self.n_reds, 2))
        positions[:n_in_bases] = in_base_positions
        positions[n_in_bases:] = out_base_positions
        
        # 生成朝向    
        directions = np.random.uniform(-np.pi, np.pi, self.n_reds)

        return positions, directions

    def distribute_blue_agents(self):
        """
        随机将蓝方智能体分配到不同的基地。
        
        返回:
        group_sizes: 每个基地中蓝方智能体的数量数组。
        """
        n_groups = len(self.blue_bases)
        group_sizes = np.random.multinomial(self.n_blues - self.blue_softkill_mode_num, np.ones(n_groups) / n_groups)

        return group_sizes

    def generate_blue_positions(self):
        """
        生成蓝方智能体的位置和朝向。
        
        返回:
        positions: 蓝方智能体的位置数组，形状为 (n_blues, 2)。
        directions: 蓝方智能体的朝向数组，形状为 (n_blues,)。
        """
        # 初始化位置矩阵
        positions = np.empty((self.n_blues, 2))
        
        # 首先初始化软杀伤载荷的智能体的位置（围绕红方基地外围分布）
        softkill_positions = self._generate_positions_on_circle(
            self.blue_softkill_mode_num, 
            center=self.red_base_center, 
            radius=self.red_base_radius + 20
        )
        
        # 分配其余智能体到各个蓝方基地
        group_sizes = self.distribute_blue_agents()
        
        # 在各个圆形区域内生成位置
        other_positions = np.vstack([
            self._generate_positions_in_circle(size, center=base['center'], radius=base['radius'])
            for size, base in zip(group_sizes, self.blue_bases)
        ])
        
        # 将生成的位置分配到不同的智能体组
        positions[self.blue_softkill_mode_mask] = softkill_positions
        positions[~self.blue_softkill_mode_mask] = other_positions
        
        # 计算智能体的朝向  
        directions = self._calculate_agent_directions(positions, self.red_base_center)
        
        return positions, directions
    
    def init_positions(self):
        """
        初始化红蓝双方智能体的位置和朝向，并设置速度。
        
        返回:
        positions: 所有智能体的位置数组，形状为 (n_reds + n_blues, 2)。
        directions: 所有智能体的朝向数组，形状为 (n_reds + n_blues,)。
        velocities: 所有智能体的速度数组，形状为 (n_reds + n_blues,)。
        """
        
        red_positions, red_directions = self.generate_red_positions()
        blue_positions, blue_directions = self.generate_blue_positions()
        
        red_velocities = np.full(self.n_reds, self.red_max_vel)
        blue_velocities = np.full(self.n_blues, self.blue_max_vel)

        return red_positions, red_directions, red_velocities, blue_positions, blue_directions, blue_velocities
    
    def _generate_positions_in_circle(self, n_agents, center, radius):
        """
        在一个圆形区域内生成随机位置。

        参数:
        n_agents: 生成的位置数量。
        center: 圆心坐标 (x, y)。
        radius: 圆的半径。

        返回:
        positions: 生成的位置数组，形状为 (n_agents, 2)。
        """
        angles = np.random.uniform(0, 2 * np.pi, n_agents)
        radii = radius * np.sqrt(np.random.uniform(0, 1, n_agents))
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        return np.vstack([x, y]).T

    def _generate_positions_on_circle(self, n_agents, center, radius):
        """
        在一个圆形的边缘生成随机位置。

        参数:
        n_agents: 生成的位置数量。
        center: 圆心坐标 (x, y)。
        radius: 圆的半径。

        返回:
        positions: 生成的位置数组，形状为 (n_agents, 2)。
        """
        angles = np.random.uniform(0, 2 * np.pi, n_agents)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        return np.vstack([x, y]).T

    def _calculate_agent_directions(self, positions, target_center):
        """
        根据位置计算智能体的朝向。

        参数:
        positions: 智能体的位置数组，形状为 (n_agents, 2)。
        target_center: 目标中心点的坐标 (x, y)。

        返回:
        directions: 计算后的朝向数组，形状为 (n_agents,)。
        """
        directions = np.arctan2(target_center[1] - positions[:, 1], target_center[0] - positions[:, 0])
        directions += np.random.uniform(-np.pi/18, np.pi/18, positions.shape[0])
        return directions
    
    def reset(self):
        """
        重置环境至初始状态，并初始化相关变量。
        
        返回：
        local_obs: 各个红方智能体的局部观测值。
        global_state: 每个红方智能体的全局状态。
        available_actions: 每个红方智能体的可用动作集。
        """
        # 调用父类的 reset 方法以重置基础环境
        super().reset()

        # 初始化威胁区域内的停留时间数组（针对每个蓝方智能体）
        self.in_threat_zone_times = np.zeros(self.n_blues)
        
        # 初始化蓝方智能体的批次掩码
        self._init_blue_mask()

        # 获取局部观测值、全局状态和可用动作
        local_obs = self.get_obs()
        global_state = [self.get_state()] * self.n_reds
        available_actions = self.get_avail_actions()

        return local_obs, global_state, available_actions

    def _init_blue_mask(self):
        """
        将蓝方智能体分组四个进攻批次
        
        - 第一批次：携带软杀伤载荷的智能体
        - 第二批次：50%的携带自爆载荷的智能体
        - 第三批次：50%的携带自爆载荷的智能体
        - 第四批次：干扰的智能体
        """
        self.blue_first_mask = self.blue_softkill_mode_mask
        self.blue_second_mask, self.blue_third_mask = self._split_explode_agents(self.blue_explode_mode_mask, 0.5)
        self.blue_fourth_mask = self.blue_interfere_mode_mask
        
        # 已经激活的蓝方智能体掩码
        self.blue_active_mask = np.zeros(self.n_blues, dtype=bool)
    
    def _split_explode_agents(self, explode_mask, split_ratio):
        """
        将携带自爆载荷的智能体分成两个批次。
        
        参数:
        - explode_mask: 布尔数组，表示携带自爆载荷的智能体。
        - split_ratio: 第一个批次占总携带自爆载荷智能体的比例。
        
        返回:
        - first_batch_mask: 布尔数组，表示第一个批次的智能体。
        - second_batch_mask: 布尔数组，表示第二个批次的智能体。
        """
        
        explode_indices = np.where(explode_mask)[0]
        np.random.shuffle(explode_indices)
        
        group_size = int(len(explode_indices) * split_ratio)
        
        first_batch_indices = explode_indices[:group_size]
        second_batch_indices = explode_indices[group_size:]
        
        first_batch_mask = np.zeros(len(explode_mask), dtype=bool)
        second_batch_mask = np.zeros(len(explode_mask), dtype=bool)
        
        first_batch_mask[first_batch_indices] = True
        second_batch_mask[second_batch_indices] = True
        
        return first_batch_mask, second_batch_mask
        
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
        # 重置动作掩码和数量
        self._reset_action_mask_and_counter()
        
        # 执行红方的动作
        self.red_step(actions)
        
        # 执行蓝方的动作
        self.blue_step()

        # 更新步数计数器
        self._total_steps += 1
        self._episode_steps += 1

        # 检查是否终止以及是否胜利
        terminated, win, res = self.get_result()
        bad_transition = self._update_result(terminated, win)
        
        # 汇总环境信息
        info = self._collect_info(bad_transition, res, terminated)

        # 获取局部观测值、全局状态、奖励和可用动作
        local_obs = self.get_obs()
        global_state = [self.get_state()] * self.n_reds
        reward = self.get_reward()
        rewards = [[reward]] * self.n_reds
        dones = np.where(terminated, True, ~self.red_alives)
        infos = [info] * self.n_reds
        available_actions = self.get_avail_actions()
    
        # 存储数据
        if self.save_sim_data:
            self.dump_data()

        return local_obs, global_state, rewards, dones, infos, available_actions
    
    def _collect_info(self, bad_transition, res, dones):
        """
        汇总环境的各种信息。

        参数:
        bad_transition: 是否为不良转移。
        res: 环境的其他结果信息。
        dones: 当前回合是否结束。

        返回:
        info: 包含环境信息的字典。
        """
        if not dones:
            return {
                "battles_won": self.battles_won,
                "battles_game": self.battles_game,
                "battles_draw": self.timeouts,
                'bad_transition': bad_transition,
                'won': self.win_counted,
            }
        else:
            red_kill_total = (
                self.blue_explode_damage_total +
                self.blue_softkill_damage_total +
                self.blue_collide_damage_total + 
                self.blue_threat_damage_total
            )
            
            red_damage_total = (
                self.red_explode_damage_total +
                self.red_softkill_damage_total +
                self.red_collide_damage_total + 
                self.red_threat_damage_total
            )
        
            return {
                "battles_won": self.battles_won,
                "battles_game": self.battles_game,
                "battles_draw": self.timeouts,
                'bad_transition': bad_transition,
                'episode_length': self._episode_steps,
                'n_red_alives': np.sum(self.red_alives),
                'n_blue_alives': np.sum(self.blue_alives),
                'red_explode_total': self.red_explode_total,
                'red_explode_damage_total': self.red_explode_damage_total,
                'red_softkill_total': self.red_softkill_total,
                'red_softkill_damage_total': self.red_softkill_damage_total,
                'red_interfere_total': self.red_interfere_total,
                'red_interfere_damage_total': self.red_interfere_damage_total,
                'red_collide_total': self.red_collide_total,
                'red_collide_damage_total': self.red_collide_damage_total,
                'red_threat_damage_total': self.red_threat_damage_total,
                'blue_explode_total': self.blue_explode_total,
                'blue_explode_damage_total': self.blue_explode_damage_total,
                'blue_softkill_total': self.blue_softkill_total,
                'blue_softkill_damage_total': self.blue_softkill_damage_total,
                'blue_interfere_total': self.blue_interfere_total,
                'blue_interfere_damage_total': self.blue_interfere_damage_total,
                'blue_collide_total': self.blue_collide_total,
                'blue_collide_damage_total': self.blue_collide_damage_total,
                'blue_threat_damage_total': self.blue_threat_damage_total,
                'red_kill_total': red_kill_total,
                'red_damage_total': red_damage_total,
                'attack_core_total': self.attack_core_total,
                'scout_core_ratio': 0,  # 高价值区域被侦察的比例
                'scout_comm_ratio': 0,  # 普通区域被侦察的比例
                'won': self.win_counted,
                "other": res
            }
        
    def flee_explode_zone(self, target_positions):
        """
        判断蓝方智能体是否在距离最近的红方智能体的自爆范围内，
        如果该红方智能体的自爆范围内还有其它蓝方智能体，
        那么该蓝方智能体需要逃离该红方智能体的自爆范围。

        参数：
        target_positions: 蓝方智能体的目标位置数组，形状为 (n_blues, 2)

        返回:
        taget_positions: 蓝方智能体的目标位置数组。
        """
        # 计算每个红方智能体自爆范围附近的蓝方智能体数量
        blue_num_in_explode_zone = np.sum(self.distances_red2blue < self.can_explode_radius, axis=1)

        # 判断哪些红方智能体有自爆倾向（即至少有一个蓝方智能体在其自爆范围内）
        red_will_explode = blue_num_in_explode_zone > 1

        # 如果没有智能体具有自爆倾向，则提前返回
        if not any(red_will_explode):
            return target_positions

        # 找到每个蓝方智能体距离最近的红方智能体的索引
        nearest_id = np.argmin(self.distances_red2blue, axis=0)

        # 判断蓝方智能体是否在最近的红方智能体的自爆范围内
        is_in_explode = self.distances_red2blue[nearest_id, np.arange(self.n_blues)] < self.explode_radius

        # 如果没有智能体在自爆范围内，则提前返回
        if not any(is_in_explode):
            return target_positions

        # 如果在最近的红方智能体的自爆范围内，且该红方智能体有自爆倾向，则该蓝方智能体需要逃离
        flee_or_not = is_in_explode & red_will_explode[nearest_id]

        # 计算逃离方向，方向为蓝方位置减去红方位置的向量
        flee_directions = self.blue_positions - self.red_positions[nearest_id, :]

        # 计算逃离角度
        flee_angles = np.arctan2(flee_directions[:, 1], flee_directions[:, 0])

        # 计算逃离位置偏移，沿逃离方向移动到自爆范围的边界
        offsets = np.stack([np.cos(flee_angles), np.sin(flee_angles)], axis=1) * self.explode_radius

        # 计算新的目标位置，将蓝方智能体的位置从红方自爆范围的边缘偏移到新的位置
        targets = self.blue_positions + offsets

        # 更新需要逃离的蓝方智能体的目标位置
        target_positions[flee_or_not] = targets[flee_or_not]

        return target_positions

    def flee_threat_zone(self, is_in_threat, target_positions):
        """
        针对当前已经在警戒区的智能体，选择最近的边界上的最近点作为目标点，从而逃离警戒区。

        参数：
        is_in_threat: 布尔数组，表示哪些蓝方智能体在警戒区内。
        target_positions: 蓝方智能体的目标位置数组，形状为 (n_blues, 2)。

        返回：
        target_positions: 更新后蓝方智能体目标位置数组。
        """

        # 如果没有智能体在威胁区域内，提前返回
        if not np.any(is_in_threat):
            return target_positions

        # 计算智能体当前位置到红方防线起点的向量，形状为 (n_blues, n_lines, 2)
        pos_vec = self.blue_positions[:, np.newaxis, :] - self.red_lines[:, 0, :]

        # 计算每个智能体位置在每条线段上的投影长度 t
        t = np.einsum('nij,ij->ni', pos_vec, self.red_lines_unitvec) / self.red_lines_len

        # 将 t 限制在 [0, 1] 范围内，确保投影点在线段上
        t = np.clip(t, 0.0, 1.0)

        # 计算线段上距离智能体最近的点的坐标，形状为 (n_blues, n_lines, 2)
        nearest = self.red_lines[:, 0, :] + t[:, :, np.newaxis] * self.red_lines_vec[np.newaxis, :, :]

        # 计算智能体当前位置到最近点的距离，形状为 (n_blues, n_lines)
        distance = np.linalg.norm(self.blue_positions[:, np.newaxis, :] - nearest, axis=2)

        # 找到每个智能体距离最近的线段的索引，形状为 (n_blues,)
        nearest_id = np.argmin(distance, axis=1)

        # 获取每个智能体最近的目标点，形状为 (n_blues, 2)
        nearest_target = nearest[np.arange(self.n_blues), nearest_id]

        # 更新在警戒区内的智能体的目标位置
        target_positions[is_in_threat] = nearest_target[is_in_threat]

        return target_positions
    
    def around_threat_zone(self, will_in_threat, target_positions):
        """
        给即将进入警戒区的蓝方智能体分配一个新的目标点，使他们绕开警戒区。
        
        参数:
        will_in_threat: 布尔数组，表示哪些蓝方智能体即将进入警戒区。
        target_positions: 蓝方智能体的目标位置数组，形状为 (n_blues, 2)。
        
        返回:
        target_positions: 更新后的蓝方智能体目标位置数组。
        """
        # 如果没有智能体即将进入威胁区，则提前返回
        if not any(will_in_threat):
            return target_positions

        # 生成一个随机角度，默认从北侧突防通道口进入
        target_angles = np.random.uniform(self.right_sector_theta2, self.left_sector_theta1, size=self.n_blues)

        # 获取蓝方智能体的y坐标
        positions_y = self.blue_positions[:, 1]

        # 如果智能体在南侧，反转角度，从南侧突防通道口进入
        target_angles = np.where(positions_y > 0, target_angles, -target_angles)

        # 计算目标位置的偏移量, 基于目标角度和红方基地的半径
        dx = np.cos(target_angles)
        dy = np.sin(target_angles)
        offsets = np.stack([dx, dy], axis=1) * self.red_base_radius

        # 计算新的目标位置，目标位置为红方基地中心点加上偏移量
        new_targets = self.red_base_center + offsets

        # 更新威胁区域外智能体的目标位置
        target_positions[will_in_threat] = new_targets[will_in_threat]

        return target_positions
    
    def is_hit_core_zone(self):
        """
        判断蓝方智能体是否在红方高价值区域内，并更新打击次数和蓝方智能体的存活状态。
        """
        # 计算每个蓝方智能体到红方高价值区域中心的距离
        dists_to_center = np.linalg.norm(self.blue_positions - self.red_core['center'], axis=1)

        # 判断自爆载荷的蓝方智能体是否在红方高价值区域内
        in_red_core = (dists_to_center < self.red_core['radius']) & self.blue_explode_mode_mask

        # 计算并更新在红方高价值区域内被打击的蓝方智能体数量
        self.attack_core_num = np.sum(in_red_core & self.blue_alives)
        self.attack_core_total += self.attack_core_num

        # 将在高价值区域内的蓝方智能体标记为死亡
        self.blue_alives[in_red_core] = False
        
        # 更新激活掩码
        self.blue_active_mask &= self.blue_alives

    def is_in_threat_zone(self):
        """
        判断蓝方智能体是否在红方基地的威胁区域内或即将进入威胁区域，
        并根据蓝方智能体在威胁区域内的时间进行软杀伤。
        
        返回:
        in_threat_zone: 布尔数组，表示哪些蓝方智能体当前在威胁区域内。
        will_in_threat_zone: 布尔数组，表示哪些蓝方智能体即将进入威胁区域。
        """

        # 1. 判断蓝方智能体是否在红方圆形基地内
        dists_to_center = np.linalg.norm(self.blue_positions - self.red_base_center, axis=1)
        in_red_base = (dists_to_center < self.red_base_radius) & self.blue_alives

        # 2. 判断蓝方智能体是否在 x 轴的威胁区域内
        x_positions = self.blue_positions[:, 0]
        x_in_left = x_positions < self.left_threat_x
        x_in_right = x_positions > self.right_threat_x

        # 3. 判断蓝方智能体是否在两个扇形区域内
        vectors_to_center = self.blue_positions - self.red_base_center
        angles = np.arctan2(vectors_to_center[:, 1], vectors_to_center[:, 0])
        angles = np.mod(angles + 2 * np.pi, 2 * np.pi)  # 将角度范围限制在 [0, 2π]

        # 左边扇形区域角度判断
        left_sector_angle_range = np.logical_or(
            (self.left_sector_theta1 <= self.left_sector_theta2) & 
            (angles > self.left_sector_theta1) & 
            (angles < self.left_sector_theta2),
            (self.left_sector_theta1 > self.left_sector_theta2) & 
            ((angles > self.left_sector_theta1) | 
            (angles < self.left_sector_theta2))
        )

        # 右边扇形区域角度判断
        right_sector_angle_range = np.logical_or(
            (self.right_sector_theta1 <= self.right_sector_theta2) & 
            (angles > self.right_sector_theta1) & 
            (angles < self.right_sector_theta2),
            (self.right_sector_theta1 > self.right_sector_theta2) & 
            ((angles > self.right_sector_theta1) | 
            (angles < self.right_sector_theta2))
        )

        # 当前在威胁区域内的蓝方智能体
        in_threat_zone = (
            (left_sector_angle_range & x_in_left & in_red_base) | 
            (right_sector_angle_range & x_in_right & in_red_base)
        )

        # 判断即将进入威胁区域的蓝方智能体
        will_in_threat_zone = (
            (left_sector_angle_range & x_in_left & ~in_red_base) | 
            (right_sector_angle_range & x_in_right & ~in_red_base)
        )

        # 更新在威胁区域内的时间
        self.in_threat_zone_times[in_threat_zone] += 1
        self.in_threat_zone_times[~in_threat_zone] = 0

        # 毁伤掩码：在威胁区域内停留超过最大时间的智能体将被标记为死亡
        kill_mask = self.in_threat_zone_times >= self.max_in_threat_zone_time
        self.blue_alives[kill_mask] = False
        
        self.blue_threat_damage_mask = kill_mask
        self.blue_threat_damage_count = np.sum(kill_mask)
        self.blue_threat_damage_total += self.blue_threat_damage_count
        
        # 更新激活掩码
        self.blue_active_mask &= self.blue_alives

        return in_threat_zone, will_in_threat_zone

    def blue_explode(self):
        """
        判断蓝方智能体是否需要自爆:
        规则：
        1. 如果蓝方存活智能体的数量超过80%且自爆范围内的红方智能体数量超过1,则自爆。
        2. 如果蓝方存活智能体的数量在60%和80%之间且自爆范围内红方智能体的数量超过2,则自爆。
        3. 其它情况下不攻击。
        """
        # 初始化蓝方自爆掩码，仅考虑已激活且携带自爆载荷的智能体
        blue_explode_mask = self.blue_explode_mode_mask & self.blue_active_mask &~self.blue_interfere_damage_mask
        
        if not np.any(blue_explode_mask):
            return
        
        # 计算当前蓝方存活智能体的数量及其占比
        alive_count = np.sum(self.blue_alives)
        alive_percentage = alive_count / self.n_blues
        
        # 判断红方智能体是否在每个蓝方自爆范围内
        red_in_explode_zone = (self.distances_blue2red < self.explode_radius) & self.red_alives
        
        # 计算每个蓝方自爆范围内的红方智能体数量
        red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)

        # 判断是否满足自爆条件并更新掩码
        if alive_percentage >= 0.8:
            # 规则1：存活比例超过80%且自爆范围内红方数量不少于1
            blue_explode_mask &= (red_counts_in_zone >= 1)
        elif 0.6 < alive_percentage <= 0.8:
            # 规则2：存活比例在60%~80%之间且自爆范围内红方数量不少于2
            blue_explode_mask &= (red_counts_in_zone >= 2)
        else:
            # 规则3：其它情况不自爆
            blue_explode_mask[:] = False

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
        
        # 更新激活掩码
        self.blue_active_mask &= self.blue_alives
        
    def blue_softkill(self):
        """
        判断蓝方智能体是否需要开启软杀伤（软杀伤只能开启一次）
        规则如下：
        1.如果蓝方智能体在基地150m的范围内，且软杀伤范围内存在红方智能体，则开启软杀伤
        2.如果蓝方剩余智能体超过80% 且 软杀伤范围内红方智能体数量超过1，则开启软杀伤
        3.如果蓝方剩余智能体在60%到80% 且 自爆范围内红方智能体数量超过2，则开启软杀伤
        4.其它情况下开启软杀伤
        """
        # 初始化蓝方软杀伤掩码，仅考虑已激活且携带软杀伤载荷的智能体
        blue_softkill_mask = self.blue_softkill_mode_mask & (self.blue_softkill_time < self.softkill_time) & self.blue_active_mask &~self.blue_interfere_damage_mask
        
        if not np.any(blue_softkill_mask):
            return
        
        # 计算当前蓝方存活智能体的数量及其占比
        alive_count = np.sum(self.blue_alives)
        alive_percentage = alive_count / self.n_blues
        
        # 判断红方智能体是否在每个蓝方软杀伤范围内
        red_in_softkill_zone = (self.distances_blue2red < self.softkill_radius) & self.red_alives
        
        # 计算每个蓝方软杀伤范围内的红方智能体数量
        red_counts_in_zone = np.sum(red_in_softkill_zone, axis=1)
        
        # 计算蓝方智能体到红方基地中心的距离
        dists_to_center = np.linalg.norm(self.blue_positions - self.red_base_center, axis=1)
        
        # 判断是否满足开启软杀伤的条件并更新掩码
        close_to_base = dists_to_center < 150  # 蓝方距离基地中心小于150米
        if alive_percentage >= 0.8:
            # 条件1: 存活比例超过80%且软杀伤范围内红方数量大于1，或距离基地近
            blue_softkill_mask &= ((red_counts_in_zone >= 1) | close_to_base)
        elif 0.6 < alive_percentage <= 0.8:
            # 条件2: 存活比例在60%-80%之间且软杀伤范围内红方数量大于2，或距离基地近
            blue_softkill_mask &= ((red_counts_in_zone >= 2) | close_to_base)
        else:
            # 条件3: 其它情况不开启软杀伤
            blue_softkill_mask[:] = False
        
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
        
        # 更新激活掩码
        self.blue_active_mask &= self.blue_alives
    
    def blue_interfere(self, target_positions):
        """
        判断蓝方智能体是否需要开启干扰
        规则如下：
        1.如果敌机与友机距离小于80m，且可以干扰到（或可以转向干扰到），则开启干扰
        2.如果在基地150m范围内存在敌机，且可以干扰到（或可以转向干扰到）敌机，则开启干扰
        """
        # 初始化蓝方干扰掩码，仅考虑已激活且携带干扰载荷的智能体
        blue_interfere_mask = self.blue_interfere_mode_mask & (self.blue_interfere_duration == 0) & self.blue_active_mask & ~self.blue_interfere_damage_mask
        
        # 已经开了干扰但未结束的智能体
        blue_interfering_mask = (
            (self.blue_interfere_duration > 0) &
            (self.blue_interfere_duration < self.interfere_duration) &
            self.blue_interfere_mode_mask &
            self.blue_alives
        )
        
        if not np.any(blue_interfere_mask) and not np.any(blue_interfering_mask):
            return target_positions
        
        # 计算蓝方干扰范围内的红方智能体
        red_in_interfere_zone = (
            (self.distances_blue2red < self.can_interfere_radius) &
            (np.abs(self.angles_diff_blue2red) < self.can_interfere_angle / 2) &
            self.red_alives
        ) # (M, N)
        
        # 计算蓝方干扰范围内的蓝方智能体
        blue_in_interfere_zone = (
            (self.distances_blue2blue < self.can_interfere_radius) &
            (np.abs(self.angles_diff_blue2blue) < self.can_interfere_angle / 2) &
            self.blue_alives
        ) # (M, M)
             
        # 计算蓝方智能体与红方智能体的距离是否小于80m
        closer_blue_red_mask = self.distances_blue2red < 80 # (M, N)
        
        # 计算红方智能体是否在基地150m范围内
        dists_to_center = np.linalg.norm(self.red_positions - self.red_base_center, axis=1) # (N, )
        red_near_center = dists_to_center < 150 # (N, )
        
        # 条件1：在干扰范围内且距离小于80m的红方智能体
        condition_1_mask = np.any(red_in_interfere_zone & closer_blue_red_mask, axis=1) # (M, )
        
        # 条件2：在基地150m范围内且干扰范围内存在的红方智能体
        condition_2_mask = np.any(red_in_interfere_zone & red_near_center, axis=1) # (M, )
        
        # 更新蓝方干扰掩码
        blue_interfere_mask &= (condition_1_mask | condition_2_mask)
        
        # 对已经在干扰中的蓝方智能体保持干扰状态
        blue_interfere_mask |= blue_interfering_mask
        
        if not np.any(blue_interfere_mask):
            return target_positions
        
        # 更新目标位置
        # 对于每个蓝方智能体，寻找最近的友机和敌机
        # 初始化目标位置为默认目标位置
        new_target_positions = target_positions.copy()
        
        # 对于干扰中的蓝方智能体，计算最近的友机和敌机
        blue_dists = np.where(blue_in_interfere_zone, self.distances_blue2blue, np.inf) # 过滤掉不在干扰范围的友机
        closest_blue_indices = np.argmin(blue_dists, axis=1)    # 找到最近的友机
        
        red_dists = np.where(red_in_interfere_zone, self.distances_blue2red, np.inf)    # 过滤掉不在干扰范围的敌机
        closest_red_indices = np.argmin(red_dists, axis=1)      # 找到最近的敌机
        
        # 更新目标位置：优先选择最近的友机，其次是敌机，最后保留默认目标
        blue_targets = np.where(
            blue_in_interfere_zone[np.arange(blue_in_interfere_zone.shape[0]), closest_blue_indices][:, np.newaxis],
            self.blue_positions[closest_blue_indices],
            np.where(
                red_in_interfere_zone[np.arange(red_in_interfere_zone.shape[0]), closest_red_indices][:, np.newaxis],
                self.red_positions[closest_red_indices],
                new_target_positions
            )
        )
        
        # 仅更新蓝方干扰掩码为True的目标位置
        new_target_positions[blue_interfere_mask] = blue_targets[blue_interfere_mask]
        
        # 更新蓝方干扰统计和状态
        self.blue_interfere_mask = blue_interfere_mask
        self.blue_interfere_count = np.sum(blue_interfere_mask[self.blue_interfere_duration == 0]) # 首次开启
        self.blue_interfere_total += self.blue_interfere_count
        self.blue_interfere_duration[blue_interfere_mask] += 1
        
        # 更新红方被干扰统计和状态
        red_interfere_damage_mask = (
            (self.distances_blue2red < self.interfere_radius) &
            (np.abs(self.angles_diff_blue2red) < self.interfere_angle / 2) &
            self.red_alives
        )
        
        self.red_interfere_damage_mask = np.any(red_interfere_damage_mask[blue_interfere_mask], axis=0)
        self.red_interfere_damage_count = np.sum(self.red_interfere_damage_mask)
        self.red_interfere_damage_total += self.red_interfere_damage_count
        
        # 存储干扰动作
        self.blue_action[blue_interfere_mask, 2] = 3
        
        return target_positions
    
    def blue_collide(self):
        """
        判断蓝方智能体是否要撞击
        规则如下：针对携带干扰和软杀伤载荷且已经开启过干扰和软杀伤的蓝方智能体
        1.如果在基地300m范围内，则选择一个视野内最近的敌机进行撞击
        """
        # 判断哪些蓝方智能体已经完成了干扰或软杀伤操作
        blue_done_interfere_mask = self.blue_interfere_duration == self.interfere_duration
        blue_done_softkill_mask = self.blue_softkill_time == self.softkill_time
        blue_collide_mask = (blue_done_interfere_mask | blue_done_softkill_mask) & self.blue_active_mask & ~self.blue_interfere_damage_mask
        
        # 计算红方智能体到基地中心的距离
        red_dists_to_center = np.linalg.norm(self.red_positions - self.red_base_center, axis=1)
        
        # 判断红方智能体是否在基地300m范围内以及是否在蓝方智能体的碰撞范围内
        red_near_center_and_collide_zone = (
            (self.distances_blue2red < self.collide_radius) &
            (np.abs(self.angles_diff_blue2red) < self.collide_angle / 2) &
            (red_dists_to_center < 300) &
            self.red_alives &
            blue_collide_mask[:, np.newaxis]
        )
        
        if np.any(red_near_center_and_collide_zone):
            # 将不再攻击范围内的智能体距离设置为无限大
            distances_blue2red = self.distances_blue2red.copy()
            distances_blue2red[~red_near_center_and_collide_zone] = np.inf
            
            # 找到每个蓝方智能体最近的红方智能体
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
            
            # 更新激活掩码
            self.blue_active_mask &= self.blue_alives

        
        
    def activate_blue_agents(self):
        """
        根据当前的步数激活相应的蓝方智能体批次。
        """    
        blue_masks = [
            self.blue_first_mask,
            self.blue_second_mask,
            self.blue_third_mask,
            self.blue_fourth_mask
        ]
        
        # 计算当前激活的批次索引
        index = self._episode_steps // self.interval
        
        if 0 <= index < len(blue_masks):
            self.blue_active_mask |= blue_masks[index]
    
    def _init_blue_target_positions(self):
        """
        初始化蓝方每个智能体的目标航迹点
        
        1. 如果智能体与基地中心的距离大于500m，则朝着半径为距离-100m的圆周上的某个点飞行。
        2. 如果自爆载荷的智能体与基地中心距离小于500m，则朝着基地中心飞行。
        3. 如果干扰和软杀伤载荷的智能体与基地中心距离小于500m，则随机飞行。
        4. 如果干扰载荷的智能体与基地中心距离小于150m，且存在敌机，则朝着最近的敌机飞行。
        """
        dist_threshold = 500 # 距离阈值
        dist_cache = 100     # 距离缓冲，避免直接飞到目标点
        
        # 初始化目标点，默认目标点为红方基地中心
        target_positions = np.tile(self.red_base_center, (self.n_blues, 1))
        
        # 计算每个智能体到基地中心的位置
        blue_dists_to_center = np.linalg.norm(self.blue_positions - self.red_base_center, axis=1) # (M, )
        
        # 生成一个随机角度 [-pi, pi]
        theta = (np.random.rand(self.n_blues) - 0.5) * 2 * np.pi
        offsets = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (M, 2)
        
        # 1. 对于在基地中心500m范围外的智能体，指定一个目标点
        mask_outside_threshold = blue_dists_to_center >= dist_threshold
        target_positions[mask_outside_threshold] = (
            offsets[mask_outside_threshold] * (blue_dists_to_center[mask_outside_threshold] - dist_cache)[:, np.newaxis]
        ) + self.red_base_center
        
        # 2. 对于在基地中心500m范围内的自爆载荷智能体，目标点设为基地中心
        mask_explode_inside_threshold = np.logical_and(blue_dists_to_center < dist_threshold, self.blue_explode_mode_mask)
        target_positions[mask_explode_inside_threshold] = self.red_base_center
        
        # 3. 对于在基地中心500m范围内的干扰和软杀伤载荷智能体，指定一个随机目标点
        mask_softkill_or_interfere = np.logical_and(blue_dists_to_center < dist_threshold, ~self.blue_explode_mode_mask)
        target_positions[mask_softkill_or_interfere] = (
            offsets[mask_softkill_or_interfere] * np.random.rand(np.sum(mask_softkill_or_interfere), 1) * dist_threshold
        ) + self.red_base_center

        # 4. 对于在基地中心150m范围内干扰载荷智能体，如果存在敌机，则追击最近的敌机
        red_dists_to_center = np.linalg.norm(self.red_positions - self.red_base_center, axis=1)
        red_near_center_mask = (red_dists_to_center < 150) & self.red_alives
        blue_interfere_near_center_mask = (blue_dists_to_center < 150) & self.blue_interfere_mode_mask
        
        if np.any(red_near_center_mask):
            dists_blue_to_red = np.where(blue_interfere_near_center_mask[:, np.newaxis], self.distances_blue2red, np.inf)
            closest_red_indices = np.argmin(dists_blue_to_red, axis=1)
            
            # 更新干扰载荷智能体的目标位置为最近敌机的位置
            valid_interfering_blues = blue_interfere_near_center_mask & (dists_blue_to_red.min(axis=1) != np.inf)
            target_positions[valid_interfering_blues] = self.red_positions[closest_red_indices[valid_interfering_blues]]
        
        return target_positions
    
    def blue_step(self):
        """
        蓝方智能体的 step 函数，执行自爆、威胁区避让、自爆区域逃离等逻辑，并更新位置和方向。
        """
        if (self._episode_steps % self.interval == 0) and (self._episode_steps // self.interval < 4):
            self.activate_blue_agents()
        
        self.blue_active_mask &= self.blue_alives
        
        # 初始化目标点，默认为红方基地中心
        target_positions = self._init_blue_target_positions()
        
        # 执行蓝方自爆逻辑
        self.blue_explode()
        
        # 执行蓝方软毁伤的逻辑
        self.blue_softkill()
        
        # 执行蓝方干扰的逻辑
        target_positions = self.blue_interfere(target_positions)
        
        # 执行蓝方碰撞的逻辑
        self.blue_collide()

        # 判断智能体是否在警戒区或即将进入警戒区
        is_in_threat, will_in_threat = self.is_in_threat_zone()
        
        # 逃离威胁区域
        target_positions = self.flee_threat_zone(is_in_threat, target_positions)

        # 逃离自爆范围
        target_positions = self.flee_explode_zone(target_positions)

        self._update_blue_position_and_direction( target_positions)

        # 判断蓝方智能体是否进入核心区域
        self.is_hit_core_zone()
        
        # 更新距离矩阵和角度矩阵
        self.update_dist_and_angles()
        
    def _update_blue_position_and_direction(self, target_positions):
        """
        基于运动学模型更新红方智能体的位置和方向。
        仅更新存活&激活的智能体。
        """
        # 仅对存活且激活的智能体进行更新
        active_mask = self.blue_active_mask & self.blue_alives
        
        # 计算期望方向
        dx = target_positions[:, 0] - self.blue_positions[:, 0]
        dy = target_positions[:, 1] - self.blue_positions[:, 1]
        desired_directions = np.arctan2(dy, dx)
        
        # 计算角度差并规范化到 [-pi, pi]区间
        angles_diff = (desired_directions - self.blue_directions + np.pi) % (2 * np.pi) - np.pi

        # 限制转向角度
        angles_diff = np.clip(angles_diff, -self.max_turn, self.max_turn)

        # 更新方向，仅更新有效的智能体
        update_directions_mask = active_mask & ~self.blue_interfere_damage_mask
        self.blue_directions[update_directions_mask] = (
            (self.blue_directions[update_directions_mask] + angles_diff[update_directions_mask] + np.pi) % (2 * np.pi) - np.pi
        )
        
        # 更新位置
        dx = self.blue_velocities[active_mask] * np.cos(self.blue_directions[active_mask]) * self.dt_time
        dy = self.blue_velocities[active_mask] * np.sin(self.blue_directions[active_mask]) * self.dt_time
        self.blue_positions[active_mask] += np.column_stack((dx, dy))
        
        # 存储数据
        self.blue_action[active_mask, 1] = angles_diff[active_mask]

    def get_reward(self):
        """
        计算红方智能体当前时间步的奖励
        """
        # 初始化奖励
        rewards = 0
        
        # 时间奖励：每对抗1步，奖励-0.1，时间步越多，系数越大
        time_penalty = -0.1 * (1 + self._episode_steps / self.episode_limit)
        rewards += time_penalty
        
        # 计算红方智能体被毁伤的数量
        be_killed_num = (
            self.red_explode_damage_count + 
            self.red_softkill_damage_count + 
            self.red_interfere_damage_count +
            self.red_collide_damage_count +
            self.red_threat_damage_count
        ) 
        
        # 计算红方智能体的总毁伤数量
        be_killed_total = (
            self.red_explode_damage_total +
            self.red_softkill_damage_total +
            self.red_interfere_damage_total +
            self.red_collide_damage_total + 
            self.red_threat_damage_total
        )
        
        # # 智能体被毁伤的惩罚：每被毁伤一个智能体，奖励-5，被毁伤的数量越多，系数越大
        # damage_penalty = -5 * be_killed_num * (1 + be_killed_total / self.n_reds)
        # rewards += damage_penalty
        
        # # 高价值区域被攻击奖励：高价值区域每被攻击一次，奖励-20，被攻击次数越多，系数越大
        # core_attack_penalty = -20 * self.attack_core_num * (1 + self.attack_core_total / self.max_attack_core_num)
        # rewards += core_attack_penalty
        
        # 计算红方智能体发动攻击的数量
        attack_num = (
            self.red_explode_count + 
            self.red_softkill_count + 
            self.red_interfere_count +
            self.red_collide_count
        ) 
        
        # 计算红方智能体的总攻击数量
        attack_total = (
            self.red_explode_total +
            self.red_softkill_total +
            self.red_interfere_total +
            self.red_collide_total
        )
        
        # 红方智能体发动攻击奖励：每发动一次攻击，奖励2，攻击次数越多，奖励越大
        attack_reward = 20 * attack_num * (1 + attack_total / self.n_reds)
        rewards += attack_reward
        
        # 计算蓝方智能体被毁伤的数量
        kill_num = (
            self.blue_explode_damage_count + 
            self.blue_softkill_damage_count + 
            self.blue_interfere_damage_count +
            self.blue_collide_damage_count +
            self.blue_threat_damage_count
        ) 
        
        # 计算蓝方智能体的总毁伤数量
        kill_total = (
            self.blue_explode_damage_total +
            self.blue_softkill_damage_total +
            self.blue_interfere_damage_total +
            self.blue_collide_damage_total + 
            self.blue_threat_damage_total
        )
        
        # 毁伤蓝方智能体奖励：每毁伤一个蓝方智能体，奖励10，毁伤的越多，系数越大
        kill_reward = 30 * (1 + kill_total // 10) * kill_num * (1 + kill_total / self.n_blues)
        rewards += kill_reward
        
        # print(f"time reward: {time_penalty:7.2f} attack reward: {attack_reward:7.2f} kill_reward: {kill_reward:7.2f} total_reward: {rewards:7.2f}")
        
        return rewards
    
    def get_result(self):
        """
        判断对抗的结果，并返回对抗是否结束，红方是否获胜以及结果的描述信息。
        
        返回：
        terminated: 布尔值，表示对抗是否结束。
        win: 布尔值，表示红方是否获胜。
        info: 字符串，描述对抗结果的信息。
        """
        
        # 计算存活的智能体数量
        n_red_alive = np.sum(self.red_alives)
        n_blue_alive = np.sum(self.blue_alives)

        # 初始化对抗结束标志和获胜方标志
        terminated = False
        win = False
        info = ""
        
        core_damage = self.attack_core_total >= self.max_attack_core_num
        
        if n_red_alive == 0:
            terminated = True
            win = False
            info = "[Defeat] All Red dead."
        
        # 判断所有蓝方智能体是否被消灭
        elif n_blue_alive == 0:
            terminated = True
            win = True
            info = "[Win] All blue dead."
            
        # 判断回合是否超时
        elif self._episode_steps >= self.episode_limit:
            terminated = True
            if core_damage:
                win = False
                info = "[Defeat] Base destroyed."
            else:
                win = True
                info = '[Win] Time out.'

        return terminated, win, info
    
    def get_result_old(self):
        """
        判断对抗的结果，并返回对抗是否结束，红方是否获胜以及结果的描述信息。
        
        返回：
        terminated: 布尔值，表示对抗是否结束。
        win: 布尔值，表示红方是否获胜。
        info: 字符串，描述对抗结果的信息。
        """
        
        # 计算存活的智能体数量
        n_red_alive = np.sum(self.red_alives)
        n_blue_alive = np.sum(self.blue_alives)

        # 初始化对抗结束标志和获胜方标志
        terminated = False
        win = False
        info = ""

        # 判断红方核心区域是否被摧毁
        if self.attack_core_total >= self.max_attack_core_num:
            terminated = True
            win = False
            info = "[Defeat] Base destroyed."
        
        # 判断所有蓝方智能体是否被消灭
        elif n_blue_alive == 0:
            terminated = True
            win = True
            info = "[Win] All blue dead."
            
        # 判断回合是否超时
        elif self._episode_steps >= self.episode_limit:
            terminated = True
            win = True
            info = '[Win] Time out.'

        return terminated, win, info
    
    def _init_red_target_positions_old(self):
        """
        初始化红方智能体的目标航迹点
        
        1. 如果智能体与基地中心的距离大于500m，则朝着半径为距离-100m的圆周上的某个点飞行。
        2. 如果智能体与基地中心的距离小于500m，则朝着半径500m圆内的任意一点飞行。
        """
        dist_threshold = 500
        dist_cache = 100
            
        # 初始化目标点，默认目标点为红方基地中心
        target_positions = np.tile(self.red_base_center, (self.n_reds, 1))
        
        # 计算每个智能体到基地中心的距离
        dists_to_center = np.linalg.norm(self.red_positions - self.red_base_center, axis=1)
        
        # 生成一个随机角度 [-pi, pi]
        theta = (np.random.rand(self.n_reds) - 0.5) * 2 * np.pi
        offsets = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        
        # 1.对于在基地中心500m范围外的智能体，指定一个目标点
        mask_outside_threshold = dists_to_center >= dist_threshold
        target_positions[mask_outside_threshold] = (
            offsets[mask_outside_threshold] * (dists_to_center[mask_outside_threshold] - dist_cache)[:, np.newaxis]
        ) + self.red_base_center
        
        # 2.对于在基地中心500m范围内的智能体，指定一个随机目标点
        mask_inside_threshold = dists_to_center < dist_threshold
        target_positions[mask_inside_threshold] = (
            offsets[mask_inside_threshold] * np.random.rand(np.sum(mask_inside_threshold), 1) * dist_threshold
        ) + self.red_base_center
        
        return target_positions 
    
    def _init_red_target_positions(self):
        """
        初始化红方智能体的目标航迹点
        
        1. 如果智能体与基地中心的距离大于500m，则朝着半径为距离-100m的圆周上的某个点飞行。
        2. 如果智能体与基地中心的距离小于500m，则朝着半径500m圆内的任意一点飞行。
        """
        dist_threshold = 500
        dist_cache = 100
            
        # 初始化目标点，默认目标点为红方基地中心
        target_positions = np.tile(self.red_base_center, (self.n_reds, 1))
        
        # 计算每个智能体到基地中心的距离
        dists_to_center = np.linalg.norm(self.red_positions - self.red_base_center, axis=1)
        
        # 生成一个随机角度 [-pi, pi]
        theta = (np.random.rand(self.n_reds) - 0.5) * 2 * np.pi
        offsets = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        
        # 判断智能体是否在基地外侧
        mask_outside_base = dists_to_center > self.red_base_radius
        
        # 1.对于在基地外侧的智能体，给一个靠近基地的随机目标点
        if np.any(mask_outside_base):
            target_positions[mask_outside_base] = (
                offsets[mask_outside_base] * (dists_to_center[mask_outside_base] - dist_cache)[:, np.newaxis]
            ) + self.red_base_center
        
        # 2. 对于在基地内的智能体，给一个随机目标点
        if np.any(~mask_outside_base):
            target_positions[~mask_outside_base] = (
                offsets[~mask_outside_base] * self.red_base_center
            ) + self.red_base_center
            
        return target_positions
        
    
    def get_avail_heading_actions(self):
        """
        获取红方智能体的可用航向动作。

        返回:
        available_actions: 布尔数组，形状为 (n_reds, heading_action_num)，
                        表示每个红方智能体的各个航向动作是否可用。
        """
        # 初始化所有航向动作为可用
        available_actions = np.ones((self.n_reds, self.heading_action_num), dtype=bool)

        # 初始化目标点
        target_positions = self._init_red_target_positions()

        # 判断哪些智能体的位置超出缓冲边界
        out_of_bounds = (
            (self.red_positions[:, 0] < -self.half_size_x) | 
            (self.red_positions[:, 0] > self.half_size_x) |
            (self.red_positions[:, 1] < -self.half_size_y) | 
            (self.red_positions[:, 1] > self.half_size_y)
        )
        
        # 获取超出边界的智能体索引
        out_of_bounds_indices = np.where(out_of_bounds)[0]
        
        # 处理超出边界的智能体
        if out_of_bounds_indices.size > 0:
            # 计算超出边界的智能体到每个边界线段的向量和投影点
            pos_vec = self.red_positions[out_of_bounds_indices, np.newaxis, :] - self.cache_bounds[:, 0, :] # 投影向量
            pos_unitvec = pos_vec / self.bounds_len[:, np.newaxis]  # 单位投影向量
            t = np.clip(np.einsum('nij,ij->ni', pos_unitvec, self.bounds_unitvec), 0.0, 1.0)    # 投影比例
            nearest = self.cache_bounds[:, 0, :] + t[:, :, np.newaxis] * self.bounds_vec[np.newaxis, :, :]  # 投影点

            # 计算最近点的距离并更新目标位置
            nearest_dist = np.linalg.norm(self.red_positions[out_of_bounds_indices, np.newaxis, :] - nearest, axis=2)
            nearest_id = np.argmin(nearest_dist, axis=1)
            target_positions[out_of_bounds_indices] = nearest[np.arange(out_of_bounds_indices.size), nearest_id]
         
        # 获取观测到的最近敌机位置
        nearest_enemy_indices = self.observed_enemies[:, 0]
        observe_valid_enemy_indices = np.where(nearest_enemy_indices != -1)[0]
        
        if observe_valid_enemy_indices.size > 0:
            # 更新观测到敌机的红方智能体的目标位置为最近的敌机位置
            nearest_enemy_positions = self.blue_positions[nearest_enemy_indices[observe_valid_enemy_indices]]
            target_positions[observe_valid_enemy_indices] = nearest_enemy_positions
            
        # 计算智能体的期望方向和角度差
        desired_directions = np.arctan2(target_positions[:, 1] - self.red_positions[:, 1],
                                        target_positions[:, 0] - self.red_positions[:, 0])
        angles_diff = (desired_directions - self.red_directions + np.pi) % (2 * np.pi) - np.pi

        # 计算动作限制限，限制智能体的左转和右转
        # 1.如果角度差大于阈值，限制只能选择右转动作（负号表示逆时针）
        mask_pos = angles_diff >= self.max_turn
        available_actions[mask_pos, :self.heading_action_mid_id + 1] = False

        # 2.如果角度差小于负的阈值，限制只能选择左转动作（正号表示顺时针）
        mask_neg = angles_diff <= -self.max_turn
        available_actions[mask_neg, self.heading_action_mid_id:] = False
        
        # 对于被干扰的智能体，只能保持航向 (self.heading_action_mid_id 号动作可用)
        interfere_mask = self.red_interfere_damage_mask
        available_actions[interfere_mask] = False
        available_actions[interfere_mask, self.heading_action_mid_id] = True

        return available_actions
    
    def _transformer_coordinate(self):
        """
        将场景中的相关坐标转换到屏幕坐标
        """
        self.transform_lines()
        self.transform_circles()
    
    def _render_scenario(self):
        """
        渲染跟场景相关的元素
        """
        self._render_circles()
        self._render_lines()
    
    def _render_circles(self):
        """
        渲染圆形区域
        """
        for i in range(self.num_circles):
            pygame.draw.circle(
                self.screen,
                self.circles_color[i],
                self.transformed_circles_center[i],
                self.transformed_circles_radius[i],
                width=self.circles_width[i]
            )
    def _render_lines(self):
        """
        渲染突防通道
        """
        for line in self.transformed_lines:
            pygame.draw.line(self.screen, (255, 0, 0), line[0], line[1], 2)
    
    def _render_text(self):
        """
        渲染屏幕上的文本信息
        """
        n_alive_reds = np.sum(self.red_alives)
        n_alive_blues = np.sum(self.blue_alives)
        
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
        
        blue_attack_text = self.font.render(
            f'Blue Attack Core: [{self.attack_core_num} / {self.attack_core_total}]',
            True,
            blue
        )
        
        blue_threat_text = self.font.render(
            f'Blue Dead by Threat: [{self.blue_threat_damage_count} / {self.blue_threat_damage_total}]',
            True,
            blue
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
        self.screen.blit(blue_attack_text, (10, 450))
        self.screen.blit(blue_threat_text, (10, 490))
        
    def transform_lines(self):
        """
        将红方防线的线段从世界坐标转换为屏幕坐标。
        """
        # 将世界坐标转换为屏幕坐标
        new_center = np.array([-self.size_x / 2, self.size_y / 2])
        new_dir = np.array([1, -1])

        # 应用转换，转换后的线段存储在 self.transformed_lines 中
        self.transformed_lines = ((self.red_lines - new_center) * new_dir * self.scale_factor).astype(int)

    def transform_circles(self):
        """
        将红方核心、基地和蓝方基地的圆形区域从世界坐标转换为屏幕坐标。
        """
        # 初始化转换后的圆心和半径列表
        circles = [self.red_core, self.red_base] + self.blue_bases
        
        self.transformed_circles_center = [
            self.transform_position(circle['center']) for circle in circles
        ]
        
        self.transformed_circles_radius = [
            circle['radius'] * self.scale_factor for circle in circles
        ]
        
        # 设置可视化属性：圆的线宽和颜色
        self.num_circles = len(self.blue_bases) + 2
        self.circles_width = [0] + [2] * (self.num_circles - 1)
        self.circles_color = [(255, 0, 0)] * 2 + [(0, 0, 255)] * (self.num_circles - 2)

def calculate_sector_theta(pos1, pos2, center):
    theta1 = np.arctan2(pos1[1] - center[1], pos1[0] - center[0])
    theta2 = np.arctan2(pos2[1] - center[1], pos2[0] - center[0])
    
    # Normalize theta to the range[0, 2*pi]
    theta1 = (theta1 + 2 * np.pi) % (2 * np.pi)
    theta2 = (theta2 + 2 * np.pi) % (2 * np.pi)

    return theta1, theta2

class Arg(object):
    def __init__(self) -> None:
        self.map_name = '100_vs_100'
        self.scenario_name = 'defense'
        self.episode_length = 400
        self.use_script = False
        self.save_sim_data = True
        self.plane_name = "plane_defense"
        self.debug = False


if __name__ == "__main__":

    args = Arg()

    env = DefenseEnv(args)

    local_obs, global_state, available_actions = env.reset()
    
    import time
    for i in range(400):
        start = time.time()
        actions = env.red_random_policy(available_actions)
        local_obs, global_state, rewards, dones, infos, available_actions = env.step(actions)
        env.render()
        # print(f'[frame: {i}]---[Time: {time.time() - start}]')
    
    env.close()

    # indices, distances = env.find_nearest_grid()
