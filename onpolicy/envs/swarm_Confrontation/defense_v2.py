# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-17
# @Description: Implementation of Agent Class

import os
import pygame
import numpy as np

try:
    from onpolicy.envs.swarm_Confrontation.baseEnv import BaseEnv
    from onpolicy.envs.swarm_Confrontation.utils import assign_target
except:
    from baseEnv import BaseEnv
    from utils import assign_target

    
os.environ["SDL_VIDEODRIVER"] = "dummy"

class DefenseV2Env(BaseEnv):
    def __init__(self, args, name=None):
        super(DefenseV2Env, self).__init__(args, name)

        # 定义红方核心区域
        self.red_core_ranges = self._create_red_cores()
        self.red_core_centers = np.array([d['center'] for d in self.red_core_ranges])
        self.red_core_radiuses = np.array([d['radius'] for d in self.red_core_ranges])
        self.red_core_num = len(self.red_core_ranges)

        # 定义蓝方和红方的边界线
        self.lines = np.array([
            [[-1000.0, 2500.0], [-1000.0, -2500.0]],
            [[ 1000.0, 2500.0], [ 1000.0, -2500.0]],
        ])
        
        self.red_base = {
            "size": np.array([3000.0, 4000.0]),
            "center": np.array([2500.0, 0.0])
        }
        
        self.blue_base = {
            "size": np.array([3000.0, 4000.0]),
            "center": np.array([-2500.0, 0.0])
        }
        
        # 基地的最大攻击次数
        self.max_attack_core_num = 15

    def _create_circle(self, center, radius):
        """
        创建一个圆形区域的字典结构。
        """
        return {'center': np.array(center), 'radius': radius}
        
    def _create_red_cores(self):
        """
        创建红方基地的字典列表。
        """
        return [
            self._create_circle([1500.0,  1500.0], 25.0), 
            self._create_circle([3000.0,   500.0], 25.0), 
            self._create_circle([2000.0, -1500.0], 25.0),
        ]
        
    def distribute_red_agents(self):
        """
        分配红方智能体到基地内外的数量。
        
        返回:
        n_in_bases: 分配到基地内部的红方智能体数量。
        n_out_bases: 分配到基地外部的红方智能体数量。
        """
        n_out_bases = int(0.25 * self.n_reds)
        n_in_bases = np.random.multinomial(self.n_reds - n_out_bases, np.ones(self.red_core_num) / self.red_core_num)
        return n_in_bases, n_out_bases
    
    def generate_red_positions_old(self):
        """
        生成红方智能体的位置和朝向。
        
        返回:
        positions: 红方智能体的位置数组，形状为 (n_reds, 2)。
        directions: 红方智能体的朝向数组，形状为 (n_reds,)。
        """
        # 分配基地内外的智能体数量
        n_in_bases, n_out_bases = self.distribute_red_agents()
        
        # 基地周围的智能体位置
        in_base_positions = np.vstack([
            self._generate_positions_in_circle(size, center, radius + 500)
            for size, center, radius in zip(n_in_bases, self.red_core_centers, self.red_core_radiuses)
        ])
        
        # 基地外部智能体的位置
        size = self.red_base['size']
        center = self.red_base['center']
        out_base_positions = (np.random.rand(n_out_bases, 2) - 0.5) * size + center

        # 合并所有智能体的位置
        positions = np.empty((self.n_reds, 2))
        positions[:n_out_bases] = out_base_positions
        positions[n_out_bases:] = in_base_positions
        
        # 生成朝向    
        directions = np.random.uniform(-np.pi, np.pi, self.n_reds)

        return positions, directions
    
    def generate_red_positions(self):
        """
        生成红方智能体的位置和朝向。
        
        返回:
        positions: 红方智能体的位置数组，形状为 (n_blues, 2)。
        directions: 红方智能体的朝向数组，形状为 (n_blues,)。
        """
        # 初始化位置矩阵
        size = self.red_base['size']
        center = self.red_base['center']
        positions = (np.random.rand(self.n_reds, 2) - 0.5) * size + center
        
        # 计算智能体的朝向  
        directions = np.random.uniform(-np.pi, np.pi, self.n_reds)
        
        return positions, directions

    def generate_blue_positions(self):
        """
        生成蓝方智能体的位置和朝向。
        
        返回:
        positions: 蓝方智能体的位置数组，形状为 (n_blues, 2)。
        directions: 蓝方智能体的朝向数组，形状为 (n_blues,)。
        """
        # 初始化位置矩阵
        size = self.blue_base['size']
        center = self.blue_base['center']
        positions = (np.random.rand(self.n_blues, 2) - 0.5) * size + center
        
        # 计算智能体的朝向  
        directions = np.random.uniform(-np.pi, np.pi, self.n_blues)
        
        return positions, directions
    
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
        
        # 初始化每个智能体的目标基地
        self._init_red_target_core()
        self._init_blue_target_core()
        
        # 初始每个高价值区域的状态
        self._init_red_core_states()

        # 获取局部观测值、全局状态和可用动作
        local_obs = self.get_obs()
        global_state = [self.get_state()] * self.n_reds
        available_actions = self.get_avail_actions()
        
        return local_obs, global_state, available_actions

    def _init_blue_target_core(self):
        """
        给每个蓝方智能体随机选择一个红方高价值区域作为打击目标
        """        
        self.blue_target_core_ids = np.random.randint(self.red_core_num, size=self.n_blues)
        self.blue_target_core_centers = self.red_core_centers[self.blue_target_core_ids]
        self.blue_target_core_radiuses = self.red_core_radiuses[self.blue_target_core_ids]
        
    def _init_red_target_core(self):
        """
        给每个蓝方智能体随机选择一个红方高价值区域作为打击目标
        """
        assert self.red_core_num == 3
        self.red_target_core_ids = assign_target(self.red_positions, self.red_core_centers)
        # self.red_target_core_ids = np.random.randint(self.red_core_num, size=self.n_reds)
        self.red_target_core_centers = self.red_core_centers[self.red_target_core_ids]
        self.red_target_core_radiuses = self.red_core_radiuses[self.red_target_core_ids]
        
    def _init_red_core_states(self):
        """
        初始化每个高价值区域的状态和打击次数
        """
        self.red_core_alives = np.ones(self.red_core_num, dtype=bool)
        self.red_core_attack_num = np.zeros(self.red_core_num)
    
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
    
    def is_hit_core_zone(self):
        """
        判断蓝方智能体是否在红方高价值区域内，并更新打击次数和蓝方智能体的存活状态。
        """
        # 计算每个蓝方智能体到红方高价值区域中心的距离
        dists_to_center = np.linalg.norm(self.blue_positions[:, np.newaxis, :] - self.red_core_centers[np.newaxis, :, :], axis=-1)

        # 判断自爆载荷的蓝方智能体是否在红方高价值区域内
        in_red_core = (dists_to_center < self.red_core_radiuses[np.newaxis, :]) & self.blue_explode_mode_mask[:, np.newaxis] # (100, 3)

        # 计算并更新在红方高价值区域内被打击的蓝方智能体数量
        alive_num = np.sum(self.red_core_alives)
        attack_core_num = np.sum(in_red_core & self.blue_alives[:, np.newaxis], axis=0) # (3,)
        self.red_core_attack_num += attack_core_num
        self.red_core_alives[self.red_core_attack_num >= self.max_attack_core_num] = False
        self.destropy_core_num = alive_num - np.sum(self.red_core_alives)
        
        self.attack_core_num = np.sum(attack_core_num)
        self.attack_core_total += self.attack_core_num

        # 将在高价值区域内的蓝方智能体标记为死亡
        self.blue_alives[np.any(in_red_core, axis=1)] = False
        
    def blue_explode(self):
        """
        判断蓝方智能体是否需要自爆:
        规则：
        1. 如果蓝方存活智能体的数量超过80%且自爆范围内的红方智能体数量超过1,则自爆。
        2. 如果蓝方存活智能体的数量在60%和80%之间且自爆范围内红方智能体的数量超过2,则自爆。
        3. 其它情况下不攻击。
        """
        # 初始化蓝方自爆掩码，仅考虑已激活且携带自爆载荷的智能体
        blue_explode_mask = (
            self.blue_explode_mode_mask & 
            self.blue_alives &
            ~self.blue_interfere_damage_mask
        )
        
        if not np.any(blue_explode_mask):
            return
        
        # 计算当前蓝方存活智能体的比例
        alive_percentage = np.sum(self.blue_alives) / self.n_blues
        
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
        blue_softkill_mask = (
            self.blue_softkill_mode_mask & 
            (self.blue_softkill_time < self.softkill_time) & 
            self.blue_alives &
            ~self.blue_interfere_damage_mask
        )
        
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
        dists_to_center = np.linalg.norm(self.blue_positions - self.blue_target_core_centers, axis=1)
        
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
        
    def blue_interfere(self, target_positions):
        """
        判断蓝方智能体是否需要开启干扰
        规则如下：
        1.如果敌机与友机距离小于80m，且可以干扰到（或可以转向干扰到），则开启干扰
        2.如果在基地150m范围内存在敌机，且可以干扰到（或可以转向干扰到）敌机，则开启干扰
        """
        # 初始化蓝方干扰掩码，仅考虑已激活且携带干扰载荷的智能体
        blue_interfere_mask = (
            self.blue_interfere_mode_mask & 
            (self.blue_interfere_duration == 0) & 
            self.blue_alives & 
            ~self.blue_interfere_damage_mask)
        
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
        dists_to_center = np.linalg.norm(self.red_positions - self.red_target_core_centers, axis=1) # (N, )
        red_near_center_mask = dists_to_center < 150 # (N, )
        
        # 条件1：在干扰范围内且距离小于80m的红方智能体
        condition_1_mask = np.any(red_in_interfere_zone & closer_blue_red_mask, axis=1) # (M, )
        
        # 条件2：在基地150m范围内且干扰范围内存在的红方智能体
        condition_2_mask = np.any(red_in_interfere_zone & red_near_center_mask, axis=1) # (M, )
        
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
        blue_collide_mask = (
            (blue_done_interfere_mask | blue_done_softkill_mask) & 
            self.blue_alives & 
            ~self.blue_interfere_damage_mask
        )
        
        # 计算红方智能体到基地中心的距离
        red_dists_to_center = np.linalg.norm(self.red_positions - self.red_target_core_centers, axis=1)
        
        # 判断红方智能体是否在基地300m范围内以及是否在蓝方智能体的碰撞范围内
        red_near_center_and_collide_zone = (
            (self.distances_blue2red < self.collide_radius) &
            (np.abs(self.angles_diff_blue2red) < self.collide_angle / 2) &
            (red_dists_to_center < 300) &
            self.red_alives &
            blue_collide_mask[:, np.newaxis]
        )
        
        if not np.any(red_near_center_and_collide_zone):
            return
        
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
        
    def _update_blue_target_core(self):
        """
        更新智能体的目标基地：如果目标基地已经被摧毁，则随机选择其它未被摧毁的基地作为打击目标。
        """
        # 获取每个蓝方智能体目标基地的状态
        target_core_alives = self.red_core_alives[self.blue_target_core_ids]
        
        # 计算需要更新目标的掩码：存活但是目标被摧毁的蓝方智能体
        mask = self.blue_alives & ~target_core_alives
        num = np.sum(mask)
        
        if num > 0 and np.any(self.red_core_alives):
            # 获取未被摧毁的基地索引
            available_targets = np.where(self.red_core_alives)[0]
            
            # 随机选择新的目标基地（允许重复选择）
            new_targets = np.random.choice(available_targets, size=num, replace=True)
            
            # 更新目标基地
            self.blue_target_core_ids[mask] = new_targets
            self.blue_target_core_centers[mask] = self.red_core_centers[new_targets]
            self.blue_target_core_radiuses[mask] = self.red_core_radiuses[new_targets]
        
    def _init_blue_target_positions(self):
        """
        初始化蓝方每个智能体的目标航迹点
        
        1. 更新智能体的目标基地。
        2. 对于距离目标基地中心超过500m的智能体，朝距离基地中心500m - 100m的圆周上的某个点飞行。
        3. 对于距离基地中心小于500m且为自爆载荷的智能体，直接飞向基地中心。
        4. 对于距离基地中心小于500m且为干扰或软杀伤载荷的智能体，随机飞行。
        5. 对于距离基地中心小于150m的干扰载荷智能体，如果有敌机存在，则追击最近的敌机。
        """
        dist_threshold = 500 # 距离阈值
        dist_cache = 100     # 距离缓冲，避免直接飞到目标点
        
        # 1.更新智能体的目标基地
        self._update_blue_target_core()
        
        # 初始化目标位置为基地中心位置
        target_positions = self.blue_target_core_centers.copy()
        
        # 生成一个随机角度 [-pi, pi]，用于随机航迹点的偏移计算
        theta = (np.random.rand(self.n_blues) - 0.5) * 2 * np.pi
        offsets = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n_blues, 2)
        
        # 计算每个蓝方智能体与目标基地中心的距离
        blue_dists_to_center = np.linalg.norm(self.blue_positions - self.blue_target_core_centers, axis=1) # (M, )
        
        # 2. 在目标基地中心500m范围外的智能体，朝着距离中心距离为（距离-100m）的某个随机点飞行。
        mask_outside_threshold = blue_dists_to_center >= dist_threshold
        if np.any(mask_outside_threshold):
            target_positions[mask_outside_threshold] = (
                offsets[mask_outside_threshold] * (blue_dists_to_center[mask_outside_threshold] - dist_cache)[:, np.newaxis]
            ) + self.blue_target_core_centers[mask_outside_threshold]
        
        # 3. 在基地中心500m范围内的自爆载荷智能体，直接飞向基地中心
        mask_explode_inside_threshold = np.logical_and(blue_dists_to_center < dist_threshold, self.blue_explode_mode_mask)
        target_positions[mask_explode_inside_threshold] = self.blue_target_core_centers[mask_explode_inside_threshold]
        
        # 4. 在基地中心500m范围内的干扰和软杀伤载荷智能体，朝着随机方向飞行
        mask_softkill_or_interfere = np.logical_and(blue_dists_to_center < dist_threshold, ~self.blue_explode_mode_mask)
        if np.any(mask_softkill_or_interfere):
            random_offsets = np.random.rand(np.sum(mask_softkill_or_interfere), 1) * dist_threshold
            target_positions[mask_softkill_or_interfere] = (
                offsets[mask_softkill_or_interfere] * random_offsets
            ) + self.blue_target_core_centers[mask_softkill_or_interfere]

        # 4. 对于在基地中心150m范围内干扰载荷智能体，如果存在敌机，则追击最近的敌机
        blue_interfere_near_center_mask = (blue_dists_to_center < 150) & self.blue_interfere_mode_mask
        
        if np.any(blue_interfere_near_center_mask) and np.any(self.red_alives):
            # 计算干扰载荷智能体与敌机的距离
            dists_blue_to_red = self.distances_blue2red[blue_interfere_near_center_mask]
            closest_red_indices = np.argmin(dists_blue_to_red, axis=1)
            
            # 更新干扰载荷智能体的目标位置为最近敌机的位置
            target_positions[blue_interfere_near_center_mask] = self.red_positions[closest_red_indices]
        
        return target_positions
    
    def blue_step(self):
        """
        蓝方智能体的 step 函数，执行自爆、威胁区避让、自爆区域逃离等逻辑，并更新位置和方向。
        """
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
        active_mask = self.blue_alives
        
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
        
        # 时间惩罚：每对抗1步，给予-0.1的惩罚
        time_penalty = -0.1
        rewards += time_penalty
        
        # 开启软杀伤奖励：每开启一次软杀伤，给予5的奖励，优先开启软杀伤
        open_softkill_reward = 5 * self.red_softkill_count
        rewards += open_softkill_reward
        
        # 开启干扰奖励：每开启一次干扰，给予2的奖励，鼓励开启干扰
        # open_interfere_reward = 2 * self.red_interfere_count
        open_interfere_reward = 5 * self.red_interfere_count
        rewards += open_interfere_reward
        
        # 开启自爆奖励：每开启一次自爆，给予0的奖励，鼓励开启
        # open_explode_reward = 0
        open_explode_reward = 3 * self.red_explode_count
        rewards += open_explode_reward
        
        # 有效软杀伤奖励：每软杀伤一个蓝方智能体，给予20的奖励
        valid_softkill_reward = 20 * self.blue_softkill_damage_count
        rewards += valid_softkill_reward
        
        # 有效干扰奖励：每有效干扰一个蓝方智能体，给予5的奖励
        valid_interfere_reward = 5 * self.blue_interfere_damage_count
        rewards += valid_interfere_reward
        
        # 有效自爆奖励：每有效自爆一个蓝方智能体，给予10的奖励
        valid_explode_reward = 10 * self.blue_explode_damage_count
        rewards += valid_explode_reward
        
        # 无效自爆惩罚：每无效自爆一次，给予-1的惩罚
        # invalid_explode_penalty = -1 * self.red_invalide_explode_count
        invalid_explode_penalty = 0
        rewards += invalid_explode_penalty
        
        # 被软杀伤惩罚：每被软杀伤一个，给予-10的惩罚
        softkill_damage_penalty = -2 * self.red_softkill_damage_count
        rewards += softkill_damage_penalty
        
        # 被干扰惩罚：每被干扰一个，给予-1的惩罚
        interfere_damage_penalty = -1 * self.red_interfere_damage_count
        rewards += interfere_damage_penalty
        
        # 被自爆惩罚：每被自爆毁伤一个，给予-5的惩罚
        explode_damage_penalty = -5 * self.red_explode_damage_count
        rewards += explode_damage_penalty
        
        # 被碰撞惩罚：每被碰撞一个，给予-5的惩罚
        collide_damge_penalty = -5 * self.red_collide_damage_count
        rewards += collide_damge_penalty
        
        # 打击基地惩罚： 基地每被打击一次，给予-20的惩罚
        # base_damge_penalty = -30 * self.attack_core_num - 50 * self.destropy_core_num
        base_damge_penalty = -20 * self.attack_core_num
        rewards += base_damge_penalty if self.reward_base else 0
        
        return rewards
    
    def get_reward_old(self):
        """
        计算红方智能体当前时间步的奖励
        """
        # 初始化奖励
        rewards = 0
        
        # 时间奖励：每对抗1步，奖励-0.1，时间步越多，系数越大
        # time_penalty = -0.1 * (1 + self._episode_steps / self.episode_limit)
        time_penalty = -0.1
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
        
        # 智能体被毁伤的惩罚：每被毁伤一个智能体，奖励-5，被毁伤的数量越多，系数越大
        # damage_penalty = -5 * be_killed_num * (1 + be_killed_total / self.n_reds)
        damage_penalty = -5 * be_killed_num
        rewards += damage_penalty
        
        # 高价值区域被攻击奖励：高价值区域每被攻击一次，奖励-20，被攻击次数越多，系数越大
        # core_attack_penalty = -20 * self.attack_core_num * (1 + self.attack_core_total / self.max_attack_core_num)
        core_attack_penalty = -20 * self.attack_core_num
        rewards += core_attack_penalty
        
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
        # attack_reward = 1 * attack_num * (1 + attack_total / self.n_reds)
        attack_reward = 2 * attack_num
        rewards += attack_reward
        
        # 计算蓝方智能体被毁伤的数量
        kill_num = (
            self.blue_explode_damage_count + 
            self.blue_softkill_damage_count + 
            self.blue_interfere_damage_count +
            self.blue_collide_damage_count
            # self.blue_threat_damage_count
        ) 
        
        # 计算蓝方智能体的总毁伤数量
        kill_total = (
            self.blue_explode_damage_total +
            self.blue_softkill_damage_total +
            self.blue_interfere_damage_total +
            self.blue_collide_damage_total
            # self.blue_threat_damage_total
        )
        
        # 毁伤蓝方智能体奖励：每毁伤一个蓝方智能体，奖励10，毁伤的越多，系数越大
        # kill_reward = 5 * (1 + kill_total // 10) * kill_num * (1 + kill_total / self.n_blues)
        kill_reward = 10 * kill_num
        rewards += kill_reward
        
        # print(f"time reward: {time_penalty:7.2f} attack reward: {attack_reward:7.2f} kill_reward: {kill_reward:7.2f} total_reward: {rewards:7.2f}")
        
        return rewards
    
    def get_result(self):
        """
        判断对抗的结果，并返回对抗是否结束，红方是否获胜以及结果的描述信息。
        
        返回：
        - terminated: 布尔值，表示对抗是否结束。
        - win: 布尔值，表示红方是否获胜。
        - info: 字符串，描述对抗结果的信息。
        """
        
        # 计算存活的智能体数量
        n_red_alive = np.sum(self.red_alives)
        n_blue_alive = np.sum(self.blue_alives)

        # 初始化对抗结束标志和获胜方标志
        terminated = False
        win = False
        info = ""

        # 判断红方核心区域是否被摧毁
        if np.sum(self.red_core_alives) < 2:
            terminated = True
            win = False
            info = "[Defeat] Base destroyed."
        
        # 判断所有蓝方智能体是否被消灭
        elif n_blue_alive == 0:
            terminated = True
            win = True
            info = "[Win] All blue dead."

        # 判断所有红方智能体是否被消灭
        elif n_red_alive == 0:
            terminated = True
            win = False
            info = "[Defeat] All red dead."
            
        # 判断回合是否超时
        elif self._episode_steps >= self.episode_limit:
            terminated = True
            win = True
            info = '[Win] Time out.'

        return terminated, win, info
    
    def _update_red_target_core(self):
        """
        更新智能体的目标基地：如果目标基地已经被摧毁，则随机选择其它未被摧毁的基地作为打击目标。
        """
        # 获取每个蓝方智能体目标基地的状态
        target_core_alives = self.red_core_alives[self.red_target_core_ids]
        
        # 计算需要更新目标的掩码：存活但是目标被摧毁的蓝方智能体
        mask = self.red_alives & ~target_core_alives
        num = np.sum(mask)
        
        if num > 0 and np.any(self.red_core_alives):
            # 获取未被摧毁的基地索引
            available_targets = np.where(self.red_core_alives)[0]
            
            # 随机选择新的目标基地（允许重复选择）
            new_targets = np.random.choice(available_targets, size=num, replace=True)
            
            # 更新目标基地
            self.red_target_core_ids[mask] = new_targets
            self.red_target_core_centers[mask] = self.red_core_centers[new_targets]
            self.red_target_core_radiuses[mask] = self.red_core_radiuses[new_targets]
            
    def _init_red_target_positions(self):
        """
        初始化红方智能体的目标航迹点
        
        1. 对于在基地外的智能体，距离缩减 100m。
        2. 对于在基地内的智能体，随机飞。
        """
        # 缓冲距离
        dist_threshold = 300
        dist_cache = 100
            
        # 初始化目标点，默认目标点为红方基地中心
        self._update_red_target_core()
        target_positions = self.red_target_core_centers.copy()
        
        # 计算每个智能体到基地中心的距离
        dists_to_center = np.linalg.norm(self.red_positions - self.red_target_core_centers, axis=1)
        
        # 生成一个随机角度 [-pi, pi] 和偏移量
        theta = np.random.uniform(-np.pi, np.pi, size=self.n_reds)
        offsets = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        
        # 判断智能体是否在基地外侧
        mask_outside_base = dists_to_center > dist_threshold
        
        # 1.对于在基地外侧的智能体，距离缩减 100
        if np.any(mask_outside_base):
            target_positions[mask_outside_base] = (
                offsets[mask_outside_base] * (dists_to_center[mask_outside_base] - dist_cache)[:, np.newaxis]
            ) + self.red_target_core_centers[mask_outside_base]
        
        # 2. 对于在基地内的智能体，给一个随机点
        if np.any(~mask_outside_base):
            target_positions[~mask_outside_base] = (
                offsets[~mask_outside_base] * np.random.rand(np.sum(~mask_outside_base), 1) * (dist_threshold - 50)
            ) + self.red_target_core_centers[~mask_outside_base]
            
        return target_positions, mask_outside_base
    
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
        pos_vec = self.red_positions[out_of_bounds_indices, np.newaxis, :] - self.cache_bounds[:, 0, :] # 投影向量
        pos_unitvec = pos_vec / self.bounds_len[:, np.newaxis]  # 单位投影向量
        t = np.clip(np.einsum('nij,ij->ni', pos_unitvec, self.bounds_unitvec), 0.0, 1.0)    # 投影比例
        nearest = self.cache_bounds[:, 0, :] + t[:, :, np.newaxis] * self.bounds_vec[np.newaxis, :, :]  # 投影点

        # 计算最近点的距离并更新目标位置
        nearest_dist = np.linalg.norm(self.red_positions[out_of_bounds_indices, np.newaxis, :] - nearest, axis=2)
        nearest_id = np.argmin(nearest_dist, axis=1)
        target_positions[out_of_bounds_indices] = nearest[np.arange(out_of_bounds_indices.size), nearest_id]
        
        return target_positions
    
    def get_avail_heading_actions_own(self):
        """
        获取红方智能体的可用航向动作。

        返回:
        - available_actions: 布尔数组，形状为 (n_reds, heading_action_num)，
                        表示每个红方智能体的各个航向动作是否可用。
        """
        # 初始化所有航向动作为可用
        available_actions = np.ones((self.n_reds, self.heading_action_num), dtype=bool)

        # 初始化目标点
        target_positions, mask_outside_core = self._init_red_target_positions()

        # 判断哪些智能体的位置超出边界
        out_of_bounds = (
            (self.red_positions[:, 0] < -self.half_size_x) | 
            (self.red_positions[:, 0] > self.half_size_x) |
            (self.red_positions[:, 1] < -self.half_size_y) | 
            (self.red_positions[:, 1] > self.half_size_y)
        )
        
        # 如果有智能体超出边界，修正目标位置
        out_of_bounds_indices = np.where(out_of_bounds)[0]
        if out_of_bounds_indices.size > 0:
            target_positions = self._correct_out_of_bounds_positions(out_of_bounds_indices, target_positions)
         
        # 获取观测到的最近敌机位置
        # NOTE: 不用追踪敌机，守护在基地附近即可
        nearest_enemy_indices = self.observed_enemies[:, 0]
        observe_valid_enemy_indices = np.where(nearest_enemy_indices != -1 & mask_outside_core)[0]
        
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
    
    def get_avail_heading_actions(self):
        if self.use_script:
            return self.get_avail_heading_actions_own()
        else:
            return super().get_avail_heading_actions()
    
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
        for line, color in zip(self.transformed_lines, self.lines_color):
            pygame.draw.line(self.screen, color, line[0], line[1], 1)
    
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
            f'Blue Attack Core: [{int(self.red_core_attack_num[0])} / {int(self.red_core_attack_num[1])} / {int(self.red_core_attack_num[2])}]',
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
        self.transformed_lines = ((self.lines - new_center) * new_dir * self.scale_factor).astype(int)
        self.lines_color = [(0, 0, 255), (255, 0, 0)]

    def transform_circles(self):
        """
        将红方核心、基地和蓝方基地的圆形区域从世界坐标转换为屏幕坐标。
        """
        # 初始化转换后的圆心和半径列表
        circles = self.red_core_ranges
        
        self.transformed_circles_center = [
            self.transform_position(circle['center']) for circle in circles
        ]
        
        self.transformed_circles_radius = [
            circle['radius'] * self.scale_factor for circle in circles
        ]
        
        # 设置可视化属性：圆的线宽和颜色
        self.num_circles = self.red_core_num
        self.circles_width = [0] * self.red_core_num
        self.circles_color = [(255, 0, 0)] * self.red_core_num

class Arg(object):
    def __init__(self) -> None:
        self.map_name = '100_vs_100'
        self.scenario_name = 'defensev2'
        self.episode_length = 400
        self.use_script = True
        self.save_sim_data = True
        self.plane_name = "plane_defense"
        self.debug = True
        self.use_group = False
        self.obs_attack = False
        self.state_attack = False
        self.reward_base = False
        
        self.share_action = True
        
        self.only_explode = False   # 是否只包含自爆
        self.shuffle = False       # 是否打乱飞机类型
        
        self.can_attack_factor = 3
        self.save_log = False


if __name__ == "__main__":

    args = Arg()
    env = DefenseV2Env(args)

    for i in range(5):
        local_obs, global_state, available_actions = env.reset()
        env.render()
        
        
        import time
        done = True
        while done:
            start = time.time()
            actions = env.red_random_policy(available_actions)
            local_obs, global_state, rewards, dones, infos, available_actions = env.step(actions)

            done = ~np.all(dones)

            env.render()

        # if env.red_explode_total == 0:
        #     print(f'Episode: {i}')
            # print(f'[frame: {i}]---[Time: {time.time() - start}]')
    
    env.close()
