# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-17
# @Description: Implementation of Agent Class

import os
import pygame
import numpy as np

try:
    from onpolicy.envs.swarm_Confrontation.baseEnv import BaseEnv
except:
    from baseEnv import BaseEnv
    
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
        
        # 奖励的定义
        self._define_rewards()

        
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
        return [
            self._create_circle([1500.0,  1500.0], 500.0),  # 上右
            self._create_circle([1500.0, -1500.0], 500.0),  # 下右
            self._create_circle([ 500.0,  1500.0], 500.0),  # 上左
            self._create_circle([ 500.0, -1500.0], 500.0),  # 下左
        ]
        
    def _define_rewards(self):
        """
        定义各种奖励参数。
        """
        self.reward_time = -0.1             # 每存活一个时间步的惩罚
        self.reward_explode_red = -5        # 被炸掉的惩罚
        self.reward_attack_core = -20       # 核心区域被攻击的惩罚
        self.reward_win = 3000              # 获胜奖励
        self.reward_defeat = 0              # 失败奖励
        self.reward_kill_blue = 5           # 每个时间步杀死蓝方奖励
        self.reward_near_area = 0.2         # 靠近目标区域的奖励
        self.reward_away_area = -0.2        # 远离目标区域的惩罚
    
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
        n_in_bases, n_out_bases = self.distribute_red_agents()
        
        # 基地内部智能体的位置
        in_base_positions = self._generate_positions_in_circle(n_in_bases, self.red_base_center, self.red_base_radius)

        # 基地外部智能体的位置
        out_base_positions = (np.random.rand(n_out_bases, 2) - 0.5) * np.array([self.size_x, self.size_y])

        # 合并所有智能体的位置和朝向
        positions = np.vstack([in_base_positions, out_base_positions])
        directions = np.random.uniform(-np.pi, np.pi, self.n_reds)

        return positions, directions

    def distribute_blue_agents(self):
        """
        随机将蓝方智能体分配到不同的基地。
        
        返回:
        group_sizes: 每个基地中蓝方智能体的数量数组。
        """
        n_groups = len(self.blue_bases) + 1
        group_sizes = np.random.multinomial(self.n_blues, np.ones(n_groups) / n_groups)

        return group_sizes

    def generate_blue_positions(self):
        """
        生成蓝方智能体的位置和朝向。
        
        返回:
        positions: 蓝方智能体的位置数组，形状为 (n_blues, 2)。
        directions: 蓝方智能体的朝向数组，形状为 (n_blues,)。
        """
        group_sizes = self.distribute_blue_agents()
        self.group_sizes = group_sizes

        blue_bases = [self.red_base] + self.blue_bases

        agent_positions = []
        for group_idx, group_size in enumerate(group_sizes):
            center = blue_bases[group_idx]['center']
            radius = blue_bases[group_idx]['radius']
            
            # 蓝方的第一个组分布在红方基地外围
            if group_idx == 0:
                positions = self._generate_positions_on_circle(group_size, center, radius + 20)
            else:
                positions = self._generate_positions_in_circle(group_size, center, radius)
                
            agent_positions.append(positions)
        
        agent_positions = np.vstack(agent_positions)
        agent_directions = self._calculate_agent_directions(agent_positions, self.red_base_center)
            
        return agent_positions, agent_directions
    
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

        positions = np.vstack([red_positions, blue_positions])
        directions = np.hstack([red_directions, blue_directions])
        velocities = np.hstack([np.full(self.n_reds, self.red_max_vel), np.full(self.n_blues, self.blue_max_vel)])

        return positions, directions, velocities
    
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

        # 初始化红方核心区域的被攻击次数
        self.attack_core_num = 0            # 每个时间步红方核心区域被攻击的次数

        # 初始化当前回合红方核心区域被打击的总次数
        self.attack_core_total = 0          # 当前回合红方高价值区域被打击的总次数
        
        # 初始化累计击杀蓝方的数量
        self.kill_total = 0

        # 初始化红方智能体与目标区域的距离变量
        self.dist2area = None               # 红方智能体与目标区域的距离 

        # 获取局部观测值、全局状态和可用动作
        local_obs = self.get_obs()
        global_state = [self.get_state()] * self.n_reds
        available_actions = self.get_avail_actions()

        return local_obs, global_state, available_actions

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
        
        # 解析红方智能体的动作
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
        
        # 执行蓝方的动作
        self.blue_step()
        
        # 合并状态
        self.merge_state()

        # 更新步数计数器
        self._total_steps += 1
        self._episode_steps += 1

        # 检查是否终止以及是否胜利
        terminated, win, res = self.get_result()
        bad_transition = self._update_result(terminated, win)
        
        # 汇总环境信息
        info = self._collect_info(bad_transition, res)

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
    
    def _collect_info(self, bad_transition, res):
        """
        汇总环境的各种信息。

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
            'explode_ratio': self.red_self_destruction_total / self.n_reds,
            'be_exploded_ratio': self.explode_red_total / self.n_reds,
            'invalid_explode_ratio': self.invalid_explode_red_total / self.n_reds,
            'collide_ratio': self.collide_blue_total / self.n_reds,
            'be_collided_ratio': self.collide_red_total / self.n_reds,
            'kill_num': self.kill_total,
            'hit_core_num': self.attack_core_total,
            'explode_ratio_blue': self.blue_self_destruction_total / self.n_blues,
            'scout_core_ratio': 0,  # 高价值区域被侦察的比例
            'scout_comm_ratio': 0,  # 普通区域被侦察的比例
            'episode_length': self._episode_steps,
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

        # 计算红方智能体与蓝方智能之间的距离矩阵
        distances_red2blue = self.get_dist()

        # 计算每个红方智能体自爆范围的蓝方智能体数量
        blue_num_in_explode_zone = np.sum(distances_red2blue < self.explode_radius, axis=1)

        # 判断哪些红方智能体有自爆倾向（即至少有一个蓝方智能体在其自爆范围内）
        red_will_explode = blue_num_in_explode_zone > 1

        # 如果没有智能体具有自爆倾向，则提前返回
        if not any(red_will_explode):
            return target_positions

        # 找到每个蓝方智能体距离最近的红方智能体的索引
        nearest_id = np.argmin(distances_red2blue, axis=0)

        # 判断蓝方智能体是否在最近的红方智能体的自爆范围内
        is_in_explode = distances_red2blue[nearest_id, np.arange(self.n_blues)] < self.explode_radius

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

        # 判断蓝方智能体是否在红方高价值区域内
        in_red_core = dists_to_center < self.red_core['radius']

        # 计算并更新在红方高价值区域内被打击的蓝方智能体数量
        self.attack_core_num = np.sum(in_red_core & self.blue_alives)
        self.attack_core_total += self.attack_core_num

        # 将在高价值区域内的蓝方智能体标记为死亡
        self.blue_alives[in_red_core] = False

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
        in_red_base = dists_to_center < self.red_base_radius

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

        # 软杀伤掩码：在威胁区域内停留超过最大时间的智能体将被标记为死亡
        soft_kill_mask = self.in_threat_zone_times >= self.max_in_threat_zone_time
        self.blue_alives[soft_kill_mask] = False

        return in_threat_zone, will_in_threat_zone

    def blue_self_destruction(self):
        """
        判断蓝方智能体是否需要自爆:
        规则：
        1. 如果蓝方存活智能体的数量超过80%且自爆范围内的红方智能体数量超过2,则自爆。
        2. 如果蓝方存活智能体的数量在60%和80%之间且自爆范围内红方智能体的数量超过3,则自爆。
        3. 其它情况下不攻击。
        """
        # 计算蓝方存活智能体的数量及其占比
        alive_count = np.sum(self.blue_alives)
        alive_percentage = alive_count / self.n_blues

        # 计算蓝方智能体与红方智能体之间的距离
        distances_blue2red = self.get_dist().T # 过滤掉了已经死亡的智能体

        # 蓝方智能体自爆范围内的红方智能体，形状为（n_blues, n_reds）
        red_in_explode_zone = distances_blue2red < self.explode_radius
        
        # 计算自爆范围内红方智能体的数量
        red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)
        
        # 蓝方智能体自爆的掩码初始化
        self_destruction_mask = np.zeros(self.n_blues, dtype=bool)

        # 根据存活比例判断是否自爆
        if alive_percentage >= 0.8:
            # 第一条规则：存活比例超过80%且自爆范围内的红方智能体数量超过1则自爆
            self_destruction_mask = red_counts_in_zone >= 1
        elif 0.6 < alive_percentage <= 0.8:
            # 第二条规则：如果蓝方存活智能体的数量在60%和80%之间且自爆范围内红方智能体的数量超过2,则自爆
            self_destruction_mask = red_counts_in_zone >= 2

        # 记录自爆的蓝方智能体数量和掩码，用于后续的处理和渲染
        self.blue_self_destruction_mask = self_destruction_mask
        self.blue_self_destruction_num = np.sum(self_destruction_mask)
        self.blue_self_destruction_total += self.blue_self_destruction_num

        # 存储自爆动作
        self.blue_action[self_destruction_mask, 2] = 2

        # 触发自爆的蓝方智能体标记为死亡
        self.blue_alives[self_destruction_mask] = False

        # 将自爆范围内的红方智能体标记为死亡，并记录自爆范围内的红方智能体
        red_explode_mask = np.any(red_in_explode_zone[self_destruction_mask], axis=0)
        self.red_explode_mask = red_explode_mask
        self.red_alives[red_explode_mask] = False
        
        # 更新被自爆炸死的红方智能体数量
        self.explode_red_num = np.sum(red_explode_mask)
        self.explode_red_total += self.explode_red_num
        
    def blue_step(self):
        """
        蓝方智能体的 step 函数，执行自爆、威胁区避让、自爆区域逃离等逻辑，并更新位置和方向。
        """
        # 执行蓝方自爆逻辑
        self.blue_self_destruction()
        
        # 计算激活的智能体组数，根据当前的时间步来决定
        if self._episode_steps <= self.interval:
            valid_num = self.group_sizes[0]
        elif self._episode_steps <= self.interval * 2:
            valid_num = sum(self.group_sizes[:3])
        else:
            valid_num = self.n_blues

        # 计算多波次的mask，决定哪些智能体在当前步内激活
        mask = np.zeros(self.n_blues, dtype=bool)
        mask[:valid_num] = True
        
        # 初始化目标点，默认为红方基地中心
        target_positions = np.tile(self.red_base_center, (self.n_blues, 1))

        # 判断智能体是否在警戒区或即将进入警戒区
        is_in_threat, will_in_threat = self.is_in_threat_zone()

        if self._episode_steps > self.interval * 2:
            # 绕过威胁区域
            target_positions = self.around_threat_zone(will_in_threat, target_positions)
            # 逃离威胁区域
            target_positions = self.flee_threat_zone(is_in_threat, target_positions)

        # 逃离自爆范围
        target_positions = self.flee_explode_zone(target_positions)

        self._update_blue_position_and_direction(mask, target_positions)

        # 判断蓝方智能体是否进入核心区域
        self.is_hit_core_zone()
        
    def _update_blue_position_and_direction(self, mask, target_positions):
        """
        基于运动学模型更新红方智能体的位置和方向。
        仅更新存活&激活的智能体。
        """
        # 仅对存活且激活的智能体进行更新
        active_mask = mask & self.blue_alives
        
        # 计算期望方向
        dx = target_positions[:, 0] - self.blue_positions[:, 0]
        dy = target_positions[:, 1] - self.blue_positions[:, 1]
        desired_directions = np.arctan2(dy, dx)
        
        # 计算角度差并规范化到 [-pi, pi]区间
        angles_diff = (desired_directions - self.blue_directions + np.pi) % (2 * np.pi) - np.pi

        # 限制转向角度
        max_turn = self.max_angular_vel * self.dt_time
        angles_diff = np.clip(angles_diff, -max_turn, max_turn)

        # 更新方向，仅更新有效的智能体
        self.blue_directions[active_mask] = (
            (self.blue_directions[active_mask] + angles_diff[active_mask] + np.pi) % (2 * np.pi) - np.pi
        )
        
        # 更新位置
        dx = self.blue_velocities[active_mask] * np.cos(self.blue_directions[active_mask]) * self.dt_time
        dy = self.blue_velocities[active_mask] * np.sin(self.blue_directions[active_mask]) * self.dt_time
        self.blue_positions[active_mask] += np.column_stack((dx, dy))
        
        # 存储数据
        self.blue_action[active_mask, 1] = angles_diff[active_mask]
        
    def get_dist_reward(self):
        """
        计算红方智能体靠近或远离目标区域的奖励。
        """
        # 获取红方目标矩形区域的边界
        x_min, x_max, y_min, y_max = self.red_target_area

        # 初始化奖励数组
        rewards = np.zeros(self.n_reds)

        # 获取红方智能体的当前位置
        red_x, red_y = self.red_positions[:, 0], self.red_positions[:, 1]

        # 判断智能体是否在目标区域内
        in_area = (x_min <= red_x) & (red_x <= x_max) & (y_min <= red_y) & (red_y <= y_max)
        
        # 筛选出不在目标区域但仍然存活的智能体
        out_area = ~in_area & self.red_alives 

        # 计算离区域边界最短的距离
        dx = np.maximum(x_min - red_x, 0, red_x - x_max)
        dy = np.maximum(y_min - red_y, 0, red_y - y_max)
        distance2area = dx + dy

        # 计算奖励,奖励基于距离的变化
        if self.dist2area is not None:
            rewards[out_area] = np.where(distance2area[out_area] < self.dist2area[out_area], 
                                        self.reward_near_area, self.reward_away_area)
            self.dist2area[out_area] = distance2area[out_area]
            self.dist2area[~out_area] = 0

        return rewards

    def get_reward(self):
        """
        计算红方智能体在当前时间步的总奖励。
        """
        # 动态时间惩罚
        time_penalty = self.reward_time * (1 + self._episode_steps / self.episode_limit)

        # 靠近目标区域的奖励以及远离目标区域的惩罚
        dist_rewards = self.get_dist_reward()
        dist_reward = np.sum(dist_rewards) 

        # 红方智能体被炸掉惩罚
        red_destroyed_penalty = self.reward_explode_red * self.explode_red_num * (
            1 + self.explode_red_total / self.n_reds)
        
        # 核心区域被打击惩罚
        core_hit_penalty = self.reward_attack_core * self.attack_core_num * (
            1 + self.attack_core_total / self.max_attack_core_num)
        
        # 每个时间步杀伤蓝方数量奖励
        kill_num = self.collide_blue_num + self.explode_blue_num
        self.kill_total += kill_num
        kill_reward = self.reward_kill_blue * (1 + self.kill_total // 10) * kill_num * (
            1 + self.kill_total / self.n_blues)

        # 计算总奖励
        total_reward = (time_penalty + dist_reward + red_destroyed_penalty + core_hit_penalty + kill_reward)
        
        return total_reward
    
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
        # 渲染存活数量和时间步等文本信息
        time_text = self.font.render(
            f'Episode: {self._episode_count} Time Step: {self._episode_steps} Win count: {self.battles_won}', 
            True, 
            (0, 0, 0)
        )
        red_text = self.font.render(
            f'Red alives: {sum(self.red_alives)} Red explode: {self.explode_blue_total} Red collide: {self.collide_blue_total}', 
            True, 
            (255, 0, 0)
        )
        blue_text = self.font.render(
            f'Blue alives: {sum(self.blue_alives)} Blue explode: {self.explode_red_total} Blue hit: {self.attack_core_total}', 
            True, 
            (0, 0, 255)
        )
        
        self.screen.blit(time_text, (10, 10))
        self.screen.blit(red_text, (10, 50))
        self.screen.blit(blue_text, (10, 90))
    
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


if __name__ == "__main__":

    args = Arg()

    env = DefenseEnv(args)

    env.reset()
    
    import time
    for i in range(100):
        start = time.time()
        actions = env.random_policy_red()
        env.step(actions)
        env.render()
        print(f'[frame: {i}]---[Time: {time.time() - start}]')
    
    env.close()

    # indices, distances = env.find_nearest_grid()
