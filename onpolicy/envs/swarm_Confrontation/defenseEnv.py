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
    
from scipy.spatial import distance

image_dir = "/home/ubuntu/sunfeng/MARL/on-policy/onpolicy/envs/swarm_Confrontation/"
os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

class DefenseEnv(BaseEnv):
    def __init__(self, args):
        super(DefenseEnv, self).__init__(args)

        # red base
        self.red_core = {
            'center': np.array([2250.0, 0.0]),
            'radius': 25.0
        }

        self.red_base = {
            'center': np.array([2250.0, 0.0]),
            'radius': 1250.0
        }

        self.red_lines = np.array([
            [[1366.0,  884.0], [1750.0,  500.0]],
            [[1750.0,  500.0], [1750.0, -500.0]],
            [[1750.0, -500.0], [1366.0, -884.0]],
            [[3134.0,  884.0], [2750.0,  500.0]],
            [[2750.0,  500.0], [2750.0, -500.0]],
            [[2750.0, -500.0], [3134.0, -884.0]],
        ])

        # 鼓励红方智能体往基地附近飞
        self.red_target_area = [1750, 2750, -1000, 1000]

        self.red_lines_vec = self.red_lines[:, 1, :] - self.red_lines[:, 0, :]
        self.red_lines_len = np.linalg.norm(self.red_lines_vec, axis=1)
        self.red_lines_unitvec = self.red_lines_vec / self.red_lines_len[:, np.newaxis]

        self.red_base_center = self.red_base['center']
        self.red_base_radius = self.red_base['radius']
        
        # 中间正方形区域
        self.red_square_size = 1000.0 / 2
        
        # 左侧威胁区
        self.left_sector_pos1 = np.array([1366.0, 884.0])
        self.left_sector_pos2 = np.array([1366.0, -884.0])
        
        self.left_sector_theta1, self.left_sector_theta2 = calculate_sector_theta(
            self.left_sector_pos1, self.left_sector_pos2, self.red_base_center)
        self.left_threat_x = self.red_base_center[0] - self.red_square_size

        # 右侧威胁区
        self.right_sector_pos1 = np.array([3134.0, -884.0])
        self.right_sector_pos2 = np.array([3134.0, 884.0])
        
        self.right_sector_theta1, self.right_sector_theta2 = calculate_sector_theta(
            self.right_sector_pos1, self.right_sector_pos2, self.red_base_center)
        self.right_threat_x = self.red_base_center[0] + self.red_square_size

        # blue base
        self.blue_bases = [
            {'center': np.array([1500.0,  1500.0]), 'radius': 500.0},     # 上右
            {'center': np.array([1500.0, -1500.0]), 'radius': 500.0},     # 下右
            {'center': np.array([ 500.0,  1500.0]), 'radius': 500.0},     # 上左
            {'center': np.array([ 500.0, -1500.0]), 'radius': 500.0},     # 下左
        ]

        # max in threat zone time
        self.max_in_threat_zone_time = 10

        self.interval = 5

        # 基地的最大攻击次数
        self.max_attack_core_num = 40

        # Reward
        self.reward_time = -0.1             # 每存活一个时间步的惩罚
        self.reward_explode_red = -5        # 被炸掉的惩罚
        self.reward_attack_core = -20       # 核心区域被攻击的惩罚
        self.reward_win = 3000              # 获胜奖励
        self.reward_defeat = 0              # 失败奖励
        self.reward_kill_blue = 5           # 每个时间步杀死蓝方奖励
        self.reward_near_area = 0.2         # 靠近目标区域的奖励
        self.reward_away_area = -0.2

        # 奖励值的统计信息
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_alpha = 0.1  # 平滑系数

    def reset(self):
        super().reset()

        self.in_threat_zone_times = np.zeros(self.n_blues)

        self.attack_core_num = 0            # 每个时间步红方核心区域被攻击的次数
        self.soft_kill_num = 0              # 每个时间步软杀伤蓝方数量

        self.attack_core_total = 0          # 当前回合红方高价值区域被打击的总次数
        self.kill_total = 0

        self.dist2area = None               # 红方智能体与目标区域的距离 

        local_obs = self.get_obs()
        global_state = [self.get_state()] * self.n_reds
        available_actions = self.get_avail_actions()

        return local_obs, global_state, available_actions

    def step(self, actions):
        # Get red actions
        at = self.acc_actions[actions[:, 0]]
        pt = self.heading_actions[actions[:, 1]]
        attack_t = actions[:, 2]

        explode_mask = (attack_t == 1)
        collide_mask = (attack_t == 2)
        soft_kill_mask = (attack_t == 3)

        # Perfor attack actions
        self.red_explode(explode_mask)
        pt = self.red_collide(collide_mask, pt)
            
        # Perform move actions
        self.red_directions += pt * self.max_angular_vel
        self.red_directions = (self.red_directions + np.pi) % (2 * np.pi) - np.pi
        self.red_velocities += at * self.dt_time
        self.red_velocities = np.clip(self.red_velocities, self.red_min_vel, self.red_max_vel)
        self.red_positions += np.column_stack((self.red_velocities * np.cos(self.red_directions),
                                               self.red_velocities * np.sin(self.red_directions))) * self.dt_time
        
        self.blue_step()
        self.merge_state()

        # self.check_boundaries()

        # Update step counter
        self._total_steps += 1
        self._episode_steps += 1

        # Update terminated flag and reward
        terminated, win, res = self.get_result()
        bad_transition = False

        if terminated:
            self.battles_game += 1
            self._episode_count += 1    
        
        if win:
            self.battles_won += 1
            self.win_counted = True
        else:
            self.defeat_counted = True

        if self._episode_steps >= self.episode_limit:
            self.timeouts += 1
            if not win:
                bad_transition = True

        info = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            'bad_transition': bad_transition,
            'explode_ratio': self.red_self_destruction_total / self.n_reds, # 红方主动自爆的比例
            'be_exploded_ratio': self.explode_red_total / self.n_reds, # 红方被自爆的比例
            'invalid_explode_ratio': self.invalid_explode_red_total / self.n_reds, # 红方无效自爆的比例
            'collide_ratio': self.collide_blue_total / self.n_reds,   # 红方主动撞击的比例
            'be_collided_ratio': self.collide_red_total / self.n_reds, # 红方被撞击的比例
            'kill_num': self.kill_total, # 红方毁伤蓝方的总数
            'hit_core_num': self.attack_core_total, # 高价值区域被打击的次数
            'explode_ratio_blue': self.blue_self_destruction_total / self.n_blues, # 蓝方主动自爆的比例
            'scout_core_ratio': 0, # 高价值区域被侦察的比例
            'scout_comm_ratio': 0, # 普通区域被侦察的比例
            'episode_length': self._episode_steps, # 轨迹长度
            'won': self.win_counted,
            "other": res
        }

        local_obs = self.get_obs()
        global_state = [self.get_state()] * self.n_reds
        reward = self.get_reward(win)

        rewards = [[reward]] * self.n_reds

        dones = np.zeros((self.n_reds), dtype=bool)
        dones = np.where(terminated, True, ~self.red_alives)

        infos = [info] * self.n_reds
        
        available_actions = self.get_avail_actions()

        return local_obs, global_state, rewards, dones, infos, available_actions

    def flee_explode_zone(self, target_positions):
        """
        判断蓝方智能体是否在距离最近的红方智能体的自爆范围内，
        如果该红方智能体的自爆范围内还有其它蓝方智能体，
        那么该蓝方智能体需要逃离该红方智能体的自爆范围。
        """    
        # 计算红方智能体与蓝方智能体之间的距离
        distances_red2blue = distance.cdist(self.red_positions, self.blue_positions, 'euclidean')

        # 创建有效性掩码，只考虑存活的红方和蓝方智能体之间的距离
        valid_mask = self.red_alives[:, np.newaxis] & self.blue_alives[np.newaxis, :]

        # 将无效的距离设置为无限大
        distances_red2blue = np.where(valid_mask, distances_red2blue, np.inf)
        
        # 转置距离矩阵，以便计算每个蓝方智能体到最近的红方智能体的距离
        distances_blue2red = distances_red2blue.T

        # 找到每个蓝方智能体距离最近的红方智能体的索引
        nearest_id = np.argmin(distances_blue2red, axis=1)

        # 判断蓝方智能体是否在最近的红方智能体的自爆范围内
        is_in_explode = distances_red2blue[nearest_id, :] < self.explode_radius

        # 判断距离最近的红方智能体的自爆范围内是否有超过2个蓝方智能体
        flee_or_not = np.sum(is_in_explode, axis=1) > 1

        # 计算逃离方向
        flee_directions = self.blue_positions - self.red_positions[nearest_id, :]
        flee_angles = np.arctan2(flee_directions[:, 1], flee_directions[:, 0])      

        # 计算逃离位置偏移
        dx = np.cos(flee_angles)
        dy = np.sin(flee_angles)
        offsets = np.stack([dx, dy], axis=1) * self.explode_radius

        # 计算新的目标位置
        targets = self.red_positions[nearest_id, :] + offsets

        # 更新需要逃离的蓝方智能体的目标位置
        target_positions[flee_or_not] = targets[flee_or_not]

        return target_positions

    def flee_threat_zone(self, is_in_threat, target_positions):
        """
        针对当前已经在警戒区的智能体，选择最近的边界上的最近点作为目标点，从而逃离警戒区
        """

        # 计算智能体当前位置到线段起点的向量
        pos_vec = self.blue_positions[:, np.newaxis, :] - self.red_lines[:, 0, :]
        
        # 计算点向量的单位向量
        pos_unitvec = pos_vec / self.red_lines_len[:, np.newaxis]

        # 计算每个智能体位置在每条线段上的投影长度 t
        t = np.einsum('nij,ij->ni', pos_unitvec, self.red_lines_unitvec)

        # 将 t 限制在 [0, 1] 范围内，确保投影点在线段上
        t = np.clip(t, 0.0, 1.0)

        # 计算线段上距离智能体最近的点的坐标
        nearest =  self.red_lines[:, 0, :] + t[:, :, np.newaxis] * self.red_lines_vec[np.newaxis, :, :]

        # 计算智能体当前位置到最近点的距离
        distance = np.linalg.norm(self.blue_positions[:, np.newaxis, :] - nearest, axis=2)

        # 找到每个智能体距离最近的线段的索引
        nearest_id = np.argmin(distance, axis=1)

        # 获取每个智能体最近的目标点
        nearest_target = nearest[np.arange(self.n_blues), nearest_id]

        # 更新在警戒区内的智能体的目标位置
        target_positions[is_in_threat] = nearest_target[is_in_threat]

        return target_positions
    
    def around_threat_zone(self, will_in_threat, target_positions):
        """
        给即将进入警戒区的智能体分配一个新的目标点，使他们绕开警戒区
        """
        # 生成一个随机角度：默认从北侧突防通道口进入
        target_angles = np.random.uniform(self.right_sector_theta2, self.left_sector_theta1, size=self.n_blues)
        positions_y = self.blue_positions[:, 1]
        # 如果智能体在南侧，反转角度，从南侧突防通道口进入
        target_angles = np.where(positions_y > 0, target_angles, -target_angles)

        # 计算目标位置的偏移量
        dx = np.cos(target_angles)
        dy = np.sin(target_angles)
        offsets = np.stack([dx, dy], axis=1) * self.red_base_radius

        # 计算新的目标位置
        new_targets = self.red_base_center + offsets

        # 更新威胁区域外智能体的目标位置
        target_positions[will_in_threat] = new_targets[will_in_threat]

        return target_positions
    
    def is_hit_core_zone(self):
        # 判断智能体是否在红方高价值区域内
        dists_to_center = np.linalg.norm(self.blue_positions - self.red_core['center'], axis=1)
        in_red_core = dists_to_center < self.red_core['radius']

        self.attack_core_num = np.sum(in_red_core & self.blue_alives)
        self.attack_core_total += self.attack_core_num

        self.blue_alives[in_red_core] = False

    def is_in_threat_zone(self):
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
        angles = np.mod(angles + 2*np.pi, 2*np.pi)

        # 左边扇形区域角度判断
        left_sector_angle_range = np.logical_or(
            (self.left_sector_theta1 <= self.left_sector_theta2) & (angles > self.left_sector_theta1) & (angles < self.left_sector_theta2),
            (self.left_sector_theta1 > self.left_sector_theta2) & ((angles > self.left_sector_theta1) | (angles < self.left_sector_theta2))
        )

        # 右边扇形区域角度判断
        right_sector_angle_range = np.logical_or(
            (self.right_sector_theta1 <= self.right_sector_theta2) & (angles > self.right_sector_theta1) & (angles < self.right_sector_theta2),
            (self.right_sector_theta1 > self.right_sector_theta2) & ((angles > self.right_sector_theta1) | (angles < self.right_sector_theta2))
        )

        # 当前在威胁区域内的：在 left/right 两个扇形区域的角度范围内 且在红方基地的范围内 且x轴坐标在left_threat_x左侧/right_threat_x右侧
        in_threat_zone = (left_sector_angle_range & x_in_left & in_red_base) | (right_sector_angle_range & x_in_right & in_red_base)
        # 将会在威胁区域内的：在 left/right 两个扇形区域的角度范围内 且在红方基地的范围外
        # will_in_threat_zone = (left_sector_angle_range & ~in_red_base) | (right_sector_angle_range & ~in_red_base)
        will_in_threat_zone = ~in_red_base

        self.in_threat_zone_times[in_threat_zone] += 1
        self.in_threat_zone_times[~in_threat_zone] = 0

        # 软杀伤掩码
        soft_kill_mask = self.in_threat_zone_times >= self.max_in_threat_zone_time
        self.soft_kill_num = np.sum(soft_kill_mask & self.blue_alives)
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
        # 计算蓝方存活智能体的数量
        alive_count = np.sum(self.blue_alives)
        alive_percentage = alive_count / self.n_blues

        # 计算每个蓝方智能体与每个红方智能体的距离
        distances_blue2red = distance.cdist(self.blue_positions, self.red_positions, 'euclidean')

        # 蓝方智能体自爆范围内的红方智能体
        red_in_explode_zone = (distances_blue2red < self.explode_radius) & self.red_alives

        # 蓝方智能体自爆的掩码
        self_destruction_mask = np.zeros(self.n_blues, dtype=bool)

        if alive_percentage >= 0.8:
            # 第一条规则：存活比例超过80%且自爆范围内的红方智能体数量超过2则自爆
            red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)
            self_destruction_mask = red_counts_in_zone >= 2
        
        elif 0.6 < alive_percentage <= 0.8:
            # 第二条规则：如果蓝方存活智能体的数量在60%和80%之间且自爆范围内红方智能体的数量超过3,则自爆
            red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)
            self_destruction_mask = red_counts_in_zone >= 3
        
        self_destruction_mask &= self.blue_alives

        # 记录蓝方自爆的智能体，用作渲染
        self.blue_self_destruction_mask = self_destruction_mask
        self.blue_self_destruction_num = np.sum(self_destruction_mask)
        self.blue_self_destruction_total += self.blue_self_destruction_num

        # 触发自爆的蓝方智能体将被标记为死亡
        self.blue_alives[self_destruction_mask] = False

        # 将自爆范围内的红方智能体标记为死亡
        red_explode_mask = np.any(red_in_explode_zone[self_destruction_mask], axis=0)

        # 记录自爆范围内的红方智能体，用作渲染
        self.red_explode_mask = red_explode_mask & self.red_alives

        self.explode_red_num = np.sum(red_explode_mask & self.red_alives)
        self.explode_red_total += self.explode_red_num

        self.red_alives[red_explode_mask] = False
        
    def blue_step(self):
        # 执行自爆
        self.blue_self_destruction()

        # 计算多波次的mask，决定哪些智能体在当前步内激活
        mask = np.zeros(self.n_blues, dtype=bool)

        # 根据当前的步骤来决定激活的智能体组
        if self._episode_steps <= self.interval:
            valid_num = self.group_sizes[0]
        elif self.interval < self._episode_steps <= self.interval * 2:
            valid_num = sum(self.group_sizes[:3])
        else:
            valid_num = self.n_blues

        mask[:valid_num] = True
        
        # 初始化每个智能体的目标点坐标，默认为红方基地中心
        target_positions = np.tile(self.red_base_center, (self.n_blues, 1))

        # 判断智能体是否在警戒区或即将进入警戒区
        is_in_threat, will_in_threat = self.is_in_threat_zone()

        if self._episode_steps > self.interval * 2:
            # 对于即将进入警戒区的智能体，更新其目标位置以绕飞警戒区
            target_positions = self.around_threat_zone(will_in_threat, target_positions)
            # 对于已经在警戒区的智能体，更新其目标位置以逃离警戒区
            target_positions = self.flee_threat_zone(is_in_threat, target_positions)

        # 对于在红方自爆范围内的智能体，更新其目标位置以逃离自爆范围
        target_positions = self.flee_explode_zone(target_positions)

        # 计算期望方向
        desired_directions = np.arctan2(target_positions[:, 1] - self.blue_positions[:, 1],
                                        target_positions[:, 0] - self.blue_positions[:, 0])
        
        # 计算当前方向到期望方向的角度
        angles_diff = desired_directions - self.blue_directions

        # 将角度差规范化到[-pi,pi] 区间内
        angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi

        # 确保转向角度不超过最大角速度
        angles_diff = np.clip(angles_diff, -self.max_angular_vel, self.max_angular_vel)

        # 更新当前方向，受激活mask限制
        self.blue_directions[mask] += angles_diff[mask]
        self.blue_directions = (self.blue_directions + np.pi) % (2 * np.pi) - np.pi

        # 更新智能体位置，受激活mask限制
        self.blue_positions[mask] += (np.column_stack((self.blue_velocities * np.cos(self.blue_directions),
                                                       self.blue_velocities * np.sin(self.blue_directions))) * self.dt_time)[mask]

        # 判断蓝方智能体是否进入核心区域
        self.is_hit_core_zone()

    def distribute_red_agents(self):
        n_out_bases = int(self.n_reds * np.random.uniform(0.1, 0.2))
        n_in_bases = int(self.n_reds - n_out_bases)

        return n_in_bases, n_out_bases
    
    def generate_red_positions(self):
        n_in_bases, n_out_bases = self.distribute_red_agents()

        # 使用 numpy 生成随机角度和半径
        angles = np.random.uniform(0, 2 * np.pi, n_in_bases)
        radii = self.red_base_radius * np.sqrt(np.random.uniform(0, 1, n_in_bases))

        # 计算智能体的位置
        x = self.red_base_center[0] + radii * np.cos(angles)
        y = self.red_base_center[1] + radii * np.sin(angles)
        in_base_positions = np.vstack([x, y]).T

        out_base_positions = (np.random.rand(n_out_bases, 2) - 0.5) * np.array([self.size_x, self.size_y])

        positions = np.vstack([in_base_positions, out_base_positions])
        directions = (np.random.rand(self.n_reds) - 0.5) * 2 * np.pi

        return positions, directions

    def distribute_blue_agents(self):
        # 随机分成 n_groups 组, 总和为 n_agents
        n_agents = self.n_blues
        n_groups = len(self.blue_bases) + 1
        group_sizes = np.random.multinomial(n_agents, np.ones(n_groups) / n_groups)

        return group_sizes

    def generate_blue_positions(self):
        group_sizes = self.distribute_blue_agents()
        agent_positions = []

        blue_bases = [self.red_base] + self.blue_bases
        self.group_sizes = group_sizes

        # Initialize agent positions in each group
        for group_idx, group_size in enumerate(group_sizes):
            center = blue_bases[group_idx]['center']
            radius = blue_bases[group_idx]['radius']
            
            # 使用 numpy 生成随机角度和半径
            if group_idx == 0:
                angles = np.random.uniform(0, 2 * np.pi, group_size)
                radii = radius + 20
            else:
                angles = np.random.uniform(0, 2 * np.pi, group_size)
                radii = radius * np.sqrt(np.random.uniform(0, 1, group_size))

            # 计算智能体的位置
            x = center[0] + radii * np.cos(angles)
            y = center[1] + radii * np.sin(angles)
            positions = np.vstack([x, y]).T

            agent_positions.append(positions)
        
        agent_positions = np.vstack(agent_positions)
        agent_directions = np.arctan2(self.red_base_center[1] - agent_positions[:, 1], 
                                      self.red_base_center[0] - agent_positions[:, 0])
        agent_directions += np.random.uniform(-np.pi/18, np.pi/18, self.n_blues)
            
        return agent_positions, agent_directions

    def get_reward_old(self, win):
        reward = self.reward_time

        num = np.array([self.explode_red_num, self.explode_blue_num, self.invalid_explode_num, self.collide_success_num, self.attack_core_num, self.red_out_of_bounds_num])
        value = np.array([self.reward_explode_red, self.reward_explode_blue, self.reward_explode_invalid, self.reward_collied, self.reward_attack_core, self.reward_out_of_bound])

        reward += np.sum(num * value)

        win_reward = self.reward_win if win else self.reward_defeat

        reward += win_reward

        return reward
    
    def get_reward_0705(self, win=False):

        # 动态时间惩罚
        time_penalty = self.reward_time * (1 + self._episode_steps / self.episode_limit)

        # 红方智能体被炸掉惩罚
        red_destroyed_penalty = self.reward_explode_red * self.explode_red_num * (
            1 + self.explode_red_total / self.n_reds)
        
        # 核心区域被打击惩罚
        core_hit_penalty = self.reward_attack_core * self.attack_core_num * (
            1 + self.attack_core_total / self.max_attack_core_num)
        
        # 每个时间步杀伤蓝方数量奖励
        kill_num = self.collide_blue_num + self.explode_blue_num
        self.kill_total += kill_num
        kill_reward = (self.reward_kill_blue * (1 + self.kill_total // 20)) * kill_num * (1 + self.kill_total / self.n_blues)

        # 获胜奖励
        win_reward = self.reward_win if win else self.reward_defeat

        total_reward = (win_reward + time_penalty + red_destroyed_penalty + core_hit_penalty + kill_reward)
        
        return total_reward

    def get_dist_reward_old(self):
        """
        计算智能体在目标区域内的奖励和惩罚，并根据地图大小适当缩放距离惩罚。
        
        参数:
        positions (numpy.ndarray): 形状为 (n, 2) 的数组，包含每个智能体的位置。
        target_area (tuple): 目标区域的边界 (x_min, x_max, y_min, y_max)。
        map_width (float): 地图的宽度。
        map_height (float): 地图的高度。
        
        返回:
        numpy.ndarray: 包含每个智能体的奖励的数组。
        """

        # 获取红方目标矩形区域的边界
        x_min, x_max, y_min, y_max = self.red_target_area

        # 初始化奖励数组
        rewards = np.zeros(self.n_reds)

        # 判断智能体是否在目标区域内
        in_area = (x_min <= self.red_positions[:, 0]) & (self.red_positions[:, 0] <= x_max) & \
              (y_min <= self.red_positions[:, 1]) & (self.red_positions[:, 1] <= y_max)

        # 进入区域的智能体奖励
        rewards[in_area] += self.reward_near_core

        # 计算离区域边界最短的距离
        dx = np.maximum(x_min - self.red_positions[:, 0], 0, self.red_positions[:, 0] - x_max)
        dy = np.maximum(y_min - self.red_positions[:, 1], 0, self.red_positions[:, 1] - y_max)
        distance_penalty = dx + dy
            

        # 地图对角线距离
        map_diagonal = np.sqrt(self.size_x**2 + self.size_y**2)

        # 缩放距离惩罚
        scaled_distance_penalty = distance_penalty / map_diagonal

        # 给不在区域内的智能体分配距离惩罚
        rewards[~in_area] -= scaled_distance_penalty[~in_area]

        return rewards
    
    def get_dist_reward(self):
        # 获取红方目标矩形区域的边界
        x_min, x_max, y_min, y_max = self.red_target_area

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

        # 计算奖励
        if self.dist2area is not None:
            rewards[out_area] = np.where(distance2area[out_area] < self.dist2area[out_area], 
                                        self.reward_near_area, self.reward_away_area)
            self.dist2area[out_area] = distance2area[out_area]
            self.dist2area[~out_area] = 0

        return rewards

    
    def get_dist_reward(self):
        # 获取红方目标矩形区域的边界
        x_min, x_max, y_min, y_max = self.red_target_area

        # 初始化奖励数组
        rewards = np.zeros(self.n_reds)

        # 判断智能体是否在目标区域内
        in_area = (x_min <= self.red_positions[:, 0]) & (self.red_positions[:, 0] <= x_max) & \
              (y_min <= self.red_positions[:, 1]) & (self.red_positions[:, 1] <= y_max)
        
        out_area = ~in_area & self.red_alives 

        # 计算离区域边界最短的距离
        dx = np.maximum(x_min - self.red_positions[:, 0], 0, self.red_positions[:, 0] - x_max)
        dy = np.maximum(y_min - self.red_positions[:, 1], 0, self.red_positions[:, 1] - y_max)
        distance2area = dx + dy

        if self.dist2area is not None:
            rewards[out_area] = np.where(distance2area < self.dist2area, self.reward_near_area, self.reward_away_area)[out_area]
            self.dist2area = distance2area.copy()

        return rewards

    def get_reward(self, win=False):

        # 动态时间惩罚
        time_penalty = self.reward_time * (1 + self._episode_steps / self.episode_limit)

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
        kill_reward = (self.reward_kill_blue * (1 + self.kill_total // 10)) * kill_num * (1 + self.kill_total / self.n_blues)

        # 获胜奖励
        # win_reward = 0

        total_reward = (time_penalty + dist_reward + red_destroyed_penalty + core_hit_penalty + kill_reward)
        
        return total_reward
    
    
    def get_result(self):
        # 计算存活的智能体数量
        n_red_alive = np.sum(self.red_alives)
        n_blue_alive = np.sum(self.blue_alives)

        # 初始化对抗结束标志和获胜方标志
        terminated = False
        win = False
        info = ""

        # 检查终止条件
        # if self._episode_steps >= self.episode_limit:
        #     terminated = True
        #     # win = self.total_hit_core_num < self.max_attack_core_num or n_blue_alive == 0
        #     # 至对抗结束，基地未被摧毁
        #     win = self.total_hit_core_num < self.max_attack_core_num
        #     if win:
        #         info = "Time limit reached. Red based were not destroyed."
        #     else:
        #         info = "Time limit reached. Red based were destroyed."
        # # elif n_red_alive == 0:
        # #     terminated = True
        # #     win = False
        # #     info = "All red units destroyed. Blue wins."
        # elif n_blue_alive == 0:
        #     terminated = True
        #     win = True
        #     info = "All blue units destroyed. Red wins."
        # elif self.total_hit_core_num >= self.max_attack_core_num:
        #     terminated = True
        #     win = False
        #     info = "Red base detroyed. Blue wins."
        # # # elif n_blue_alive + self.total_hit_core_num < self.max_attack_core_num:
        # #     terminated = True
        # #     win = True
        # #     info = "Remaining blue units insufficient to destroy red base. Red wins."
            
        if self.attack_core_total >= self.max_attack_core_num:
            terminated = True
            win = False
            info = "[Defeat] Base destroyed."
        elif n_blue_alive == 0:
            terminated = True
            win = True
            info = "[Win] All blue dead."
        elif self._episode_steps >= self.episode_limit:
            terminated = True
            win = True
            info = '[Win] Time out.'

        return terminated, win, info


    def init_positions(self):
        
        red_positions, red_directions = self.generate_red_positions()
        blue_positions, blue_directions = self.generate_blue_positions()

        positions = np.vstack([red_positions, blue_positions])
        directions = np.hstack([red_directions, blue_directions])
        velocities = np.hstack([np.ones(self.n_reds) * self.red_max_vel, np.ones(self.n_blues) * self.blue_max_vel])

        return positions, directions, velocities
    
    def transform_lines(self):
        # 将世界坐标转换为屏幕坐标
        new_center = np.array([-self.size_x / 2, self.size_y/2])
        new_dir = np.array([1, -1])

        self.transformed_lines = ((self.red_lines - new_center) * new_dir * self.scale_factor).astype(int)

    def transform_circles(self):
        self.transformed_circles_center = []
        self.transformed_circles_radius = []

        circles = [self.red_core, self.red_base] + self.blue_bases

        for circle in circles:
            self.transformed_circles_center.append(self.transform_position(circle['center']))
            self.transformed_circles_radius.append(circle['radius'] * self.scale_factor)
            
        
    def render(self, mode='human'):

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            red_plane_img = pygame.image.load(f'{image_dir}/png/red_plane_s.png').convert_alpha()
            blue_plane_img = pygame.image.load(f'{image_dir}/png/blue_plane_s.png').convert_alpha()

            # 缩放飞机贴图
            scale_factor = 0.2  # 调整缩放比例
            self.red_plane_img = pygame.transform.scale(red_plane_img, (int(red_plane_img.get_width() * scale_factor), 
                                                                        int(red_plane_img.get_height() * scale_factor)))
            self.blue_plane_img = pygame.transform.scale(blue_plane_img, (int(blue_plane_img.get_width() * scale_factor), 
                                                                          int(blue_plane_img.get_height() * scale_factor)))

            pygame.display.set_caption("Swarm Confrontation")

            self.transform_lines()
            self.transform_circles()

            self.num_circles = len(self.blue_bases) + 2

            # 初始化字体
            self.font = pygame.font.SysFont(None, 36)

        self.screen.fill((255, 255, 255))
        self.transform_positions()
        angles = np.degrees(self.directions)

        # 渲染基地
        for i in range(self.num_circles):
            width = 0 if i == 0 else 2
            color = (255, 0, 0) if i <= 1 else (0, 0, 255)
            pygame.draw.circle(self.screen, color, self.transformed_circles_center[i], self.transformed_circles_radius[i], width=width)

        # 渲染突防通道
        for line in self.transformed_lines:
            pygame.draw.line(self.screen, (255, 0, 0), (line[0,0], line[0,1]), (line[1,0], line[1,1]), 2)
        
        # 渲染飞机
        for i in range(self.n_agents):
            if self.alives[i]:
                image = self.red_plane_img if i < self.n_reds else self.blue_plane_img
                # rotated_img = pygame.transform.rotate(image, -angles[i])
                rotated_img = pygame.transform.rotate(image, angles[i])
                new_rect = rotated_img.get_rect(center=self.transformed_positions[i])
                self.screen.blit(rotated_img, new_rect)

        # 计算存活的智能体数量
        red_alive = sum(self.red_alives)
        blue_alive = sum(self.blue_alives)

        # 渲染存活数量文本
        time_text = self.font.render(f'Episode: {self._episode_count} Time Step: {self._episode_steps} Win count: {self.battles_won}', True, (0, 0, 0))
        red_text = self.font.render(f'Red alives: {red_alive} Red explode: {self.explode_blue_total} Red collide: {self.collide_blue_total}', True, (0, 0, 0))
        blue_text = self.font.render(f'Blue alives: {blue_alive} Blue explode: {self.explode_red_total} Blue hit: {self.attack_core_total}', True, (0, 0, 0))
        
        self.screen.blit(time_text, (10, 10))
        self.screen.blit(red_text, (10, 50))
        self.screen.blit(blue_text, (10, 90))

        # 渲染自爆效果
        self.render_explode()

        # 渲染碰撞效果
        self.render_collide()

        pygame.display.flip()

        frame_dir = f"{image_dir}/Defense_frames/"
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        frame_path = os.path.join(frame_dir, f"frame_{self._total_steps:06d}.png")

        pygame.image.save(self.screen, frame_path)

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


if __name__ == "__main__":

    args = Arg()

    env = DefenseEnv(args)

    env.reset()
    
    import time
    for i in range(50):
        start = time.time()
        actions = env.scripted_policy_red()
        env.step(actions)
        env.render()
        print(f'[frame: {i}]---[Time: {time.time() - start}]')

    # indices, distances = env.find_nearest_grid()
