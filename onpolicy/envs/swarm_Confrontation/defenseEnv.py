# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-17
# @Description: Implementation of Agent Class

import os
import pygame
import imageio
import numpy as np

from baseEnv import BaseEnv
from scipy.spatial import distance
from gym.spaces import MultiDiscrete


os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

class DefenseEnv(BaseEnv):
    def __init__(self):
        super(DefenseEnv, self).__init__()
        
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

        self.red_lines_vec = self.red_lines[:, 1, :] - self.red_lines[:, 0, :]
        self.red_lines_len = np.linalg.norm(self.red_lines_vec, axis=1)
        self.red_lines_unitvec = self.red_lines_vec / self.red_lines_len[:, np.newaxis]

        self.red_base_center = self.red_base['center']
        self.red_base_radius = self.red_base['radius']
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
            {'center': np.array([1500.0, 1500.0]), 'radius': 500.0},    # 上右
            {'center': np.array([1500.0, -1500.0]), 'radius': 500.0},   # 下右
            {'center': np.array([500.0, 1500.0]), 'radius': 500.0},     # 上左
            {'center': np.array([500.0, -1500.0]), 'radius': 500.0},    # 下左
        ]

        # max in threat zone time
        self.max_in_threat_zone_time = 10

        self.interval = 5

        # 基地的最大攻击次数
        self.max_attack_core_num = 40

        # Reward
        self.reward_time = -0.1             # 每存活一个时间步的惩罚
        self.reward_explode_red = -10       # 被炸掉的惩罚
        self.reward_explode_blue = 10       # 炸掉蓝方的奖励
        self.reward_explode_invalid = -15   # 无效自爆惩罚
        self.reward_attack_core = -20       # 核心区域被攻击的惩罚
        self.reward_collied = 10            # 撞击成功的奖励
        self.reward_win = 100               # 获胜奖励
        self.reward_defeat = 0              # 失败奖励
        self.reward_out_of_bound = -5       # 出界惩罚

        # 定义动作空间 （多离线动作空间）
        self.action_space = [MultiDiscrete([self.acc_action_num,
                                            self.heading_action_num,
                                            self.attack_action_num])] * self.n_reds
        
        # 定义观测空间
        self.observation_space = [self.get_obs_size()] * self.n_reds

        # 定义状态空间
        self.share_observation_space = [self.get_state_size()] * self.n_reds

    def reset(self):
        obs = super().reset()

        self.in_threat_zone_times = np.zeros(self.n_blues)

        self.explode_red_num = 0            # 每个时间步被炸毁的红方智能体数量
        self.explode_blue_num = 0           # 每个时间步被炸毁的蓝方智能体数量
        self.invalid_explode_num = 0        # 每个时间步无效自爆的红方智能体数量
        self.collide_success_num = 0        # 每个时间撞击成功的红方智能体数量
        self.attack_core_num = 0            # 每个时间步红方核心区域被攻击的次数

        self.total_hit_core_num = 0         # 当前回合红方高价值区域被打击的总次数

        return obs

    def red_explode(self, explode_mask):
        # 更新 explode_mask， 排除已经死掉的智能体
        valid_explode_mask = explode_mask & self.red_alives

        # 计算每个红方智能体与每个蓝方智能体之间的距离
        distances_red2blue = distance.cdist(self.red_positions, self.blue_positions, 'euclidean')
        
        # 红方智能体自爆范围内的蓝方智能体
        blue_in_explode_zone = distances_red2blue < self.explode_radius

        # 统计无效自爆的智能体数量
        valid_blue_mask = blue_in_explode_zone & self.blue_alives
        self.invalid_explode_num = np.sum(np.sum(valid_blue_mask[valid_explode_mask], axis=1) == 0)

        # 触发自爆的红方智能体将被标记为死亡
        self.red_alives[valid_explode_mask] = False

        # 将自爆范围内的蓝方智能体标记为死亡，并统计有效毁伤的蓝方智能体数量
        blue_explode_mask = np.any(blue_in_explode_zone[valid_explode_mask], axis=0)
        self.explode_blue_num = np.sum(blue_explode_mask & self.blue_alives)
        self.blue_alives[blue_explode_mask] = False


    def red_collide(self, collide_mask, pt):
        """
        红方智能体与其目标的碰撞
        """
        # 计算红方智能体到蓝方智能体的方向向量
        delta_red2blue = self.blue_positions[np.newaxis, :, :] - self.red_positions[:, np.newaxis, :]   # nred x nblue x 2

        # 计算红方智能体到蓝方智能体的角度
        angles_red2blue = np.arctan2(delta_red2blue[:, :, 1], delta_red2blue[:, :, 0])                  # nred x nblue

        # 计算红方智能体当前方向与到蓝方智能体的方向的角度差
        angles_diff_red2blue = angles_red2blue - self.red_directions[:, np.newaxis]                     # nred x nblue
        angles_diff_red2blue = (angles_diff_red2blue + np.pi) % (2 * np.pi) - np.pi

        # 计算红方智能体到蓝方智能体的距离
        distances_red2blue = distance.cdist(self.red_positions, self.blue_positions, 'euclidean')

        # 创建有效性掩码，只考虑存活的红方和蓝方智能体之间的距离
        valid_mask = self.red_alives[:, np.newaxis] & self.blue_alives[np.newaxis, :]

        # 将无效的距离设置为无限大
        distances_red2blue = np.where(valid_mask, distances_red2blue, np.inf)

        # 红方智能体攻击范围内的蓝方智能体
        blue_in_attack_zone = (distances_red2blue < self.attack_radius) & (
            np.abs(angles_diff_red2blue) < self.attack_angle / 2
        )

        # 将不在攻击范围内的距离设置为无限大
        distances_red2blue[~blue_in_attack_zone] = np.inf

        # 找到每个红方智能体最近的蓝方智能体
        nearest_blue_id = np.argmin(distances_red2blue, axis=1)

        # 如果红方智能体没有攻击范围内的蓝方智能体，目标设为-1
        nearest_blue_id[np.all(np.isinf(distances_red2blue), axis=1)] = -1

        # 更新红方智能体的目标
        red_targets = nearest_blue_id

        # 更新 collide_mask，排除没有 target 的智能体
        valid_collide_mask = collide_mask & (red_targets != -1) & self.red_alives

        # 获取有效的 target_id
        target_ids = red_targets[valid_collide_mask]
        agent_ids = np.where(valid_collide_mask)[0]

        # 获取红方智能体和其目标之间的距离
        valid_distances = distances_red2blue[valid_collide_mask, target_ids]

        # 判断撞击成功的情况
        collide_success_mask = valid_distances < self.collide_distance + self.red_velocities[valid_collide_mask] * self.dt_time 

        # 获取撞击成功的 agent_id 和 target_id
        success_agent_ids = agent_ids[collide_success_mask]
        success_target_ids = target_ids[collide_success_mask]

        self.collide_success_num = success_agent_ids.size

        # 更新红方智能体和目标蓝方智能体的存活状态
        self.red_alives[success_agent_ids] = False
        self.blue_alives[success_target_ids] = False

        # 更新红方智能体的方向
        self.red_directions[valid_collide_mask] = angles_red2blue[valid_collide_mask, target_ids]
        pt[valid_collide_mask] = 0

        return pt

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

        self.check_boundaries()

        # Update step counter
        self._total_steps += 1
        self._episode_steps += 1

        # Update terminated flag and reward
        reward = self.reward_battle()
        info = {'win': False, "other": ""}

        terminated, win, res = self.get_result()

        if terminated:
            self._episode_count += 1
        
        if win:
            reward += self.reward_win
        else:
            reward += self.reward_defeat

        info['win'] = win
        info['other'] = res

        return reward, terminated, info

    def merge_state(self):
        self.positions = np.vstack([self.red_positions, self.blue_positions])
        self.directions = np.hstack([self.red_directions, self.blue_directions])
        self.velocities = np.hstack([self.red_velocities, self.blue_velocities])
        self.alives = np.hstack([self.red_alives, self.blue_alives])

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
        self.total_hit_core_num += self.attack_core_num

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

        self.blue_alives[self.in_threat_zone_times >= self.max_in_threat_zone_time] = False

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
        red_in_explode_zone = distances_blue2red < self.explode_radius

        # 蓝方智能体自爆的掩码
        self_destruction_mask = np.zeros(self.n_blues, dtype=bool)

        if alive_percentage > 0.8:
            # 第一条规则：存活比例超过80%且自爆范围内的红方智能体数量超过2则自爆
            red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)
            self_destruction_mask = red_counts_in_zone > 2
        
        elif 0.6 < alive_percentage <= 0.8:
            # 第二条规则：如果蓝方存活智能体的数量在60%和80%之间且自爆范围内红方智能体的数量超过3,则自爆
            red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)
            self_destruction_mask = red_counts_in_zone > 3
        
        self_destruction_mask &= self.blue_alives

        # 触发自爆的蓝方智能体将被标记为死亡
        self.blue_alives[self_destruction_mask] = False

        # 将自爆范围内的红方智能体标记为死亡
        red_explode_mask = np.any(red_in_explode_zone[self_destruction_mask], axis=0)

        self.explode_red_num = np.sum(red_explode_mask & self.red_alives)

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

    def reward_battle(self):
        reward = self.reward_time

        num = np.array([self.explode_red_num, self.explode_blue_num, self.invalid_explode_num, self.collide_success_num, self.attack_core_num, self.out_of_bounds_num])
        value = np.array([self.reward_explode_red, self.reward_explode_blue, self.reward_explode_invalid, self.reward_collied, self.reward_attack_core, self.reward_out_of_bound])

        reward += np.sum(num * value)

        return reward
    
    def get_result(self):
        # 计算存活的智能体数量
        n_red_alive = np.sum(self.red_alives)
        n_blue_alive = np.sum(self.blue_alives)

        # 初始化对抗结束标志和获胜方标志
        terminated = False
        win = False
        info = ""

        # 检查终止条件
        if self._episode_count >= self.episode_limit:
            terminated = True
            win = self.total_hit_core_num < self.max_attack_core_num or n_blue_alive == 0
            if win:
                info = "Time limit reached. Red wins because blue could not destroy the base or all blue units were destroyed."
            else:
                info = "Time limit reached. Blue wins by destroying the red base."
        elif n_red_alive == 0:
            terminated = True
            win = False
            info = "All red units destroyed. Blue wins."
        elif n_blue_alive == 0:
            terminated = True
            win = True
            info = "All blue units destroyed. Red wins."
        elif self.total_hit_core_num >= self.max_attack_core_num:
            terminated = True
            win = False
            info = "Red base detroyed. Blue wins."
        elif n_blue_alive + self.total_hit_core_num < self.max_attack_core_num:
            terminated = True
            win = True
            info = "Remaining blue units insufficient to destroy red base. Red wins."

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
        half_size_x = self.size_x / 2
        half_size_y = self.size_y / 2

        self.transformed_lines = np.zeros_like(self.red_lines)
        self.transformed_lines[:, :, 0] = ((self.red_lines[:, :, 0] + half_size_x) / self.size_x * self.screen_width).astype(int)
        self.transformed_lines[:, :, 1] = ((self.red_lines[:, :, 1] + half_size_y) / self.size_y * self.screen_height).astype(int)

    def transform_circles(self):
        self.transformed_circles_center = []
        self.transformed_circles_radius = []

        circles = [self.red_core, self.red_base] + self.blue_bases

        for circle in circles:
            self.transformed_circles_center.append(self.transform_position(circle['center']))
            self.transformed_circles_radius.append(circle['radius'] / self.size_x * self.screen_width)
            
        
    def render(self, frame_num=0, save_frames=False):

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            red_plane_img = pygame.image.load('./png/red_plane_s.png').convert_alpha()
            blue_plane_img = pygame.image.load('./png/blue_plane_s.png').convert_alpha()

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
                # cache = self.red_img_cache if i < self.n_reds else self.blue_img_cache

                rotated_img = pygame.transform.rotate(image, -angles[i])
                # rotated_img = self.get_rotated_image(image, angles[i], cache, i)
                new_rect = rotated_img.get_rect(center=self.transformed_positions[i])
                self.screen.blit(rotated_img, new_rect)

        # 计算存活的智能体数量
        red_alive = sum(self.alives[:self.n_reds])
        blue_alive = sum(self.alives[self.n_reds:])

        # 渲染存活数量文本
        red_text = self.font.render(f'Red Alive: {red_alive}', True, (255, 0, 0))
        blue_text = self.font.render(f'Blue Alive: {blue_alive}', True, (0, 0, 255))
        self.screen.blit(red_text, (10, 10))
        self.screen.blit(blue_text, (10, 50))

        pygame.display.flip()

        if save_frames:
            frame_path = f"frames/frame_{frame_num:04d}.png"
            pygame.image.save(self.screen, frame_path)
            



def calculate_sector_theta(pos1, pos2, center):
    theta1 = np.arctan2(pos1[1] - center[1], pos1[0] - center[0])
    theta2 = np.arctan2(pos2[1] - center[1], pos2[0] - center[0])
    
    # Normalize theta to the range[0, 2*pi]
    theta1 = (theta1 + 2 * np.pi) % (2 * np.pi)
    theta2 = (theta2 + 2 * np.pi) % (2 * np.pi)

    return theta1, theta2


if __name__ == "__main__":

    world = DefenseEnv()

    import time
    obs = world.reset()
    num_frames = 100

    time_list = []
    world.render(frame_num=0, save_frames=True)

    for i in range(1, num_frames):
        print('-'* 30)
        start_time = time.time()
        last_time = time.time()

        world.get_state()
        print("获取状态: {:.5f}".format(time.time() - last_time))
        last_time = time.time()

        world.get_obs()
        print("获取观测: {:.5f}".format(time.time() - last_time))
        last_time = time.time()

        actions = world.scripted_policy_red()
        print("脚本策略: {:.5f}".format(time.time() - last_time))
        last_time = time.time()
        
        world.step(actions)
        print("环境更新: {:.5f}".format(time.time() - last_time))
        last_time = time.time()
        
        world.render(frame_num=i, save_frames=True)
        print("环境渲染: {:.5f}".format(time.time() - last_time))
        time_list.append(time.time() - start_time)
    
    time_list = np.array(time_list)

    print(time_list.mean(), time_list.std())

    world.close()


    


        





        



