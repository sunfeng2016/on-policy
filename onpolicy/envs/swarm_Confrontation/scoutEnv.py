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

image_dir = "/home/ubuntu/sunfeng/MARL/on-policy/onpolicy/envs/swarm_Confrontation/"
os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

class ScoutEnv(BaseEnv):
    def __init__(self, args):
        super(ScoutEnv, self).__init__(args)

        # 蓝方待侦察区域
        self.scout_width = 6000
        self.scout_height = 4000
        self.scout_pos = np.array([-self.scout_width / 2, self.scout_height / 2])

        # 蓝方候选高价值区域 (从左到右，从上到下的顺序)
        self.candidate_core = [
            {'center': np.array([-2250.0,  1250.0]), 'radius': 250.0},
            {'center': np.array([-2250.0, -1250.0]), 'radius': 250.0}, 
            {'center': np.array([-1700.0,   700.0]), 'radius': 300.0}, 
            {'center': np.array([-1750.0, -1050.0]), 'radius': 250.0}, 
            {'center': np.array([ -700.0,  -100.0]), 'radius': 300.0}, 
            {'center': np.array([  300.0,  -800.0]), 'radius': 300.0}, 
            {'center': np.array([ 2450.0,  1650.0]), 'radius': 250.0}, 
            {'center': np.array([ 2250.0, -1250.0]), 'radius': 250.0}, 
        ]
        # 从8个可能的高价值区域中随机选择4个作为高价值区域
        self.core_ranges_num = 4

        # 蓝方的威胁区域
        self.threat_ranges_num = 3
        self.threat_ranges = [
            {'center': np.array([-1250.0,   250.0]), 'radius': 250.0},
            {'center': np.array([-1100.0,  -700.0]), 'radius': 300.0},
            {'center': np.array([ 1000.0,  -800.0]), 'radius': 200.0},
        ]

        # 红方分布的区域
        self.red_group_num = 2

        self.red_init_range = [
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
            }
        ]

        # 划分网格
        # 将整个侦察区域划分为 50x50 的小格子
        self.grid_size = 50

        # 计算小格子的数量
        self.num_grids_x = self.scout_width // self.grid_size
        self.num_grids_y = self.scout_height // self.grid_size
        self.num_grids = self.num_grids_x * self.num_grids_y

        # 生成每个小格子的中心坐标
        x_centers = np.linspace(-self.scout_width/2 + self.grid_size/2,
                                self.scout_width/2 - self.grid_size/2,
                                self.num_grids_x)
        y_centers = np.linspace(-self.scout_height/2 + self.grid_size/2,
                                self.scout_height/2 - self.grid_size/2,
                                self.num_grids_y)
        
        # 使用 meshgrid 创建网络
        X, Y = np.meshgrid(x_centers, y_centers)

        # 将 X 和 Y 合并成一个 nx x ny x 2 的数组
        self.grid_centers = np.stack((X, Y), axis=-1)

        # 每个格子左上角的顶点坐标
        self.grid_left_tops = self.grid_centers + np.array([-self.grid_size / 2, self.grid_size / 2])

        self.scout_dist = 10

        # 防守的距离
        self.guard_dist = 100

        # 修改状态空间
        # base_state_size = self.get_state_size()
        # base_state_size[0] += self.num_grids
        # base_state_size.append([self.num_grids_y, self.num_grids_x])
        # self.share_observation_space = [base_state_size] * self.n_reds

        # Reward
        self.reward_time = 0.1
        self.reward_explode_red = -10
        self.reward_explode_blue = 5
        self.reward_explode_invalid = -20
        self.reward_collied_red = -10
        self.reward_collied_blue = 5
        self.reward_out_of_bound = -10
        self.reward_out_of_scout = -1
        self.reward_in_threat = -10
        self.reward_scout_core = 20
        self.reward_scout_comm = 5

        self.reward_win = 3000
        self.reward_defeat = 0


    def reset(self):
        super().reset()

        # 每个格子是否被扫描
        self.grid_scout = np.zeros((self.num_grids_y, self.num_grids_x), dtype=bool)

        # 每个格子是否在高价值区域内
        self.core_grids = self.reset_core_grids()
        # 核心侦察区域的格子数目
        self.core_grids_num = np.sum(self.core_grids)

        # 每个格子是否在威胁区域内
        self.threat_grids = self.reset_threat_grids()
        # 威胁区域的格子数目
        self.threat_grids_num = np.sum(self.threat_grids)

        # 普通侦察区域的格子数目
        self.comm_grids_num = self.num_grids - self.core_grids_num - self.threat_grids_num

        # 核心区被侦察的格子数
        self.scouted_core_num = 0
        self.scouted_core_total = 0

        # 普通区被侦察的格子数
        self.scouted_comm_num = 0
        self.scouted_comm_total = 0

        # 在侦察区域外的智能体数
        self.outofscout_num = 0             # 每个时间步在侦察区域外的智能体数目
        self.outofscout_total = 0

        # 在威胁区的智能体数目
        self.in_threat_red_num = 0
        self.in_threat_red_total = 0

        local_obs = self.get_obs()
        
        # agent_state = self.get_state()
        # state = np.concatenate((agent_state, self.grid_scout.astype(float).flatten()))
        # global_state = [state] * self.n_reds

        global_state = [self.get_state()] * self.n_reds

        available_actions = None

        return local_obs, global_state, available_actions

    def reset_core_grids(self):
        core_grids = np.zeros((self.num_grids_y, self.num_grids_x), dtype=bool)
        grid_centers_flat = self.grid_centers.reshape(-1, 2)  # 将 grid_centers 扁平化为二维数组

        for core_range in self.core_ranges:
            is_core_grids = self.in_circle(grid_centers_flat, core_range['center'], core_range['radius'])
            core_grids |= is_core_grids.reshape(self.num_grids_y, self.num_grids_x)

        return core_grids

    def reset_threat_grids(self):
        threat_grids = np.zeros((self.num_grids_y, self.num_grids_x), dtype=bool)
        grid_centers_flat = self.grid_centers.reshape(-1, 2)  # 将 grid_centers 扁平化为二维数组
        
        for threat_range in self.threat_ranges:
            is_threat_grids = self.in_circle(grid_centers_flat, threat_range['center'], threat_range['radius'])
            threat_grids |= is_threat_grids.reshape(self.num_grids_y, self.num_grids_x)

        return threat_grids
        
    def in_threat_area(self):
        in_threat = np.zeros(self.n_reds, dtype=bool)
        
        for threat_range in self.threat_ranges:
            is_threat_grids = self.in_circle(self.red_positions, threat_range['center'], threat_range['radius'])
            in_threat |= is_threat_grids

        # 将威胁范围内的红方智能体标记为死亡
        self.in_threat_red_num = np.sum(in_threat & self.red_alives)
        self.red_alives[in_threat] = False

    def in_circle(self, positions, center, radius, return_distance=False):
        """
        判断 positions 中的坐标是否在给定的圆形区域内
        """

        distances = np.linalg.norm(positions - center, axis=1)
        
        if return_distance:
            return distances < radius, distances
        else:
            return distances < radius

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

        self.in_threat_area()
        self.update_scout()
        
        self.blue_step()
        self.merge_state()

        self.check_boundaries()

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
            'scout_core_ratio': self.scouted_core_total / self.core_grids_num,
            'scout_comm_ratio': self.scouted_comm_total / self.comm_grids_num,
            'outofbound_ratio': self.outofbound_red_total / self.n_reds, # 出界死 （TODO）
            'invalid_explode_ratio': self.invalid_explode_red_total / self.n_reds, # 无效自爆死
            'be_exploded_ratio': self.explode_red_total / self.n_reds, # 被炸死
            'collided_ratio': (self.collide_blue_total + self.collide_red_total) / self.n_reds, # 撞死
            'hit_core_num': 0,
            "won": self.win_counted,
            "other": res
        }

        local_obs = self.get_obs()

        # agent_state = self.get_state()
        # state = np.concatenate((agent_state, self.grid_scout.astype(float).flatten()))
        # global_state = [state] * self.n_reds

        global_state = [self.get_state()] * self.n_reds

        reward = self.get_reward(win)
        rewards = [[reward]] * self.n_reds

        dones = np.zeros((self.n_reds), dtype=bool)
        dones = np.where(terminated, True, ~self.red_alives)

        infos = [info] * self.n_reds
        
        available_actions = None

        return local_obs, global_state, rewards, dones, infos, available_actions

    def blue_explode(self):
        """
        判断蓝方智能体是否需要自爆:
        规则如下：
            1. 如果红方存活智能体数量超过70%, 且蓝方智能体自爆范围内超过3个红方智能体, 则蓝方智能体自爆
            2. 如果红方存活智能体数量在40%到70%之间, 且蓝方智能体的自爆范围内超过2个红方智能体, 则蓝方智能体自爆
            3. 如果红方存活智能体数量少于40%, 且蓝方智能体的自爆范围内超过1个红方智能体, 则蓝方智能体自爆
        """

        # 计算存活的红方智能体数量
        alive_count = np.sum(self.red_alives)
        alive_percentage = alive_count / self.n_reds

        # 计算每个蓝方智能体与每个红方智能体的距离
        distances_blue2red = distance.cdist(self.blue_positions, self.red_positions, 'euclidean')

        # 蓝方智能体自爆范围内的红方智能体
        red_in_explode_zone = distances_blue2red < self.explode_radius
        red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)

        if alive_percentage >= 0.7:
            self_destruction_mask = red_counts_in_zone >= 3
        elif 0.4 <= alive_percentage < 0.7:
            self_destruction_mask = red_counts_in_zone >= 2
        else:
            self_destruction_mask = red_counts_in_zone >= 1

        self_destruction_mask &= self.blue_alives

        # 将自爆范围内的红方智能体标记为死亡
        red_explode_mask = np.any(red_in_explode_zone[self_destruction_mask], axis=0)

        self.explode_red_num = np.sum(red_explode_mask & self.red_alives)
        self.explode_red_total += self.explode_red_num

        self.red_alives[red_explode_mask] = False

    def blue_collide(self, pt):
        # 计算蓝方智能体到红方智能体的方向向量
        delta_blue2red = self.red_positions[np.newaxis, :, :] - self.blue_positions[:, np.newaxis, :]

        # 计算蓝方智能体到红方智能体的角度
        angles_blue2red = np.arctan2(delta_blue2red[:, :, 1], delta_blue2red[:, :, 0])

        # 计算蓝方智能体当前方向与到红方智能体的方向的角度差
        angles_diff_blue2red = angles_blue2red - self.blue_directions[:, np.newaxis]
        angles_diff_blue2red = (angles_diff_blue2red + np.pi) % (2 * np.pi) - np.pi

        # 计算蓝方智能体到红方智能体之间的距离
        distances_blue2red = distance.cdist(self.blue_positions, self.red_positions, 'euclidean')

        # 创建有效性掩码，只考虑存活的蓝方和红方智能体之间的距离
        valid_mask = self.blue_alives[:, np.newaxis] & self.red_alives[np.newaxis, :]

        # 将无效距离设置为无限大
        distances_blue2red = np.where(valid_mask, distances_blue2red, np.inf)

        # 蓝方智能体攻击范围内的红方智能体
        red_in_attack_zone = (distances_blue2red < self.attack_radius) & (
            np.abs(angles_diff_blue2red) < self.attack_angle / 2
        )

        # 将不在攻击范围内的距离设置为无限大
        distances_blue2red[~red_in_attack_zone] = np.inf

        # 找个每个蓝方智能体最近的红方智能体
        nearest_red_id = np.argmin(distances_blue2red, axis=1)

        # 如果蓝方智能体攻击范围内没有红方智能体，目标设为-1
        nearest_red_id[np.all(np.isinf(distances_blue2red), axis=1)] = -1

        # 更新蓝方智能体的目标
        blue_targets = nearest_red_id

        # 更新 collide_mask, 排除没有 target 的智能体
        collide_mask = (blue_targets != -1) & self.blue_alives

        # 获取有效的 target_id
        target_ids = blue_targets[collide_mask]
        agent_ids = np.where(collide_mask)[0]

        # 获取蓝方智能体与其目标之间的距离
        distances = distances_blue2red[collide_mask, target_ids]

        # 判断撞击成功的情况
        collide_success_mask = distances < (self.collide_distance + self.blue_velocities[collide_mask] * self.dt_time)

        # 获取撞击成功的 agent_id 和 target_id
        success_agent_ids = agent_ids[collide_success_mask]
        success_target_ids = target_ids[collide_success_mask]

        self.collide_red_num = success_target_ids.size
        self.collide_red_total += self.collide_red_num

        # 更新蓝方智能体和目标红方智能体的存活状态
        self.blue_alives[success_agent_ids] = False
        self.red_alives[success_target_ids] = False

        # 更新蓝方智能体的方向
        self.blue_directions[collide_mask] = angles_blue2red[collide_mask, target_ids]
        pt[collide_mask] = 0

        return pt

    def blue_guard(self, pt):
        distances_to_core = []
        in_core_range = np.zeros(self.n_blues, dtype=bool)

        # 判断蓝方智能体是否在高价值区域内，且计算与高价值区域中心的距离
        for core in self.core_ranges:
            in_core_range_i, distances_i = self.in_circle(self.blue_positions, 
                core['center'], core['radius'] + self.guard_dist, return_distance=True)
            in_core_range |= in_core_range_i
            distances_to_core.append(distances_i)

        # 不考虑游曳的智能体
        in_core_range[:self.free_blue_num] = True
        # 不考虑死亡的智能体
        in_core_range |= ~self.blue_alives

        if np.sum(in_core_range) == self.n_blues:
            return pt

        # 找到每个智能体最近的高价值中心
        distances_to_core = np.column_stack(distances_to_core)
        nearest_core = np.argmin(distances_to_core, axis=1)

        # 获取目标高价值中心的坐标
        target_core = self.core_centers[nearest_core]

        # 计算期望方向
        desired_directions = np.arctan2(target_core[:, 1] - self.blue_positions[:, 1],
                                        target_core[:, 0] - self.blue_positions[:, 0])

        # 计算当前方向到期望方向的角度
        angles_diff = desired_directions - self.blue_directions

        # 将角度差规范到 [-pi, pi] 区间内
        angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi

        # 确保转向角度不超过最大角速度
        angles_diff = np.clip(angles_diff, -self.max_angular_vel, self.max_angular_vel)

        # 更新当前方向，受in_core_range限制
        self.blue_directions[~in_core_range] += angles_diff[~in_core_range]

        pt[~in_core_range] = 0

        return pt

    def blue_step(self):
        self.blue_explode()

        pt = np.random.uniform(-1.0, 1.0, size=self.n_blues)
        
        pt = self.blue_collide(pt)

        pt = self.blue_guard(pt)

        self.blue_directions += pt * self.max_angular_vel
        self.blue_directions = (self.blue_directions + np.pi) % (2 * np.pi) - np.pi
        self.blue_positions += np.column_stack((self.blue_velocities * np.cos(self.blue_directions), 
                                                self.blue_velocities * np.sin(self.blue_directions))) * self.dt_time

    def get_result(self):
        n_red_alive = np.sum(self.red_alives)

        terminated = False
        win = False
        info = ""

        core_percentage = self.scouted_core_total / self.core_grids_num
        comm_percentage = self.scouted_comm_total / self.comm_grids_num

        if core_percentage >= 1.0 and comm_percentage >= 0.7:
            terminated = True
            win = True
            info = '[Win] Finish Scout.'
        elif n_red_alive == 0:
            terminated = True
            win = False
            info = '[Defeat] All Dead.'
        elif self._episode_steps >= self.episode_limit:
            terminated = True
            win = False
            info = '[Defeat] Time out.'
        
        return terminated, win, info

    def get_reward(self, win=False):
        
        # 动态时间奖励
        time_penalty = self.reward_time * (1 + self._episode_steps / self.episode_limit)

        # 动态高价值区域侦察奖励 (侦察的越多，给的奖励越多)
        scout_core_reward = self.reward_scout_core * self.scouted_core_num * (1 + self.scouted_core_total / self.core_grids_num)

        # 动态低价值区域侦察奖励 (侦察的越多，给的奖励越多)
        scout_comme_reward = self.reward_scout_comm * self.scouted_comm_num * (1 + self.scouted_comm_total / self.comm_grids_num)

        # 自爆毁伤蓝方智能体奖励 (给固定的奖励)
        destroyed_reward = self.reward_explode_blue * self.explode_blue_num

        # 撞击毁伤蓝方智能体奖励 (给固定的奖励)
        collide_reward = self.reward_explode_blue * self.collide_blue_num

        # 动态出界惩罚 (出界越多，给的惩罚越多)
        out_of_bounds_penalty = self.reward_out_of_bound * self.outofbound_red_num * (1 + self.outofbound_red_total / self.n_reds)

        # 动态在侦察区域外惩罚
        out_of_scout_area_penalty = self.reward_out_of_scout * self.outofscout_num * (1 + self._episode_steps / self.episode_limit)

        # 被自爆炸死惩罚
        destroyed_penalty = self.reward_explode_red * self.explode_red_num

        # 被撞死惩罚
        collide_penalty = self.reward_collied_red * self.collide_red_num

        # 无效自爆惩罚
        invalid_explode_penalty = self.reward_explode_invalid * self.invalid_explode_red_num

        # 威胁区惩罚
        in_threat_penalty = self.reward_in_threat * self.in_threat_red_num

        # 获胜奖励
        win_reward = self.reward_win if win else self.reward_defeat

        total_reward = (win_reward + time_penalty + scout_core_reward + scout_comme_reward + 
                        destroyed_reward + collide_reward + out_of_bounds_penalty + out_of_bounds_penalty +
                        out_of_scout_area_penalty + destroyed_penalty + collide_penalty + invalid_explode_penalty +
                        in_threat_penalty)

        return total_reward


    def update_scout(self):
        """
        更新侦察区域的扫描情况
        """
        self.scouted_core_num = 0
        self.scouted_comm_num = 0

        # 获取存活的红方智能体位置
        alive_reds = self.red_positions[self.red_alives]

        # 判断智能体是否在矩形侦察区域内
        in_scout_area = (np.abs(alive_reds[:, 0]) <= self.scout_width / 2) & (np.abs(alive_reds[:, 1]) <= self.scout_height / 2)
        valid_positions = alive_reds[in_scout_area]

        self.outofscout_num = np.sum(~in_scout_area)
        self.outofscout_total += self.outofscout_num

        if len(valid_positions) == 0:
            return

        # 将中心点为 (0,0) 的坐标转换为左下角为原点的坐标系
        shifted_positions = (valid_positions + np.array([self.scout_width / 2, self.scout_height / 2]))

        # 计算智能体所在的格子索引
        grid_indices_x = (shifted_positions[:, 0] // self.grid_size).astype(int)
        grid_indices_y = (shifted_positions[:, 1] // self.grid_size).astype(int)

        # 获取格子的中心
        grid_centers = self.grid_centers[grid_indices_y, grid_indices_x]

        # 计算智能体位置与格子中心的距离
        distances = np.linalg.norm(valid_positions - grid_centers, axis=1)

        # 判断距离小于10的格子
        scouted = distances < self.scout_dist

        if not np.any(scouted):
            return

        # 获取已经被侦察过的格子
        already_scouted = self.grid_scout[grid_indices_y, grid_indices_x]

        # 获取在威胁区的格子
        in_threat_area = self.threat_grids[grid_indices_y, grid_indices_x]

        # 获取在核心区的格子
        in_core_area = self.core_grids[grid_indices_y, grid_indices_x]

        # 筛选出新的侦察到的格子
        new_scouted = scouted & ~already_scouted & ~in_threat_area

        if not any(new_scouted):
            return

        # 筛选出侦察到的核心区格子
        scouted_core = new_scouted & in_core_area

        # 去重：获取被侦察到的格子的唯一索引
        unique_scouted_indices = np.unique(np.stack((grid_indices_y[new_scouted],
                                                     grid_indices_x[new_scouted]),
                                                     axis=1), axis=0)
        unique_scouted_core_indices = np.unique(np.stack((grid_indices_y[scouted_core], 
                                                          grid_indices_x[scouted_core]),
                                                          axis=1), axis=0)

        # 计算侦察核心区的格子数目
        self.scouted_core_num = len(unique_scouted_core_indices)
        self.scouted_core_total += self.scouted_core_num

        # 计算侦察普通区的格子数目
        self.scouted_comm_num = len(unique_scouted_indices) - self.scouted_core_num
        self.scouted_comm_total += self.scouted_comm_num

        # 更新被侦察的格子
        self.grid_scout[unique_scouted_indices[:, 0], unique_scouted_indices[:, 1]] = True


    def init_in_rect(self, x_range, y_range, angle, num):
        x = np.random.uniform(x_range[0]+100, x_range[1]-100, num)
        y = np.random.uniform(y_range[0]+100, y_range[1]-100, num)

        positions = np.column_stack((x, y))
        directions = np.full(num, angle)

        return positions, directions
    
    def init_in_circle(self, center, radius, num):
        angles = np.random.uniform(0, 2 * np.pi, num)
        radii = radius * np.sqrt(np.random.uniform(0, 1, num))

        # 计算智能体的位置
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)

        positions = np.column_stack((x, y))

        return positions
    
    def generate_blue_positions(self):
        # 随机选取高价值区域
        self.core_ranges = random.sample(self.candidate_core, self.core_ranges_num)
        self.core_centers = [core['center'] for core in self.core_ranges]
        self.core_centers = np.vstack(self.core_centers)
        self.transform_circles()
        
        # 计算每个高价值区域的面积
        areas = np.array([core['radius'] ** 2 for core in self.core_ranges])

        # 计算每个区域面积所占比例
        ratios = areas / np.sum(areas)

        # 计算概率分布
        probs = np.array([0.5] + list(0.5 * ratios))

        # 根据概率分布生成每个组的大小
        group_sizes = np.random.multinomial(self.n_blues, probs)
        in_rect_size = group_sizes[0]
        in_core_sizes = group_sizes[1:]
    
        in_rect_positions = (np.random.rand(in_rect_size, 2) - 0.5) * (np.array([self.scout_width, self.scout_height])-100)
        
        in_core_positions = np.vstack([
            self.init_in_circle(core['center'], core['radius'], size)
            for core, size in zip(self.core_ranges, in_core_sizes)
        ]) 

        positions = np.vstack([in_rect_positions, in_core_positions])
        directions = np.random.uniform(-np.pi, np.pi, self.n_blues)

        # 随机游曳的蓝方智能体
        self.free_blue_num = in_rect_size

        return positions, directions

    def generate_red_positions(self):
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
        red_positions, red_directions = self.generate_red_positions()
        blue_positions, blue_directions = self.generate_blue_positions()

        positions = np.vstack([red_positions, blue_positions])
        directions = np.hstack([red_directions, blue_directions])
        velocities = np.hstack([np.full(self.n_reds, self.red_max_vel), np.full(self.n_blues, self.blue_max_vel)])

        return positions, directions, velocities

    def transform_circles(self):
        self.transformed_circles_center = []
        self.transformed_circles_radius = []
        self.circles_width = [5] * self.core_ranges_num + [5] * self.threat_ranges_num
        self.circles_color = [(0, 0, 255)] * self.core_ranges_num + [(255, 0, 0)] * self.threat_ranges_num

        circles = self.core_ranges + self.threat_ranges

        for circle in circles:
            self.transformed_circles_center.append(self.transform_position(circle['center']))
            self.transformed_circles_radius.append(circle['radius'] / self.size_x * self.screen_width)

    def transform_grids(self):
        new_center = np.array([-self.size_x / 2, self.size_y / 2])
        new_dir = np.array([1, -1])

        self.screen_grid_size = int(self.grid_size * self.scale_factor)
        self.screen_grid_left_tops = ((self.grid_left_tops - new_center) * new_dir * self.scale_factor).astype(int)

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            red_plane_img = pygame.image.load(f'{image_dir}/png/red_plane_s.png').convert_alpha()
            blue_plane_img = pygame.image.load(f'{image_dir}/png/blue_plane_s.png').convert_alpha()

            # 缩放飞机贴图
            scale_factor = 0.2
            self.red_plane_img = pygame.transform.scale(red_plane_img, (int(red_plane_img.get_width() * scale_factor), 
                                                                        int(red_plane_img.get_height() * scale_factor)))
            self.blue_plane_img = pygame.transform.scale(blue_plane_img, (int(blue_plane_img.get_width() * scale_factor), 
                                                                          int(blue_plane_img.get_height() * scale_factor)))

            pygame.display.set_caption("Swarm Confrontation")

            self.transform_grids()
            transformed_rect_center = self.transform_position(self.scout_pos)
            self.transformed_rect = (transformed_rect_center[0], transformed_rect_center[1], 
                                     self.scout_width*self.scale_factor, self.scout_height*self.scale_factor)
            self.num_circles = self.core_ranges_num + self.threat_ranges_num

            # 初始化字体
            self.font = pygame.font.SysFont(None, 36)

        self.screen.fill((255, 255, 255))
        self.transform_positions()
        angles = -np.degrees(self.directions)

        # 渲染 rect
        pygame.draw.rect(self.screen, (0, 0, 0), self.transformed_rect, 3)

        # 渲染 Circles
        for i in range(self.num_circles):
            pygame.draw.circle(self.screen, self.circles_color[i], self.transformed_circles_center[i], 
                               self.transformed_circles_radius[i], width=self.circles_width[i])
            
        # 渲染网格
        for i in range(self.num_grids_y):
            for j in range(self.num_grids_x):
                x, y = self.screen_grid_left_tops[i, j, :]
                width = 0 if self.grid_scout[i, j] else 1
                pygame.draw.rect(self.screen, (0, 0, 0), (x, y, self.screen_grid_size, self.screen_grid_size), width)
        
        # 渲染 Plane
        for i in range(self.n_agents):
            if self.alives[i]:
                image = self.red_plane_img if i < self.n_reds else self.blue_plane_img
                rotated_img = pygame.transform.rotate(image, -angles[i])
                new_rect = rotated_img.get_rect(center=self.transformed_positions[i])
                self.screen.blit(rotated_img, new_rect)

        # 计算存活的智能体数量
        red_alive = sum(self.red_alives)
        blue_alive = sum(self.blue_alives)

        # 渲染 text
        red_text = self.font.render(f'Red Alive: {red_alive}', True, (255, 0, 0))
        blue_text = self.font.render(f'Blue Alive: {blue_alive}', True, (0, 0, 255))
        self.screen.blit(red_text, (10, 10))
        self.screen.blit(blue_text, (10, 50))

        pygame.display.flip()

        frame_dir = f"{image_dir}/Scout_frames/"
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        frame_path = os.path.join(frame_dir, f"frame_{self._total_steps:06d}.png")

        pygame.image.save(self.screen, frame_path)

class Arg(object):
    def __init__(self) -> None:
        self.map_name = '100_vs_100'
        self.scenario_name = 'scout'
        self.episode_length = 400

if __name__ == "__main__":
    args = Arg()

    env = ScoutEnv(args)

    env.reset()
    
    import time
    for i in range(200):
        start = time.time()
        actions = env.scripted_policy_red()
        env.step(actions)
        env.render()
        print(f'[frame: {i}]---[Time: {time.time() - start}]')

    # indices, distances = env.find_nearest_grid()


    


    


        





        



