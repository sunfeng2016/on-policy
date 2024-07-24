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

from onpolicy.envs.swarm_Confrontation.utils import append_to_csv
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
        # self.candidate_core = [
        #     {'center': np.array([-2250.0,  1250.0]), 'radius': 250.0},
        #     {'center': np.array([-2250.0, -1250.0]), 'radius': 250.0}, 
        #     {'center': np.array([-1700.0,   700.0]), 'radius': 300.0}, 
        #     {'center': np.array([-1750.0, -1050.0]), 'radius': 250.0}, 
        #     {'center': np.array([ -700.0,  -100.0]), 'radius': 300.0}, 
        #     {'center': np.array([  300.0,  -800.0]), 'radius': 300.0}, 
        #     {'center': np.array([ 2450.0,  1650.0]), 'radius': 250.0}, 
        #     {'center': np.array([ 2250.0, -1250.0]), 'radius': 250.0}, 
        # ]
        self.candidate_core = [
            {'center': np.array([-1700.0,   700.0]), 'radius': 300.0}, 
            {'center': np.array([-1750.0, -1050.0]), 'radius': 250.0}, 
            {'center': np.array([ -700.0,  -100.0]), 'radius': 300.0}, 
            {'center': np.array([  300.0,  -800.0]), 'radius': 300.0}, 
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
        self.red_group_num = 4

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
        # self.num_grids_x = self.scout_width // self.grid_size
        # self.num_grids_y = self.scout_height // self.grid_size
        self.num_grids_x = self.size_x // self.grid_size
        self.num_grids_y = self.size_y // self.grid_size
        self.num_grids = self.num_grids_x * self.num_grids_y

        # 生成每个小格子的中心坐标
        # x_centers = np.linspace(-self.scout_width/2 + self.grid_size/2,
        #                         self.scout_width/2 - self.grid_size/2,
        #                         self.num_grids_x)
        # y_centers = np.linspace(-self.scout_height/2 + self.grid_size/2,
        #                         self.scout_height/2 - self.grid_size/2,
        #                         self.num_grids_y)
        
        x_centers = np.linspace(-self.size_x/2 + self.grid_size/2,
                                self.size_x/2 - self.grid_size/2,
                                self.num_grids_x)
        y_centers = np.linspace(-self.size_y/2 + self.grid_size/2,
                                self.size_y/2 - self.grid_size/2,
                                self.num_grids_y)
        
        # 使用 meshgrid 创建网络
        X, Y = np.meshgrid(x_centers, y_centers)

        # 将 X 和 Y 合并成一个 nx x ny x 2 的数组
        self.grid_centers = np.stack((X, Y), axis=-1)

        # 每个格子左上角的顶点坐标
        self.grid_left_tops = self.grid_centers + np.array([-self.grid_size / 2, self.grid_size / 2])

        # 飞机位置与格子的中心距离阈值
        self.scout_dist = 25

        # 蓝方防守在高价值区域的飞机的比例
        self.guard_ratio = 0.3

        # 防守的距离
        self.guard_dist = 100

        # 修改状态空间
        base_state_size = self.get_state_size()
        state_size = [base_state_size[0] + self.num_grids, [base_state_size[0]], [1, self.num_grids_y, self.num_grids_x]]
        self.share_observation_space = [state_size] * self.n_reds

        # Reward
        self.time_reward = 0.1
        self.scout_core_reward = 10
        self.scout_comm_reward = 1
        self.kill_reward = 0.5
        self.be_killed_penalty = -1
        self.out_scout_penalty = -0.05
        self.in_threat_penalty = -0.5 # 增加红方进入威胁区域的惩罚
        self.repeated_scouted_penalty = -0.1 # 重复侦查惩罚
        self.reward_near_area = 0.1         # 靠近目标区域的奖励
        self.reward_away_area = -0.1

        self.reward_win = 1000
        self.reward_defeat = 0

        # 在威胁区的最大时间
        self.max_in_threat_time = 5

        # 侦查边界
        self.half_scout_size_x = self.scout_width/2
        self.half_scout_size_y = self.scout_height/2
        self.scout_bounds = np.array([
            [[ 1,  1], [-1,  1]],   # 上边界
            [[-1,  1], [-1, -1]],   # 左边界
            [[-1, -1], [ 1, -1]],   # 下边界
            [[ 1, -1], [ 1,  1]]    # 右边界
        ]) * np.array([self.half_scout_size_x, self.half_scout_size_y])

        # 侦查边界向量、长度以及单位向量
        self.scout_bounds_vec = self.scout_bounds[:, 1, :] - self.scout_bounds[:, 0, :]
        self.scout_bounds_len = np.linalg.norm(self.scout_bounds_vec, axis=1)
        self.scout_bounds_unitvec = self.scout_bounds_vec / self.scout_bounds_len[:, np.newaxis]

    def reset(self):
        super().reset()

        # 每个格子的扫描情况：1：被扫描，0：未被扫描
        self.scouted_grids = np.zeros((self.num_grids_y, self.num_grids_x), dtype=bool)

        # 每个格子的类型 1: 普通区域，2：高价值区域，3：威胁区域， 4: 非侦察区域
        self.grids_type = np.ones((self.num_grids_y, self.num_grids_x), dtype=int)

        # 观测到格子信息
        self.grids_info = np.zeros_like(self.grids_type)

        # 每个格子是否在高价值区域内
        self.core_grids = self.reset_core_grids()
        # 核心侦察区域的格子数目
        self.core_grids_num = np.sum(self.core_grids)
        # 更新类型
        self.grids_type[self.core_grids] = 2

        # 每个格子是否在威胁区域内
        self.threat_grids = self.reset_threat_grids()
        # 威胁区域的格子数目
        self.threat_grids_num = np.sum(self.threat_grids)
        # 更新类型
        self.grids_type[self.threat_grids] = 3

        # 每个格子是否在非扫描区域
        self.out_grids = self.reset_out_grids()
        # 非侦察区域的格子数目
        self.out_grids_num = np.sum(self.out_grids)
        # 更新类型
        self.grids_type[self.out_grids] = 4

        # 每个格子是否是在低价值区域
        self.comm_grids = (self.grids_type == 1)
        # 普通侦察区域的格子数目
        self.comm_grids_num = np.sum(self.comm_grids)

        assert self.num_grids == (self.comm_grids_num + self.core_grids_num + self.threat_grids_num + self.out_grids_num)

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
        self.repeated_scouted = None

        # 红方智能体与目标区域的距离 
        self.dist2area = None               

        # 在威胁区的智能体数目
        self.in_threat_red_num = 0
        self.in_threat_red_total = 0

        # 红方智能体进入威胁区的次数
        self.in_threat_time = np.zeros(self.n_reds)

        local_obs = self.get_obs()
        
        agent_state = self.get_state()
        state = np.concatenate((agent_state, self.scouted_grids.astype(float).flatten()))
        global_state = [state] * self.n_reds

        # global_state = [self.get_state()] * self.n_reds

        available_actions = self.get_avail_actions()

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
    
    def reset_out_grids(self):
        out_grids = np.zeros((self.num_grids_y, self.num_grids_x), dtype=bool)
        grid_centers_flat = self.grid_centers.reshape(-1, 2)  # 将 grid_centers 扁平化为二维数组

        out_scout_x = (grid_centers_flat[:, 0] < -self.scout_width/2) | (grid_centers_flat[:, 0] > self.scout_width/2)
        out_scout_y = (grid_centers_flat[:, 1] < -self.scout_height/2) | (grid_centers_flat[:, 1] > self.scout_height/2)
        out_scout = out_scout_x | out_scout_y

        out_grids = out_scout.reshape(self.num_grids_y, self.num_grids_x)

        return out_grids

        
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

        # 存储数据
        self.red_action = np.stack([at, pt * self.max_angular_vel * 180 / np.pi, attack_t], axis=-1)

        explode_mask = (attack_t == 1)
        collide_mask = (attack_t == 2)
        soft_kill_mask = (attack_t == 3)
        
        # if (self.only_render or self.only_eval) and self._episode_steps <= 10:
        #     self.red_scripted_policy()

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

        # self.in_threat_area()
        self.update_scout()
        
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
            'kill_num': self.explode_blue_total + self.collide_blue_total, # 红方毁伤蓝方的总数
            'hit_core_num': 0, # 高价值区域被打击的次数
            'explode_ratio_blue': self.blue_self_destruction_total / self.n_blues, # 蓝方主动自爆的比例
            'scout_core_ratio': self.scouted_core_total / self.core_grids_num, # 高价值区域被侦察的比例
            'scout_comm_ratio': self.scouted_comm_total / self.comm_grids_num, # 普通区域被侦察的比例
            'episode_length': self._episode_steps, # 轨迹长度
            'won': self.win_counted,
            "other": res
        }


        local_obs = self.get_obs()

        agent_state = self.get_state()
        state = np.concatenate((agent_state, self.grids_info.astype(float).flatten()))
        global_state = [state] * self.n_reds

        # global_state = [self.get_state()] * self.n_reds

        rewards = self.get_reward(win)
        # rewards = [[reward]] * self.n_reds

        dones = np.zeros((self.n_reds), dtype=bool)
        dones = np.where(terminated, True, ~self.red_alives)

        infos = [info] * self.n_reds
        
        available_actions = self.get_avail_actions()
        
        # 避开威胁区域
        available_actions = self.avoid_threat_zone(available_actions)

        # # 存储数据
        # self.dump_data()

        # # 输出数据到csv
        # if terminated:
        #     append_to_csv(self.agent_data, self.filename)

        return local_obs, global_state, rewards, dones, infos, available_actions

    def get_avail_heading_actions(self):
        # 初始化有效动作数组
        available_actions = np.ones((self.n_reds, self.heading_action_num), dtype=bool)

        # 判断智能体是否在扫描边界外
        out_of_bounds_x = (self.red_positions[:, 0] < -(self.scout_width / 2)) | (self.red_positions[:, 0] > (self.scout_width / 2))
        out_of_bounds_y = (self.red_positions[:, 1] < -(self.scout_height / 2)) | (self.red_positions[:, 1] > (self.scout_height / 2))
        out_of_bounds = out_of_bounds_x | out_of_bounds_y

        # 找到出界的智能体索引
        out_of_bounds_indices = np.where(out_of_bounds)[0]

        if out_of_bounds_indices.size == 0:
            return available_actions

        # 计算智能体当前位置到每个边界起点的向量
        pos_vec = self.red_positions[out_of_bounds_indices, np.newaxis, :] - self.scout_bounds[:, 0, :]  # (n, 4, 2)

        # 计算点向量的单位向量
        pos_unitvec = pos_vec / self.scout_bounds_len[:, np.newaxis]

        # 计算每个智能体位置在每条线段上的投影比例
        t = np.einsum('nij,ij->ni', pos_unitvec, self.scout_bounds_unitvec)   # (n, 4)

        # 将 t 限制在 [0, 1] 范围内，确保投影点在线段上
        t = np.clip(t, 0.0, 1.0)

        # 计算线段上距离智能体最近的点的坐标 (投影点)
        nearest = self.scout_bounds[:, 0, :] + t[:, :, np.newaxis] * self.scout_bounds_vec[np.newaxis, :, :]  # (n, 4, 2)

        # 计算智能体当前位置到最近点的距离
        nearest_dist = np.linalg.norm(self.red_positions[out_of_bounds_indices, np.newaxis, :] - nearest, axis=2) # (n, 4)

        # 找到每个智能体距离最近的线段的索引
        nearest_id = np.argmin(nearest_dist, axis=1)    # (n)

        # 获取每个智能体最近的目标点
        nearest_target = nearest[np.arange(out_of_bounds_indices.size), nearest_id] # (n, 2)

        # 计算智能体的期望方向
        desired_directions = np.arctan2(nearest_target[:, 1] - self.red_positions[out_of_bounds_indices, 1],
                                        nearest_target[:, 0] - self.red_positions[out_of_bounds_indices, 0])    # (n)
        
        # 计算当前方向到期望方向的角度
        angles_diff = desired_directions - self.red_directions[out_of_bounds_indices] # (n)
        angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi
        angles_diff_threshold = self.max_angular_vel

        # # 如果角度差大于angles_diff_threshold, 则只用-1, -2号动作
        # mask_neg = angles_diff <= -angles_diff_threshold
        # available_actions[out_of_bounds_indices[mask_neg], :self.heading_action_mid_id+1] = 0

        # # 如果角度差小于-angles_diff_threshold，则只用0, 1号动作
        # mask_pos = angles_diff >= angles_diff_threshold
        # available_actions[out_of_bounds_indices[mask_pos], self.heading_action_mid_id:] = 0

        # 如果角度差大于angles_diff_threshold, 则只用-1, -2号动作
        mask_pos = angles_diff >= angles_diff_threshold
        available_actions[out_of_bounds_indices[mask_pos], :self.heading_action_mid_id+1] = 0

        # 如果角度差小于-angles_diff_threshold，则只用0, 1号动作
        mask_neg = angles_diff <= -angles_diff_threshold
        available_actions[out_of_bounds_indices[mask_neg], self.heading_action_mid_id:] = 0

        return available_actions

    def avoid_threat_zone(self, available_actions):
        
        near_threat_radius = 100
        threat_center = np.array([d['center'] for d in self.threat_ranges])
        threat_radius = np.array([d['radius'] for d in self.threat_ranges])

        # 计算智能体当前位置到最近的威胁区的中心点的向量
        threat_vec = self.red_positions[:, np.newaxis, :] - threat_center[np.newaxis, :, :] # (n, 4)

        # 计算智能体当前位置到最近的威胁区的中心点的距离
        threat_dist = np.linalg.norm(threat_vec, axis=-1, ord=2)
        
        # 计算智能体当前位置到最近的威胁区的中心点的索引
        nearest_id = np.argmin(threat_dist, axis=-1)
        
        # 计算智能体当前位置到最近的威胁区的中心点的最短距离
        nearest_dist = np.amin(threat_dist, axis=-1)

        # 选出需要避开威胁区的智能体
        nearest_threat_radius = threat_radius[nearest_id]
        near_threat_agent = (nearest_dist - (near_threat_radius + nearest_threat_radius)) <= 0
        
        # 选出需要避开威胁区的智能体的索引
        near_threat_agent_id = np.where(near_threat_agent)[0]

        # 获取离每个需要避开威胁区的智能体最近的威胁区中心点
        nearest_threat = threat_center[nearest_id][near_threat_agent] # (n, 2)

        # 计算需要避开威胁区的智能体到威胁区的期望方向
        desired_directions = np.arctan2(nearest_threat[:, 1] - self.red_positions[near_threat_agent, 1],
                                        nearest_threat[:, 0] - self.red_positions[near_threat_agent, 0])    # (n)
        
        # print(near_threat_agent)
        
        # 计算当前方向到期望方向的角度
        angles_diff = desired_directions - self.red_directions[near_threat_agent]
        angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi
        angles_diff_threshold = self.max_angular_vel

        available_actions = np.array(available_actions)

        # 如果角度差大于angles_diff_threshold, 则只用-1, -2号动作
        mask_neg = angles_diff <= -angles_diff_threshold
        available_actions[near_threat_agent_id[mask_neg], self.acc_action_num:(self.acc_action_num+self.heading_action_mid_id+1)] = 0

        # 如果角度差小于-angles_diff_threshold，则只用0, 1号动作
        mask_pos = angles_diff >= angles_diff_threshold
        available_actions[near_threat_agent_id[mask_pos], (self.acc_action_num+self.heading_action_mid_id):(self.acc_action_num+self.heading_action_num)] = 0

        available_actions = [available_actions[i, :].tolist() for i in range(self.n_reds)]

        return available_actions        
        
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
        red_in_explode_zone = (distances_blue2red < self.explode_radius) & self.red_alives
        red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)

        if alive_percentage >= 0.7:
            self_destruction_mask = red_counts_in_zone >= 3
        elif 0.4 <= alive_percentage < 0.7:
            self_destruction_mask = red_counts_in_zone >= 2
        else:
            self_destruction_mask = red_counts_in_zone >= 1

        self_destruction_mask &= self.blue_alives

        # 记录蓝方自爆的智能体，用作渲染
        self.blue_self_destruction_mask = self_destruction_mask
        self.blue_self_destruction_num = np.sum(self_destruction_mask)
        self.blue_self_destruction_total += self.blue_self_destruction_num

        # 存储数据
        self.blue_action[self_destruction_mask == 1, 2] = 1

        # 将自爆范围内的红方智能体标记为死亡
        red_explode_mask = np.any(red_in_explode_zone[self_destruction_mask], axis=0)

        self.be_exploded_flag = (red_explode_mask & self.red_alives)

        # 记录自爆范围内的红方智能体，用作渲染
        self.red_explode_mask = red_explode_mask & self.red_alives

        self.explode_red_num = np.sum(red_explode_mask & self.red_alives)
        self.explode_red_total += self.explode_red_num

        self.red_alives[red_explode_mask] = False
        self.blue_alives[self_destruction_mask] = False

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
        red_in_explode_zone = (distances_blue2red < self.explode_radius) & self.red_alives
        red_counts_in_zone = np.sum(red_in_explode_zone, axis=1)

        if alive_percentage >= 0.7:
            self_destruction_mask = red_counts_in_zone >= 3
        elif 0.4 <= alive_percentage < 0.7:
            self_destruction_mask = red_counts_in_zone >= 2
        else:
            self_destruction_mask = red_counts_in_zone >= 1

        self_destruction_mask &= self.blue_alives

        # 记录蓝方自爆的智能体，用作渲染
        self.blue_self_destruction_mask = self_destruction_mask
        self.blue_self_destruction_num = np.sum(self_destruction_mask)
        self.blue_self_destruction_total += self.blue_self_destruction_num

        # 存储数据
        self.blue_action[self_destruction_mask == 1, 2] = 1

        # 将自爆范围内的红方智能体标记为死亡
        red_explode_mask = np.any(red_in_explode_zone[self_destruction_mask], axis=0)

        self.be_exploded_flag = (red_explode_mask & self.red_alives)

        # 记录自爆范围内的红方智能体，用作渲染
        self.red_explode_mask = red_explode_mask & self.red_alives

        self.explode_red_num = np.sum(red_explode_mask & self.red_alives)
        self.explode_red_total += self.explode_red_num

        self.red_alives[red_explode_mask] = False
        self.blue_alives[self_destruction_mask] = False
        
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

        # 存储数据
        self.blue_action[success_agent_ids, 2] = 2

        # 记录蓝方撞击成功的智能体，用作渲染
        self.blue_collide_agent_ids = success_agent_ids
        self.blue_collide_target_ids = success_target_ids

        self.be_collided_flag = np.zeros(self.n_reds, dtype=bool)
        self.be_collided_flag[success_target_ids] = True

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

    def blue_return(self, pt):
        
        # 判断智能体是否出缓冲边界
        out_of_bounds_x = (self.blue_positions[:, 0] < -self.half_size_x) | (self.blue_positions[:, 0] > self.half_size_x)
        out_of_bounds_y = (self.blue_positions[:, 1] < -self.half_size_y) | (self.blue_positions[:, 1] > self.half_size_y)
        out_of_bounds = out_of_bounds_x | out_of_bounds_y

        # 找到出界的智能体索引
        out_of_bounds_indices = np.where(out_of_bounds)[0]

        if out_of_bounds_indices.size == 0:
            return pt

        # 计算智能体当前位置到每个边界起点的向量
        pos_vec = self.blue_positions[out_of_bounds_indices, np.newaxis, :] - self.cache_bounds[:, 0, :]  # (n, 4, 2)

        # 计算点向量的单位向量
        pos_unitvec = pos_vec / self.bounds_len[:, np.newaxis]

        # 计算每个智能体位置在每条线段上的投影比例
        t = np.einsum('nij,ij->ni', pos_unitvec, self.bounds_unitvec)   # (n, 4)

        # 将 t 限制在 [0, 1] 范围内，确保投影点在线段上
        t = np.clip(t, 0.0, 1.0)

        # 计算线段上距离智能体最近的点的坐标 (投影点)
        nearest = self.cache_bounds[:, 0, :] + t[:, :, np.newaxis] * self.bounds_vec[np.newaxis, :, :]  # (n, 4, 2)

        # 计算智能体当前位置到最近点的距离
        nearest_dist = np.linalg.norm(self.blue_positions[out_of_bounds_indices, np.newaxis, :] - nearest, axis=2) # (n, 4)

        # 找到每个智能体距离最近的线段的索引
        nearest_id = np.argmin(nearest_dist, axis=1)    # (n)

        # 获取每个智能体最近的目标点
        nearest_target = nearest[np.arange(out_of_bounds_indices.size), nearest_id] # (n, 2)

        # 计算智能体的期望方向
        desired_directions = np.arctan2(nearest_target[:, 1] - self.blue_positions[out_of_bounds_indices, 1],
                                        nearest_target[:, 0] - self.blue_positions[out_of_bounds_indices, 0])    # (n)
        
        # 计算当前方向到期望方向的角度
        angles_diff = desired_directions - self.blue_directions[out_of_bounds_indices] # (n)
        
        # 将角度差规范到 [-pi, pi] 区间内
        angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi

        # 确保转向角度不超过最大角速度
        angles_diff = np.clip(angles_diff, -self.max_angular_vel, self.max_angular_vel)

        # 更新所有out of bounds的当前方向
        self.blue_directions[out_of_bounds_indices] += angles_diff

        pt[out_of_bounds] = 0

        return pt

    def blue_step(self):
        self.blue_explode()

        pt = np.random.uniform(-1.0, 1.0, size=self.n_blues)
        
        pt = self.blue_collide(pt)

        pt = self.blue_guard(pt)

        # 蓝方出界返回
        pt = self.blue_return(pt)

        self.blue_directions += pt * self.max_angular_vel
        self.blue_directions = (self.blue_directions + np.pi) % (2 * np.pi) - np.pi
        self.blue_positions += np.column_stack((self.blue_velocities * np.cos(self.blue_directions), 
                                                self.blue_velocities * np.sin(self.blue_directions))) * self.dt_time

        # 存储数据
        self.blue_action[:, 1] = pt * self.max_angular_vel

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
    
    def get_dist_reward(self):
        # 获取红方目标矩形区域的边界
        x_min, x_max, y_min, y_max = -self.scout_width/2, self.scout_width/2, -self.scout_height/2, self.scout_height/2

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
    
    def get_reward(self, win=False):
        # 初始化每个智能体的奖励
        rewards = np.zeros(self.n_reds, dtype=float)

        # 计算奖励系数
        episode_progress = 1 + self._episode_steps / self.episode_limit
        dead_red_ratio = 1  + (1 - np.sum(self.red_alives) / self.n_reds)
        dead_blue_ratio = 1 + (1 - np.sum(self.blue_alives) / self.n_blues)
        scouted_core_ratio = 1 + self.scouted_core_total / self.core_grids_num
        scouted_comm_ratio = 1 + self.scouted_comm_total / self.comm_grids_num
        in_threat_progress = 1 + self.in_threat_time / self.max_in_threat_time

        # 时间奖励, 存活时间越久，奖励系数越大
        rewards += self.time_reward * self.red_alives * episode_progress

        # 扫描高价值区域的奖励，扫描的区域越多，奖励系数越大
        rewards += self.scout_core_reward * self.scout_core_grids * scouted_core_ratio

        # 扫描低价值区域的奖励，扫描的区域越多，奖励系数越大
        rewards += self.scout_comm_reward * self.scout_comm_grids * scouted_comm_ratio

        # 毁伤蓝方智能体的奖励，毁伤的敌方智能体数量越多，奖励系数越大
        kill_blue_agents = self.explode_flag | self.collide_flag
        rewards += self.kill_reward * kill_blue_agents * dead_blue_ratio

        # 获胜奖励
        rewards += self.reward_win if win else self.reward_defeat

        # 被毁伤惩罚，被毁伤的智能体越多，惩罚系数越大
        kill_red_agents = self.be_exploded_flag | self.be_collided_flag | self.dead_in_threat
        rewards += self.be_killed_penalty * kill_red_agents * dead_red_ratio

        # # 在扫描区外的惩罚，时间越久，惩罚系数越大
        # rewards += self.out_scout_penalty * self.scout_out_grids * episode_progress

        # 在威胁区的惩罚，时间越久，惩罚系数越大
        rewards += self.in_threat_penalty * self.scout_threat_grids * in_threat_progress

        # 重复侦查区域惩罚
        rewards += self.repeated_scouted_penalty * self.repeated_scouted

        # 在扫描区外的智能体，靠近侦查区域给奖励，远离给惩罚
        rewards += self.get_dist_reward()

        rewards = np.expand_dims(rewards, axis=1)

        return rewards.tolist()
    
    def update_scout(self):
        # 将中心点为 (0, 0) 的坐标转换为左下角为原点的坐标
        shifted_positions = (self.red_positions + np.array([self.size_x/2, self.size_y/2])) # (100, 2)

        # 计算智能体所在格子的索引
        grid_indices_x = (shifted_positions[:, 0] // self.grid_size).astype(int)    # (100, )
        grid_indices_y = (shifted_positions[:, 1] // self.grid_size).astype(int)    # (100, )

        out_of_bound = (grid_indices_x >= self.num_grids_x) | (grid_indices_y >= self.num_grids_y) | (grid_indices_x < 0) | (grid_indices_y < 0)

        grid_indices_x[out_of_bound] = 0
        grid_indices_y[out_of_bound] = 0

        # 扫描格子的判定
        scouted = np.ones(self.n_reds, dtype=bool)  # (100, )

        # # 获取格子的中心
        # grid_centers = self.grid_centers[grid_indices_y, grid_indices_x]            # (100, 2)

        # # 计算智能体位置与格子中心的距离
        # distances = np.linalg.norm(self.red_positions - grid_centers, axis=1)       # (100, )

        # # 判断距离小于条件的格子
        # scouted = distances <= self.scout_dist # (100, )
        
        # 排除出界的智能体
        scouted &= ~out_of_bound

        # 获取已经被侦察过的格子
        try:
            already_scouted = self.scouted_grids[grid_indices_y, grid_indices_x] # (100, )
        except:
            print('error')
        
        # 筛选出新的被侦察过的格子
        new_scouted = scouted & ~already_scouted & self.red_alives # (100, )

        # 记录重复侦查格子的智能体
        self.repeated_scouted = scouted & already_scouted # (100,)

        # 更新格子的扫描情况
        self.scouted_grids[grid_indices_y[new_scouted], grid_indices_x[new_scouted]] = True

        # 更新观测到的格子信息
        self.grids_info[self.scouted_grids] = self.grids_type[self.scouted_grids]

        # 扫描到低价值区的
        self.scout_comm_grids = np.zeros(self.n_reds, dtype=bool)
        self.scout_comm_grids[new_scouted] = self.comm_grids[grid_indices_y[new_scouted], grid_indices_x[new_scouted]]
        self.scouted_comm_total = np.sum(self.comm_grids & self.scouted_grids)

        # 扫描到高价值区的
        self.scout_core_grids = np.zeros(self.n_reds, dtype=bool)
        self.scout_core_grids[new_scouted] = self.core_grids[grid_indices_y[new_scouted], grid_indices_x[new_scouted]]
        self.scouted_core_total = np.sum(self.core_grids & self.scouted_grids)

        # 扫描到威胁区的 ()
        # self.scout_threat_grids = np.zeros(self.n_reds, dtype=bool)
        # self.scout_threat_grids[new_scouted] = self.threat_grids[grid_indices_y[new_scouted], grid_indices_x[new_scouted]]
        self.scout_threat_grids = self.threat_grids[grid_indices_y, grid_indices_x] & self.red_alives
        self.in_threat_time[self.scout_threat_grids] += 1
        self.in_threat_time[~self.scout_threat_grids] = 0
        self.dead_in_threat = (self.in_threat_time >= self.max_in_threat_time)
        self.red_alives[self.dead_in_threat] = False
        
        # 扫描到非侦察区
        # self.scout_out_grids = np.zeros(self.n_reds, dtype=bool)
        # self.scout_out_grids[new_scouted] = self.out_grids[grid_indices_y[new_scouted], grid_indices_x[new_scouted]]
        self.scout_out_grids = self.out_grids[grid_indices_y, grid_indices_x] & self.red_alives


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
        probs = np.array([1-self.guard_ratio] + list(self.guard_ratio * ratios))

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
        # for i in range(self.num_grids_y):
        #     for j in range(self.num_grids_x):
        #         if self.out_grids[i,j]: 
        #             continue
        #         else:
        #             x, y = self.screen_grid_left_tops[i, j, :]
        #             width = 0 if self.scouted_grids[i, j] else 1
        #             pygame.draw.rect(self.screen, (0, 0, 0), (x, y, self.screen_grid_size, self.screen_grid_size), width)
        
        for i in range(self.num_grids_y):
            for j in range(self.num_grids_x):
                if self.out_grids[i,j] or not self.scouted_grids[i, j]: 
                    continue
                else:
                    x, y = self.screen_grid_left_tops[i, j, :]
                    # width = 0 if self.scouted_grids[i, j] else 1
                    width = 1
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
        scout_core_ratio = self.scouted_core_total / self.core_grids_num
        scout_comm_ratio = self.scouted_comm_total / self.comm_grids_num

        # 渲染 text
        time_text = self.font.render(f'Episode: {self._episode_count} Time Step: {self._episode_steps} Win count: {self.battles_won}', True, (0, 0, 0))
        red_text = self.font.render(f'Red Alive: {red_alive}', True, (255, 0, 0))
        blue_text = self.font.render(f'Blue Alive: {blue_alive}', True, (0, 0, 255))
        scout_text = self.font.render(f'Scout Core: {round(scout_core_ratio, 2)} Scout Comm: {round(scout_comm_ratio, 2)}', True, (255, 0, 0))
        self.screen.blit(red_text, (10, 10))
        self.screen.blit(blue_text, (10, 50))
        self.screen.blit(scout_text, (10, 90))
        self.screen.blit(time_text, (10, 130))

        # 渲染自爆效果
        self.render_explode()

        # 渲染碰撞效果
        self.render_collide()

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
    for i in range(50):
        start = time.time()
        actions = env.scripted_policy_red()
        env.step(actions)
        env.render()
        print(f'[frame: {i}]---[Time: {time.time() - start}]')

    # indices, distances = env.find_nearest_grid()

