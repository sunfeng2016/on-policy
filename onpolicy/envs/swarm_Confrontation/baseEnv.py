# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-20
# @Description: Implementation of base Environment

import os
import sys
import pygame
import numpy as np
import re
import math
import datetime
import subprocess

sys.path.append("/home/ubuntu/sunfeng/MARL/on-policy/")
os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

from scipy.spatial import distance
from onpolicy.utils.multi_discrete import MultiDiscrete
from onpolicy.envs.swarm_Confrontation.plane_params import get_plane_params
from onpolicy.envs.swarm_Confrontation.utils import append_to_csv

class BaseEnv():
    def __init__(self, args):
        
        # 初始化场景参数
        self.map_name = args.map_name
        self.plane_name = args.plane_name
        self.scenario_name = args.scenario_name
        self.size_x, self.size_y = 8000, 5000
        self.n_reds, self.n_blues = map(int, re.findall(r'\d+', self.map_name))
        self.n_agents = self.n_reds + self.n_blues
        
        self.episode_limit = args.episode_length
        self.use_script = args.use_script
        self.save_sim_data = args.save_sim_data
        
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 获取当前脚本的文件路径和目录
        current_file_path = os.path.abspath(__file__)
        self.directory = os.path.dirname(current_file_path)

        # 设置观测参数
        self.max_observed_allies = 5
        self.max_observed_enemies = 5
        
        # 初始化时间步长
        self._episode_count = self._episode_steps = self._total_steps = 0
        
        # 设置仿真时间
        self.dt_time = 1.0
        
        # 初始化距离和角度参数
        self._init_capability_params(args)
               
        # 初始化观测和状态参数
        self._init_observation_and_state_params()
                
        # 初始化动作参数
        self._init_action_params()
        
        # 初始化战斗统计参数
        self.battles_game = self.battles_won = self.timeouts = 0
        
        # 初始化仿真数据
        self.sim_data = None
        
        # 初始化屏幕参数
        self._init_screen_params()
        
        # 初始化边界参数
        self._init_boundaries()    

        # 初始化渲染参数
        self._init_explosion_rendering()
        self._init_collision_rendering()
        self._init_soft_kill_rendering()
    
    def _init_capability_params(self, args):
        """
        初始化能力参数
        """
        plane_params = get_plane_params(args.plane_name)
        
        # 设置速度参数
        self.red_max_vel, self.red_min_vel = plane_params['red_vel']
        self.blue_max_vel, self.blue_min_vel = plane_params['blue_vel']
        self.max_angular_vel = plane_params['max_angular_vel']
        
        # 圆形通信范围
        self.detection_radius = plane_params['detection_radius']
        
        # 扇形攻击范围
        self.attack_radius = plane_params['attack_radius']
        self.attack_angle = plane_params['attack_angle']
        
        # 允许的自爆的距离和自爆的半径
        self.explode_radius = plane_params['explode_radius']
        self.can_explode_radius = self.explode_radius + 20
        
        # 碰撞距离
        self.collide_distance = plane_params['collide_distance']
        
        # 软杀伤的距离和半径
        self.soft_kill_distance = plane_params['soft_kill_distance']
        self.soft_kill_angle = plane_params['soft_kill_angle']
        
    def _init_observation_and_state_params(self):
        """
        初始化观测和状态参数
        """
        self.obs_own_feats_size = 4         # x, y, v, phi
        self.obs_ally_feats_size = 4        # x, y, d, v
        self.obs_enemy_feats_size = 5       # x, y, d, theta, phi
        self.obs_size = (self.obs_own_feats_size +
                         self.obs_ally_feats_size +
                         self.obs_enemy_feats_size)
        
        self.red_state_size = 4             # x, y, v, phi
        self.blue_state_size = 4            # x, y, v, phi
        
        # 局部观测空间
        self.observation_space = [self.get_obs_size()] * self.n_reds
        # 状态空间
        self.share_observation_space = [self.get_state_size()] * self.n_reds
        
    def _init_action_params(self):
        """
        初始化动作参数
        """
        # 加速类动作数目
        self.acc_action_num = 5
        # 航向类动作数目
        self.heading_action_num = 5
        # 动机类动作数目
        self.attack_action_num = 4
        
        self.acc_action_mid_id = self.acc_action_num // 2
        self.heading_action_mid_id = self.heading_action_num // 2
        
        self.acc_action_max = 5.0
        self.heading_action_max = 1.0
        
        # 单次开启软杀伤时长
        self.soft_kill_max_time = 10
        # 每个智能体每局可以开启软杀伤的次数
        self.soft_kill_max_num = 1
        # 在软杀伤范围内停留的最长时间，超过则被击毁
        self.max_time_in_soft_kill = 3
        
        # 加速动作空间        
        self.acc_actions = np.linspace(-self.acc_action_max,
                                       self.acc_action_max,
                                       self.acc_action_num)
        # 航向动作空间
        self.heading_actions = np.linspace(-self.heading_action_max,
                                           self.heading_action_max,
                                           self.heading_action_num)
        # 攻击动作空间
        self.attack_actions = np.arange(0, self.attack_action_num)
        
        # 多头多离散动作空间
        self.action_space = [MultiDiscrete([[0, self.acc_action_num-1],
                                            [0, self.heading_action_num-1],
                                            [0, self.attack_action_num-1]])] * self.n_reds
        
    def _init_soft_kill_params(self):
        """
        初始化软杀伤参数
        """
        # 单次开启软杀伤时长
        self.soft_kill_max_time = 10
        # 每个智能体每局可以开启软杀伤的次数
        self.soft_kill_max_num = 1
        # 在软杀伤范围内停留的最长时间，超过则被击毁
        self.max_time_in_soft_kill = 3
        # 记录红方智能体每局开过软杀伤的次数
        self.red_soft_kill_num = np.zeros(self.n_reds)
        # 记录蓝方智能体位于软杀伤范围内的时长
        self.blue_in_soft_kill_time = np.zeros((self.n_reds, self.n_blues))
        # 记录红方智能体开启软杀伤的时长
        self.red_soft_kill_time = np.zeros(self.n_reds)
                
    def _init_screen_params(self):
        """
        初始化屏幕参数
        """
        # 设置屏幕大小
        self.screen = None
        self.scale_factor = 0.5
        self.screen_width = self.size_x * self.scale_factor
        self.screen_height = self.size_y * self.scale_factor
        
        # 初始化贴图
        self.red_plane_img = self.blue_plane_img = None
        
        # 初始化帧率
        self.framerate = 5
        
        # 初始化帧保存路径
        self.frame_dir = None
        
        # 初始化屏幕中心的偏移和方向转换
        self.screen_offset = np.array([-self.size_x / 2, self.size_y / 2])
        self.direction_scale = np.array([1, -1])
        
    def _init_boundaries(self):
        """
        初始化边界参数
        """
        self.cache_factor = 0.92
        self.half_size_x = (self.size_x * self.cache_factor) / 2
        self.half_size_y = (self.size_y * self.cache_factor) / 2
        
        self.cache_bounds = np.array([
            [[1, 1], [-1, 1]],
            [[-1, 1], [-1, -1]],
            [[-1, -1], [1, -1]],
            [[1, -1], [1, 1]]
        ]) * np.array([self.half_size_x, self.half_size_y])
        
        self.bounds_vec = self.cache_bounds[:, 1, :] - self.cache_bounds[:, 0, :]
        self.bounds_len = np.linalg.norm(self.bounds_vec, axis=1)
        self.bounds_unitvec = self.bounds_vec / self.bounds_len[:, np.newaxis]
        
        self.max_out_of_bounds_time = 10
        
    def _init_explosion_rendering(self):
        """
        初始化爆炸渲染参数
        """
        # 爆炸特效持续5帧
        self.explode_render_frames = 5
        
        # 自爆特效还需要渲染的帧数
        self.red_explosion_frames_remaining = np.zeros(self.n_reds)
        self.blue_explosion_frames_remaining = np.zeros(self.n_blues)
        
        # 是否需要渲染主动自爆
        self.red_self_destruction_render = np.zeros(self.n_reds, dtype=bool)
        self.blue_self_destruction_render = np.zeros(self.n_blues, dtype=bool)
        
        # 当前时间步被自爆炸死的智能体
        self.red_explode_mask = np.zeros(self.n_reds, dtype=bool)
        self.blue_explode_mask = np.zeros(self.n_blues, dtype=bool)

        # 当前时间步主动自爆的智能体
        self.red_self_destruction_mask = np.zeros(self.n_reds, dtype=bool)
        self.blue_self_destruction_mask = np.zeros(self.n_blues, dtype=bool)
        
    def _init_collision_rendering(self):
        """
        初始化碰撞渲染参数
        """
        # 碰撞特效需要持续渲染5帧
        self.collide_render_frames = 5
        
        # 碰撞特效还需要渲染的帧数
        self.collision_frames_remaining = np.zeros(self.n_agents)
        
        # 碰撞成功的目标
        self.collide_success_target_id = np.full(self.n_agents, -1, dtype=int)
        
        # 红方发动撞击的智能体id和目标id
        self.red_collide_agent_ids = np.array([])
        self.red_collide_target_ids = np.array([])

        # 蓝方发动撞击的智能体id和目标id
        self.blue_collide_agent_ids = np.array([])
        self.blue_collide_target_ids = np.array([])
        
    def _init_soft_kill_rendering(self):
        """
        初始化软杀伤渲染参数
        """
        # 软杀伤特效还需要渲染的帧数
        self.red_soft_kill_frames_remaining = np.zeros(self.n_reds)
        self.blue_soft_kill_frames_remaining = np.zeros(self.n_blues)
        
        # 软杀伤的位置
        self.soft_kill_transformed_positions = np.zeros((self.n_agents, 2))

        # 软杀伤的智能体
        self.red_soft_kill_mask = np.zeros(self.n_reds)
        self.blue_soft_kill_mask = np.zeros(self.n_blues)
    
    def reset(self):
        """
        重置环境
        """
        # 重置基本计数器和状态变量
        self._reset_basic_state()
                       
        # 重置观测相关变量
        self._reset_observations()
        
        # 重置碰撞相关变量
        self._reset_collision_params() 
        
        # 重置对抗结果
        self.win_counted = False
        self.defeat_counted = False
        
        # 重置脚本索引
        self._reset_script_indices()
        
        # 重置攻击统计参数
        self._reset_attack_stats()
        
        # 重置每个智能体的出界时间
        self.out_of_bounds_time = np.zeros(self.n_agents)
        
        # 数据存储
        self.red_action = np.zeros((self.n_reds, 3))
        self.blue_action = np.zeros((self.n_blues, 3))  # 加速度，航向，攻击
        self.sim_data = []
            
    def _reset_basic_state(self):
        """
        重置计数器和基本状态变量
        """
        # 重置时间步
        self._episode_steps = 0

        # 重置位置、方向和速度
        self.positions, self.directions, self.velocities = self.init_positions()
        # 重置屏幕坐标
        self.transformed_positions = np.zeros_like(self.positions, dtype=int)
        
        # 重置存活状态
        self.alives = np.ones(self.n_agents, dtype=bool)
        
        # 状态分离
        self.split_state()
        
    def _reset_observations(self):
        """
        重置观测相关变量
        """
        self.observed_allies = -np.ones((self.n_reds, self.max_observed_allies), dtype=int)
        self.observed_enemies = -np.ones((self.n_reds, self.max_observed_enemies), dtype=int)
        self.distance_observed_allies = np.zeros((self.n_reds, self.max_observed_allies))
        self.distance_observed_enemies = np.zeros((self.n_reds, self.max_observed_enemies))
        
    def _reset_collision_params(self):
        """
        重置状态相关变量
        """
        self.distances_red2blue = None
        self.angles_red2blue = None
        self.red_targets = None
        self.angles_diff_red2blue = None
        
    def _reset_script_indices(self):
        """
        重置脚本索引
        """
        # 定义每种脚本操作的红方智能体数量
        self.script_explode_num = 10
        self.script_collide_num = 10
        self.script_soft_kill_num = 10
        
        # 生成红方智能体的索引并随机打乱
        red_agents_indices = np.random.permutation(self.n_reds)
        
        # 分配不同脚本操作的红方智能体索引
        start_id, end_id = 0, self.script_explode_num
        self.script_explode_ids = red_agents_indices[start_id:end_id]
        start_id, end_id = end_id, end_id + self.script_collide_num
        self.script_collide_ids = red_agents_indices[start_id:end_id]
        start_id, end_id = end_id, end_id + self.script_soft_kill_num
        self.script_soft_kill_ids = red_agents_indices[start_id:end_id]
        
    def _reset_attack_stats(self):
        """
        重置攻击统计参数
        """
        # 当前步被炸死的红方智能体数量
        self.explode_red_num = 0 
        # 当前回合被炸死的红方智能体总数
        self.explode_red_total = 0 
        
        # 当前步被炸死的蓝方智能体数量
        self.explode_blue_num = 0
        # 当前回合被炸死的蓝方智能体总数
        self.explode_blue_total = 0
        
        # 当前步无效自爆的红方智能体数量
        self.invalid_explode_red_num = 0
        # 当前回合无效自爆的红方智能体总数
        self.invalid_explode_red_total = 0
        
        # 当前步被撞死的红方智能体数量
        self.collide_red_num = 0
        # 当前回合被撞死的红方智能体总数
        self.collide_red_total = 0
        
        # 当前步被撞死的蓝方智能体数量
        self.collide_blue_num = 0
        # 当前回合被撞死的蓝方智能体总数
        self.collide_blue_total = 0
        
        # 当前步出界的红方智能体数量
        self.outofbound_red_num = 0
        # 当前回合出界的红方智能体总数
        self.outofbound_red_total = 0
        
        # 当前步出界的蓝方智能体数量
        self.outofbound_blue_num = 0
        # 当前回合出界的蓝方智能体总数
        self.outofbound_blue_total = 0
        
        # 当前时间步被软杀伤的蓝方智能体数量
        self.soft_kill_blue_num = 0
        # 当前回合被软杀伤杀死的蓝方智能体总数
        self.soft_kill_blue_total = 0 

        # 当前时间步自爆的红方智能体数量
        self.red_self_destruction_num = 0
        # 当前回合自爆的红方智能体总数
        self.red_self_destruction_total = 0

        # 当前时间步自爆的蓝方智能体数量
        self.blue_self_destruction_num = 0
        # 当前时间步自爆的蓝方智能体总数
        self.blue_self_destruction_total = 0
        
        # 红方是否自爆
        self.explode_flag = np.zeros(self.n_reds, dtype=bool)
        # 红方是否被自爆
        self.be_exploded_flag = np.zeros(self.n_reds, dtype=bool)
        # 红方是否撞击
        self.collide_flag = np.zeros(self.n_reds, dtype=bool)
        # 红方是否被撞击
        self.be_collided_flag = np.zeros(self.n_reds, dtype=bool)
        # 红方是否无效自爆
        self.invalid_explode_flag = np.zeros(self.n_reds, dtype=bool)
        # 记录每个红方智能体的软杀伤范围内的最近目标
        self.soft_kill_targets = np.full(self.n_reds, -1)
        
        # 记录红方智能体每局开过软杀伤的次数
        self.red_soft_kill_num = np.zeros(self.n_reds)
        # 记录蓝方智能体位于软杀伤范围内的时长
        self.blue_in_soft_kill_time = np.zeros((self.n_reds, self.n_blues))
        # 记录红方智能体开启软杀伤的时长
        self.red_soft_kill_time = np.zeros(self.n_reds)
        
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def init_positions(self):
        """
        随机初始化红蓝双方智能体的位置
        """
        positions = (np.random.rand(self.n_agents, 2) - 0.5) * np.array([self.size_x, self.size_y])
        directions = (np.random.rand(self.n_agents) - 0.5) * 2 * np.pi
        velocities = np.hstack([np.ones(self.n_reds) * self.red_max_vel, np.ones(self.n_blues) * self.blue_max_vel])

        return positions, directions, velocities
    
    def check_boundaries(self):
        """
        检查所有智能体是否超出边界，并更新超出边界的计时和状态
        """
        # 计算区域的半高和半宽
        half_size_x = self.size_x / 2
        half_size_y = self.size_y / 2

        # 检查智能体是否超出了x轴或者y轴的边界
        positions_x = self.positions[:, 0]
        positions_y = self.positions[:, 1]
        
        out_of_bounds_x = (positions_x < -half_size_x) | (positions_x > half_size_x)
        out_of_bounds_y = (positions_y < -half_size_y) | (positions_y > half_size_y)
        
        # 合并x和y方向的边界检查结果，获取最终超出边界的智能体
        out_of_bounds = out_of_bounds_x | out_of_bounds_y
        
        # 对于出界的智能体，增加其出界的时间计数；未出边界的重置为0
        self.out_of_bounds_time[out_of_bounds] += 1
        self.out_of_bounds_time[~out_of_bounds] = 0

        # 检查哪些智能体出界时间超过了允许的最大时间，标记为死亡
        dead_or_not = self.out_of_bounds_time >= self.max_out_of_bounds_time 

        # 计算红色和蓝色智能体中，因出界而死亡的数量
        self.outofbound_red_num = np.sum(dead_or_not[:self.n_reds] & self.red_alives)
        self.outofbound_blue_num = np.sum(dead_or_not[self.n_reds:] & self.blue_alives)

        # 更新因出界死亡的红方和蓝方智能体总数
        self.outofbound_red_total += self.outofbound_red_num
        self.outofbound_blue_total += self.outofbound_blue_num

        # 将出界的智能体标记为死亡
        self.alives[dead_or_not] = False

    def explode(self, i):
        pass

    def collide(self, i):
        pass

    def soft_kill(self, i):
        pass
    
    def split_state(self):
        """
        将整体的状态拆分成红方和蓝方的状态
        """
        # 根据红方的数量 n_reds, 将位置、方向、速度和存活状态拆分为红方和蓝方的部分
        slice_reds = slice(self.n_reds)
        slice_blues = slice(self.n_reds, None)    
        
        self.red_positions = self.positions[slice_reds]
        self.red_directions = self.directions[slice_reds]
        self.red_velocities = self.velocities[slice_reds]
        self.red_alives = self.alives[slice_reds]

        self.blue_positions = self.positions[slice_blues]
        self.blue_directions = self.directions[slice_blues]
        self.blue_velocities = self.velocities[slice_blues]
        self.blue_alives = self.alives[slice_blues]

    def merge_state(self):
        """
        将红方和蓝方的状态合并为整体的状态
        """
        # 使用numpy的vstack和hstack将蓝方和红方的状态合并为整体
        self.positions = np.vstack([self.red_positions, self.blue_positions])
        self.directions = np.hstack([self.red_directions, self.blue_directions])
        self.velocities = np.hstack([self.red_velocities, self.blue_velocities])
        self.alives = np.hstack([self.red_alives, self.blue_alives])
        
    def _perform_attack_actions(self, attack_t, pt):
        """
        执行红方的攻击动作,包括自爆,碰撞和软杀伤
        """
        # 处理攻击动作的掩码
        explode_mask = (attack_t == 1)
        collide_mask = (attack_t == 2)
        soft_kill_mask = (attack_t == 3)
        
        # 执行攻击动作
        self.red_explode(explode_mask)
        pt = self.red_collide(collide_mask, pt)
        pt = self.red_soft_kill(soft_kill_mask, pt)
        
    def _update_red_position_and_direction(self, at, pt):
        """
        基于运动学模型更新红方智能体的位置和方向。
        仅更新存活的智能体。
        """
        # 只更新存活的智能体
        alive_mask = self.red_alives
        
        # 更新方向：确保角度在[-pi, pi]区间内
        self.red_directions[alive_mask] = (
            (self.red_directions[alive_mask] + pt[alive_mask] * self.max_angular_vel * self.dt_time + np.pi) \
            % (2 * np.pi) - np.pi
        )    
        
        # 更新速度：受限于最小和最大速度
        self.red_velocities[alive_mask] = np.clip(
            self.red_velocities[alive_mask] + at[alive_mask] * self.dt_time,
            self.red_min_vel,
            self.red_max_vel
        )

        # 更新位置
        dx = self.red_velocities[alive_mask] * np.cos(self.red_directions[alive_mask]) * self.dt_time
        dy = self.red_velocities[alive_mask] * np.sin(self.red_directions[alive_mask]) * self.dt_time
        self.red_positions[alive_mask] += np.column_stack((dx, dy))
        
    def _update_result(self, terminated, win):
        """
        更新计数器和标志位
        """
        if terminated:
            self.battles_game += 1
            self._episode_count += 1    
        
        if win:
            self.battles_won += 1
            self.win_counted = True
        else:
            self.defeat_counted = True

        bad_transition = False
        if self._episode_steps >= self.episode_limit:
            self.timeouts += 1
            if not win:
                bad_transition = True
        
        return bad_transition
    
    def red_explode(self, explode_mask):
        """
        处理红方智能体的自爆逻辑，包括有效和无效自爆的统计，以及对蓝方智能体的影响
        
        参数：
        explode_mask: 表示哪些红方智能体选择自爆的布尔掩码
        """
        # 过滤掉已死亡的红方智能体，获取有效的自爆掩码
        valid_explode_mask = explode_mask & self.red_alives         # (nreds,)
        
        if not np.any(valid_explode_mask):
            return

        # 记录有效的自爆掩码，用作渲染
        self.red_self_destruction_mask = valid_explode_mask

        # 计算有效自爆红方智能体与所有蓝方智能体之间的距离
        distances_red2blue = self.get_dist()                        # (nreds, nblues)
        
        # 判断蓝方智能体是否在自爆范围内
        blue_in_explode_zone = distances_red2blue[valid_explode_mask] < self.explode_radius     # (nreds_explode, nblues)

        # 标记有效自爆的红方智能体为死亡，并更新自爆数量和总数
        self.red_alives[valid_explode_mask] = False  
        self.red_self_destruction_num = np.sum(valid_explode_mask)
        self.red_self_destruction_total += self.red_self_destruction_num

        # 统计无效自爆（即自爆未击中任何蓝方智能体）的红方智能体数量
        valid_blue_mask = blue_in_explode_zone & self.blue_alives                   # (nreds_explode, nblues)
        valid_explode = np.any(valid_blue_mask, axis=1) # 判断每个红方自爆是否有效    # (nreds_explode,)
        self.invalid_explode_red_num = np.sum(~valid_explode)
        self.invalid_explode_red_total += self.invalid_explode_red_num
        
        # 更新自爆标记和无效自爆标记
        self.explode_flag[valid_explode_mask] = valid_explode
        self.invalid_explode_flag[valid_explode_mask] = ~valid_explode

        # 标记被自爆击中的蓝方智能体为死亡，并统计有效击中的蓝方智能体数量
        blue_explode_mask = np.any(blue_in_explode_zone, axis=0) & self.blue_alives
        self.explode_blue_num = np.sum(blue_explode_mask)
        self.explode_blue_total += self.explode_blue_num
        self.blue_alives[blue_explode_mask] = False
        self.blue_explode_mask = blue_explode_mask

    def red_collide(self, collide_mask, pt):
        """
        处理红方智能体与其目标之间的碰撞逻辑
        
        参数：
        collide_mask: 表示哪些红方智能体尝试与目标碰撞的布尔掩码。
        pt: 红方智能体的状态参数数组，用于更新智能体的方向。
        
        返回:
        更新后的状态数组 pt。
        """
        
        # 过滤掉已经死亡的红方智能体，生成有效的碰撞掩码
        valid_collide_mask = collide_mask & self.red_alives
        
        if not np.any(valid_collide_mask):
            return pt
        
        # 获取红方智能体与蓝方智能体的角度差
        angles_red2blue, angles_diff_red2blue = self.get_angles_diff() #(nreds, nblues)
        
        # 获取红方智能体到蓝方智能体的距离
        distances_red2blue = self.get_dist()
        
        # 判断蓝方智能体是否在红方攻击范围内
        blue_in_attack_zone = (distances_red2blue < self.attack_radius) & (
            np.abs(angles_diff_red2blue) < self.attack_angle/2) # (n_reds, nblues)
        
        # 将不在攻击范围内的智能体距离设置为无限大
        distances_red2blue[~blue_in_attack_zone] = np.inf
        
        # 找个每个红方智能体最近的蓝方智能体
        nearest_blue_id = np.argmin(distances_red2blue, axis=1)
        
        # 如果红方智能体没有攻击范围内的蓝方智能体，目标设为1
        nearest_blue_id[np.all(np.isinf(distances_red2blue), axis=1)] = -1
        
        # 获取红方智能体的目标索引
        red_targets = nearest_blue_id
        
        if not np.any(red_targets != -1):
            return pt

        # 获取有效的红方智能体和目标的索引
        agent_ids = np.where(valid_collide_mask)[0]
        target_ids = red_targets[valid_collide_mask]

        # 计算有效红方智能体与其目标之间的距离
        valid_distances = distances_red2blue[valid_collide_mask, target_ids]

        # 判断哪些红方智能体成功撞击到目标
        collide_success_mask = valid_distances < self.collide_distance
        success_agent_ids = agent_ids[collide_success_mask]
        success_target_ids = target_ids[collide_success_mask]
        
        # 更新撞击成功的红方智能体和蓝方目标智能体的存活状态
        self.red_alives[success_agent_ids] = False
        self.blue_alives[success_target_ids] = False

        # 统计成功撞击的数量并更新总数
        self.collide_blue_num = success_agent_ids.size
        self.collide_blue_total += self.collide_blue_num

        # 记录成功撞击的红方智能体和目标蓝方智能体的索引，用于渲染
        self.red_collide_agent_ids = success_agent_ids
        self.red_collide_target_ids = success_target_ids

        # 更新碰撞标记
        self.collide_flag = np.zeros(self.n_reds, dtype=bool)
        self.collide_flag[success_agent_ids] = True

        # 更新有效碰撞的红方智能体的方向，使其朝向目标
        self.red_directions[valid_collide_mask] = angles_red2blue[valid_collide_mask, red_targets[valid_collide_mask]]
        
        # 将有效碰撞的红方智能体pt值设为0
        pt[valid_collide_mask] = 0

        return pt
    
    def red_soft_kill(self, soft_kill_mask, pt):
        """
        处理红方智能体的软杀伤逻辑，包括目标选择、蓝方智能体的软杀伤判定以及红方智能体的航向调整。
        
        参数:
        soft_kill_mask: 表示哪些红方智能体尝试执行软杀伤的布尔掩码。
        pt: 红方智能体的状态参数数组，用于更新智能体的航向。
        
        返回：
        更新后的状态参数数组 pt。
        """
        # 更新软杀伤的掩码，过滤掉已经死亡或软杀伤次数已经达到上限的红方智能体
        valid_soft_kill_mask = soft_kill_mask & self.red_alives & (self.red_soft_kill_num < self.soft_kill_max_num)
        
        if not np.any(valid_soft_kill_mask):
            return pt   # 如果没有有效的红方智能体需要软杀伤，直接返回
        
        # 计算红方智能体与蓝方智能体之间的距离
        distances_red2blue = self.get_dist()
        
        # 计算红方智能体与蓝方智能体之间的角度差
        angles_red2blue, angles_diff_red2blue = self.get_angles_diff()
        
        # 判断蓝方智能体是否在红方智能体的软杀伤范围内
        blue_in_soft_kill_zone = (distances_red2blue <= self.soft_kill_distance) & (np.abs(angles_diff_red2blue) <= self.soft_kill_angle / 2)
        
        # 对于无效的红方软杀伤掩码，将对应蓝方的软杀伤范围设为False
        blue_in_soft_kill_zone[~valid_soft_kill_mask] = False
        
        # 将不在软杀伤范围内蓝方距离设置为无穷大，以便后续选择最近目标
        distances_red2blue[~blue_in_soft_kill_zone] = np.inf
        
        # 找出每个有效红方智能体的软杀伤目标（最近的蓝方智能体）
        soft_kill_targets = np.argmin(distances_red2blue[valid_soft_kill_mask], axis=1)
        
        # 如果软杀伤范围内没有蓝方智能体，则目标设为-1
        soft_kill_targets[np.all(np.isinf(distances_red2blue[valid_soft_kill_mask]), axis=1)] = -1
        
        # 对于尝试软杀伤的红方智能体，增加其软杀伤次数
        self.red_soft_kill_num[valid_soft_kill_mask] += 1
        
        # 对于有效的软杀伤掩码智能体，记录其软杀伤状态的时长
        self.red_soft_kill_time[valid_soft_kill_mask] = self.soft_kill_max_time
        
        # 存储每个有效红方智能体的软杀伤目标
        self.soft_kill_targets[valid_soft_kill_mask] = soft_kill_targets
        
        # 记录蓝方智能体在软杀伤范围内的时长
        self.blue_in_soft_kill_time[blue_in_soft_kill_zone] += 1
        # TODO：加上角度差的判断
        
        # 初始化蓝方软杀伤的掩码，并标记在软杀伤范围内停留时间超过阈值的蓝方智能体为死亡
        self.blue_soft_kill_mask = np.any(self.blue_in_soft_kill_time >= self.max_time_in_soft_kill, axis=0)
        self.blue_alives[self.blue_soft_kill_mask] = False
        
        # 统计总共被软杀伤的蓝方智能体数量
        self.soft_kill_blue_num = np.sum(self.blue_soft_kill_mask)
        self.soft_kill_blue_total += self.soft_kill_blue_num
        
        # 记录红方开启软杀伤的智能体，用作渲染
        self.red_soft_kill_mask = valid_soft_kill_mask
        
        # 处理处于软杀伤状态的红方智能体
        soft_kill_open = self.red_soft_kill_time > 0 # (n_reds,)
        
        if np.any(soft_kill_open):
            # 更新智能体的方向
            valid_target_mask = soft_kill_open & (self.soft_kill_targets != -1)
            
            valid_indices = np.where(valid_target_mask)[0]
            self.red_directions[valid_indices] = angles_red2blue[valid_target_mask, self.soft_kill_targets[valid_target_mask]]
            pt[valid_indices] = 0
            
            # 软杀伤状态时长减少
            self.red_soft_kill_time[soft_kill_open] -= 1
            
        return pt

    def step(self, actions):
        """
        执行所有智能体的动作，包括攻击和移动，并更新各项状态。
        
        参数：
        actions: 包含所有智能体动作，形状为 (n,3)
                 其中第一列为加速度动作，第二列为航向动作，第三列为攻击动作。        
        """
        # 获取加速度、航向和攻击动作
        at = self.acc_actions[actions[:, 0]]
        pt = self.heading_actions[actions[:, 1]]
        attack_t = actions[:, 2]

        # 执行攻击动作
        for i in range(self.n_agents):
            if not self.alives[i]:
                continue
            
            if attack_t[i] == 1:
                # 自爆逻辑
                pass
            elif attack_t[i] == 2:
                # 碰撞逻辑
                pass
            elif attack_t[i] == 3:
                # 软杀伤逻辑
                pass
            elif attack_t[i] != 0:
                raise ValueError(f"未知的攻击动作：{attack_t[i]}")

        # 更新方向
        self.directions += pt * self.max_angular_vel * self.dt_time
        self.directions = (self.directions + np.pi) % (2 * np.pi) - np.pi
        
        # 更新速度
        self.velocities += at * self.dt_time
        self.velocities[:self.n_reds] = np.clip(self.velocities[:self.n_reds], self.red_min_vel, self.red_max_vel)
        self.velocities[self.n_reds:] = np.clip(self.velocities[self.n_reds:], self.blue_min_vel, self.blue_max_vel)
        
        # 更新位置
        delta_positions = np.column_stack((self.velocities * np.cos(self.directions), 
                                       self.velocities * np.sin(self.directions))) * self.dt_time
        self.positions += delta_positions
        
        # 分离状态并检查边界
        self.split_state()
        self.check_boundaries()

        # 更新步骤计数器
        self._total_steps += 1
        self._episode_steps += 1  

    def update_observed_entities(self, positions, alives, max_num):
        """
        更新红方智能体观测到的实体信息，返回每个红方智能体在通信范围内的最近实体的索引和距离。

        参数:
        positions: 实体的位置数组，形状为 (n_entities, 2)。
        alives: 实体的存活状态布尔数组，形状为 (n_entities,)。
        max_num: 每个红方智能体可以观测到的最大实体数量。

        返回:
        nearest_id: 每个红方智能体在通信范围内的最近实体的索引，形状为 (n_reds, max_num)。
                    若不足 max_num 个实体，则用 -1 填充。
        nearest_dist: 每个红方智能体到最近实体的距离，形状为 (n_reds, max_num)。
                    若不足 max_num 个实体，则用 np.inf 填充。
        """
        # 计算红方智能体与所有实体之间的欧几里得距离
        distance_red2entity = distance.cdist(self.red_positions, positions, 'euclidean')

        # 创建有效性掩码，只考虑存活的红方智能体与存活的实体之间的距离
        valid_mask = self.red_alives[:, np.newaxis] & alives[np.newaxis, :]

        # 将无效的距离设置为无限大，以便后续处理
        distance_red2entity = np.where(valid_mask, distance_red2entity, np.inf)

        # 判断哪些实体在红方智能体的通信范围内
        in_radius_entities = distance_red2entity < self.detection_radius

        # 对于不在通信范围内的实体，将其距离设置为无限大
        distance_red2entity[~in_radius_entities] = np.inf

        # 找到每个红方智能体通信范围内的最近 max_num 个实体的索引
        sorted_indices = np.argsort(distance_red2entity, axis=1)

        # 获取前 max_num 个最近实体的索引
        nearest_id = sorted_indices[:, :max_num]

        # 根据这些索引获取对应的距离
        nearest_dist = distance_red2entity[np.arange(self.n_reds)[:, np.newaxis], nearest_id]

        # 对于距离为无限大的实体，将其索引替换为 -1
        nearest_id[np.isinf(nearest_dist)] = -1

        return nearest_id, nearest_dist

    def get_angles_diff(self):
        """
        计算红方智能体到蓝方智能体的角度差
        """
        # 计算红方智能体到蓝方智能体的方向向量
        delta_red2blue = self.blue_positions[np.newaxis, :, :] - self.red_positions[:, np.newaxis, :]   # nred x nblue x 2

        # 计算红方智能体到蓝方智能体的角度
        angles_red2blue = np.arctan2(delta_red2blue[:, :, 1], delta_red2blue[:, :, 0])                  # nred x nblue

        # 计算红方智能体当前方向与到蓝方智能体的方向的角度差
        angles_diff_red2blue = angles_red2blue - self.red_directions[:, np.newaxis]                     # nred x nblue
        angles_diff_red2blue = (angles_diff_red2blue + np.pi) % (2 * np.pi) - np.pi

        return angles_red2blue, angles_diff_red2blue
    
    def get_dist(self):
        """
        计算红方智能体到蓝方智能体的距离矩阵
        """
        dist = distance.cdist(self.red_positions, self.blue_positions, 'euclidean')
        mask = self.red_alives[:, np.newaxis] & self.blue_alives[np.newaxis, :]
        dist[~mask] = np.inf
        
        return dist
    
    def update_angles_diff(self):
        """
        计算红方智能体到蓝方智能体的角度差
        """
        # 计算红方智能体到蓝方智能体的方向向量
        delta_red2blue = self.blue_positions[np.newaxis, :, :] - self.red_positions[:, np.newaxis, :]   # nred x nblue x 2

        # 计算红方智能体到蓝方智能体的角度
        angles_red2blue = np.arctan2(delta_red2blue[:, :, 1], delta_red2blue[:, :, 0])                  # nred x nblue

        # 计算红方智能体当前方向与到蓝方智能体的方向的角度差
        angles_diff_red2blue = angles_red2blue - self.red_directions[:, np.newaxis]                     # nred x nblue
        angles_diff_red2blue = (angles_diff_red2blue + np.pi) % (2 * np.pi) - np.pi

        return angles_diff_red2blue
            
    def get_obs_size(self):
        """
        获取观测空间的大小，包括自身特征、盟友特征和敌人特征的总大小，以及各个部分的细节大小。

        返回:
        [all_feats, [1, own_feats], [n_allies, n_ally_feats], [n_enemies, n_enemy_feats]]: 
        包含总特征大小，以及自身、盟友和敌人特征的细节信息。
        """
        # 自身特征的大小
        own_feats = self.obs_own_feats_size
        
        # 盟友特征的数量和大小
        n_allies, n_ally_feats = self.max_observed_allies, self.obs_ally_feats_size

        # 敌人特征的数量和大小
        n_enemies, n_enemy_feats = self.max_observed_enemies, self.obs_enemy_feats_size

        # 计算盟友和敌人特征的总大小
        ally_feats = n_allies * n_ally_feats
        enemy_feats = n_enemies * n_enemy_feats

        # 总的特征大小，包括自身特征、盟友特征和敌人特征
        all_feats = own_feats + ally_feats + enemy_feats

        # 返回总特征大小和各部分的详细大小
        return [all_feats, [1, own_feats], [n_allies, n_ally_feats], [n_enemies, n_enemy_feats]]
    
    def get_obs(self):
        """
        获取所有红方智能体的观测，包括自身特征、盟友特征和敌人特征。

        返回:
        obs: 所有红方智能体的观测，形状为 (n_reds, total_obs_size) 的列表。
        """
        # 更新所有红方智能体观察到的盟友和敌人
        self.observed_allies, self.distance_observed_allies = self.update_observed_entities(
            self.red_positions, self.red_alives, self.max_observed_allies)
        self.observed_enemies, self.distance_observed_enemies = self.update_observed_entities(
            self.blue_positions, self.blue_alives, self.max_observed_enemies)
        
        # 更新红方智能体与蓝方智能体之间的角度差
        self.angles_diff_red2blue = self.update_angles_diff()

        # 初始化特征数组
        own_feats = np.zeros((self.n_reds, self.obs_own_feats_size), dtype=np.float32)
        ally_feats = np.zeros((self.n_reds, self.max_observed_allies, self.obs_ally_feats_size), dtype=np.float32)
        enemy_feats = np.zeros((self.n_reds, self.max_observed_enemies, self.obs_enemy_feats_size), dtype=np.float32)

        # 仅处理存活的智能体
        alive_mask = self.red_alives

        # 填充自身特征
        # own_feats[alive_mask, 0:2] = self.red_positions[alive_mask] / np.array([self.size_x / 2, self.size_y / 2])
        own_feats[alive_mask, 2] = (self.red_velocities[alive_mask] - self.red_min_vel) / (self.red_max_vel - self.red_min_vel)
        own_feats[alive_mask, 3] = self.red_directions[alive_mask] / np.pi

        # 填充盟友特征
        valid_allies_mask = self.observed_allies != -1
        ally_ids = self.observed_allies[valid_allies_mask]
        agent_ids, ally_indices = np.where(valid_allies_mask)
        
        ally_positions = self.red_positions[ally_ids]
        ally_feats[agent_ids, ally_indices, 0:2] = (ally_positions - self.red_positions[agent_ids]) / self.detection_radius
        ally_feats[agent_ids, ally_indices, 2] = self.distance_observed_allies[valid_allies_mask] / self.detection_radius
        ally_feats[agent_ids, ally_indices, 3] = self.red_directions[ally_ids] / np.pi

        # 填充敌人特征
        valid_enemies_mask = self.observed_enemies != -1
        enemy_ids = self.observed_enemies[valid_enemies_mask]
        agent_ids, enemy_indices = np.where(valid_enemies_mask)
        
        enemy_positions = self.blue_positions[enemy_ids]
        enemy_feats[agent_ids, enemy_indices, 0:2] = (enemy_positions - self.red_positions[agent_ids]) / self.detection_radius
        enemy_feats[agent_ids, enemy_indices, 2] = self.distance_observed_enemies[valid_enemies_mask] / self.detection_radius
        enemy_feats[agent_ids, enemy_indices, 3] = self.blue_directions[enemy_ids] / np.pi
        enemy_feats[agent_ids, enemy_indices, 4] = self.angles_diff_red2blue[agent_ids, enemy_ids] / (self.attack_angle / 2)

        # 将所有特征合并成一个单一的观测数组
        agents_obs = np.concatenate(
            (
                own_feats,
                ally_feats.reshape(self.n_reds, -1),
                enemy_feats.reshape(self.n_reds, -1)
            ),
            axis=1
        )

        obs = [agents_obs[i, :] for i in range(self.n_reds)]

        # 返回每个智能体的观测
        return obs

    def get_state_size(self):
        """
        获取全局状态空间的大小，包括红方和蓝方智能体的状态信息。
        
        返回:
        [size, [self.n_reds, nf_al], [self.n_blues, nf_en]]: 
        包含总状态大小以及红方和蓝方状态的详细大小。
        """
        # 计算红方和蓝方的状态特征大小
        red_feats_size = self.red_state_size
        blue_feats_size = self.blue_state_size

        # 计算总的状态大小
        total_size = self.n_reds * red_feats_size + self.n_blues * blue_feats_size

        # 返回总状态大小以及每个阵营的详细状态大小
        return [total_size, [self.n_reds, red_feats_size], [self.n_blues, blue_feats_size]]
    
    def get_state(self):
        """
        获取当前全局状态，包括红方和蓝方智能体的规范化位置、速度和方向。
        
        返回:
        state: 扁平化后的全局状态数组，包含所有红方和蓝方智能体的状态信息。
        """
        # 规范化红方和蓝方的位置
        normalized_red_positions = self.red_positions / [self.size_x / 2, self.size_y / 2]
        normalized_blue_positions = self.blue_positions / [self.size_x / 2, self.size_y / 2]

        # 规范化红方和蓝方的速度
        normalized_red_velocities = (self.red_velocities - self.red_min_vel) / (self.red_max_vel - self.red_min_vel)
        normalized_blue_velocities = (self.blue_velocities - self.blue_min_vel) / (self.blue_max_vel - self.blue_min_vel)

        # 规范化红方和蓝方的方向
        normalized_red_directions = self.red_directions / np.pi
        normalized_blue_directions = self.blue_directions / np.pi

        # 初始化状态数组并填充红方状态
        red_state = np.zeros((self.n_reds, self.red_state_size), dtype=np.float32)
        alive_reds = self.red_alives.astype(bool)
        red_state[alive_reds, 0:2] = normalized_red_positions[alive_reds]
        red_state[alive_reds, 2] = normalized_red_velocities[alive_reds]
        red_state[alive_reds, 3] = normalized_red_directions[alive_reds]

        # 初始化状态数组并填充蓝方状态
        blue_state = np.zeros((self.n_blues, self.blue_state_size), dtype=np.float32)
        alive_blues = self.blue_alives.astype(bool)
        blue_state[alive_blues, 0:2] = normalized_blue_positions[alive_blues]
        blue_state[alive_blues, 2] = normalized_blue_velocities[alive_blues]
        blue_state[alive_blues, 3] = normalized_blue_directions[alive_blues]

        # 扁平化并连接红方和蓝方的状态数组
        state = np.concatenate((red_state.flatten(), blue_state.flatten()))

        return state
    
    def get_avail_actions(self):
        """
        获取所有红方智能体的可用动作列表，包括加速、航向和攻击动作。

        返回:
        available_actions: 每个红方智能体的可用动作列表，形状为 (n_reds, total_actions)。
        """
        # 获取加速类动作的 available_actions
        available_acc_actions = self.get_avail_acc_actions()

        # 获取航向类动作的 available_actions
        available_heading_actions = self.get_avail_heading_actions()

        # 获取攻击类动作的 available_actions
        available_attack_actions = self.get_avail_attack_actions()

        # 将三类动作拼起来
        agent_avail_actions = np.hstack((available_acc_actions, available_heading_actions, available_attack_actions))

        # 将可用动作转换为每个智能体的列表形式
        available_actions = agent_avail_actions.tolist()

        return available_actions

    def get_avail_acc_actions(self):
        """
        获取红方智能体的可用加速动作。

        返回:
        available_actions: 布尔数组，形状为 (n_reds, acc_action_num)，
                        表示每个红方智能体的各个加速动作是否可用。
        """
        # 初始化所有动作为可用
        available_actions = np.ones((self.n_reds, self.acc_action_num), dtype=bool)

        # 判断哪些智能体的速度已经达到最大或最小值
        max_vel_mask = self.red_velocities >= self.red_max_vel
        min_vel_mask = self.red_velocities <= self.red_min_vel

        # 限制达到最大速度的智能体的加速动作（只能减速或保持）
        if np.any(max_vel_mask):
            available_actions[max_vel_mask, self.acc_action_mid_id + 1:] = False

        # 限制达到最小速度的智能体的减速动作（只能加速或保持）
        if np.any(min_vel_mask):
            available_actions[min_vel_mask, :self.acc_action_mid_id] = False

        return available_actions

    def get_avail_heading_actions(self):
        """
        获取红方智能体的可用航向动作。

        返回:
        available_actions: 布尔数组，形状为 (n_reds, heading_action_num)，
                        表示每个红方智能体的各个航向动作是否可用。
        """
        # 初始化所有航向动作为可用
        available_actions = np.ones((self.n_reds, self.heading_action_num), dtype=bool)

        # 判断哪些智能体的位置超出缓冲边界
        out_of_bounds_x = (self.red_positions[:, 0] < -self.half_size_x) | (self.red_positions[:, 0] > self.half_size_x)
        out_of_bounds_y = (self.red_positions[:, 1] < -self.half_size_y) | (self.red_positions[:, 1] > self.half_size_y)
        out_of_bounds = out_of_bounds_x | out_of_bounds_y

        # 获取超出边界的智能体索引
        out_of_bounds_indices = np.where(out_of_bounds)[0]

        # 如果没有智能体超出边界，返回默认的可用动作
        if out_of_bounds_indices.size == 0:
            return available_actions

        # 计算超出边界的智能体到每个边界线段的向量
        pos_vec = self.red_positions[out_of_bounds_indices, np.newaxis, :] - self.cache_bounds[:, 0, :]

        # 计算向量的单位向量
        pos_unitvec = pos_vec / self.bounds_len[:, np.newaxis]

        # 计算智能体位置在每条线段上的投影比例
        t = np.einsum('nij,ij->ni', pos_unitvec, self.bounds_unitvec)

        # 将 t 限制在 [0, 1] 范围内，确保投影点在线段上
        t = np.clip(t, 0.0, 1.0)

        # 计算每条线段上距离智能体最近的点的坐标
        nearest = self.cache_bounds[:, 0, :] + t[:, :, np.newaxis] * self.bounds_vec[np.newaxis, :, :]

        # 计算智能体当前位置到最近点的距离
        nearest_dist = np.linalg.norm(self.red_positions[out_of_bounds_indices, np.newaxis, :] - nearest, axis=2)

        # 找到每个智能体距离最近的线段的索引
        nearest_id = np.argmin(nearest_dist, axis=1)

        # 获取每个智能体最近的目标点
        nearest_target = nearest[np.arange(out_of_bounds_indices.size), nearest_id]

        # 计算智能体的期望方向
        desired_directions = np.arctan2(nearest_target[:, 1] - self.red_positions[out_of_bounds_indices, 1],
                                        nearest_target[:, 0] - self.red_positions[out_of_bounds_indices, 0])
        
        # 计算当前方向到期望方向的角度差
        angles_diff = (desired_directions - self.red_directions[out_of_bounds_indices] + np.pi) % (2 * np.pi) - np.pi
        angles_diff_threshold = self.max_angular_vel * self.dt_time

        # 如果角度差大于阈值，限制只能选择右转动作（负号表示逆时针）
        mask_pos = angles_diff >= angles_diff_threshold
        available_actions[out_of_bounds_indices[mask_pos], :self.heading_action_mid_id + 1] = False

        # 如果角度差小于负的阈值，限制只能选择左转动作（正号表示顺时针）
        mask_neg = angles_diff <= -angles_diff_threshold
        available_actions[out_of_bounds_indices[mask_neg], self.heading_action_mid_id:] = False

        return available_actions


    def get_avail_attack_actions(self):
        """
        获取红方智能体的可用攻击动作。

        返回:
        available_actions: 布尔数组，形状为 (n_reds, attack_action_num)，
                        表示每个红方智能体的各个攻击动作是否可用。
        """
        # 初始化有效动作数组
        available_actions = np.ones((self.n_reds, self.attack_action_num), dtype=bool)

        # 判断自爆动作的可用性
        available_actions[:, 1] = self.get_avail_explode_action()

        # 判断撞击动作的可用性
        available_actions[:, 2] = self.get_avail_collide_action()

        # 软杀伤动作默认不可用
        available_actions[:, 3] = False

        return available_actions

    def get_avail_collide_action(self):
        """
        获取红方智能体的可用碰撞动作
        
        返回:
        available_actions: 布尔数组，形状为 (n_reds, )，
                           表示每个智能体的碰撞动作是否可用
        """
        
        # 获取红方智能体与蓝方智能体的角度差
        angles_red2blue, angles_diff_red2blue = self.get_angles_diff() #(nreds, nblues)
        
        # 获取红方智能体到蓝方智能体的距离
        distances_red2blue = self.get_dist()
        
        # 判断蓝方智能体是否在红方攻击范围内
        blue_in_attack_zone = (distances_red2blue < self.attack_radius) & (
            np.abs(angles_diff_red2blue) < self.attack_angle/2) # (n_reds, nblues)
        
        # 将不在攻击范围内的智能体距离设置为无限大
        distances_red2blue[~blue_in_attack_zone] = np.inf
        
        # 找个每个红方智能体最近的蓝方智能体
        nearest_blue_id = np.argmin(distances_red2blue, axis=1)
        
        # 如果红方智能体没有攻击范围内的蓝方智能体，目标设为1
        nearest_blue_id[np.all(np.isinf(distances_red2blue), axis=1)] = -1
        
        # 获取红方智能体的目标索引
        red_targets = nearest_blue_id

        available_actions = (red_targets != -1)

        return available_actions

    def get_avail_explode_action(self):
        """
        获取红方智能体的可用自爆动作
        
        返回:
        available_actions: 布尔数组，形状为 (n_reds, )，
                           表示每个智能体的自爆动作是否可用
        """
        # 计算每个红方智能体与每个蓝方智能体之间的距离
        distances_red2blue = self.get_dist()
        
        # 红方智能体自爆范围内的蓝方智能体
        blue_in_red_explode_zone = distances_red2blue < self.can_explode_radius # (nr, nb)

        # distances_red2blue里面已经过滤掉了死亡的智能体
        available_actions = np.any(blue_in_red_explode_zone, axis=1)

        return available_actions
    
    def random_policy_red(self):
        """
        为红方智能体生成随机策略，包括加速、航向和攻击动作。

        返回:
        actions: 动作数组，形状为 (n_reds, 3)，每行包含一个智能体的加速、航向和攻击动作。
        """
        # 随机选择加速动作
        acc_action = np.random.randint(0, self.acc_action_num, size=self.n_reds)

        # 随机选择航向动作
        heading_action = np.random.randint(0, self.heading_action_num, size=self.n_reds)

        # 按概率随机选择攻击动作
        attack_probabilities = np.array([0.9, 0.02, 0.08, 0])  # 攻击动作的概率分布
        attack_action = np.random.choice(np.arange(self.attack_action_num), size=self.n_reds, p=attack_probabilities)

        # 将加速、航向和攻击动作组合成一个动作数组
        actions = np.column_stack((acc_action, heading_action, attack_action))

        return actions
    
    def transform_position(self, position):
        """
        将单个世界坐标转换为屏幕坐标。

        参数:
        postion: 世界坐标中的一个点，形状为(2, )

        返回:
        tranformed_postion: 屏幕坐标中的对应点，形状为(2, )
        """        

        # 转换世界坐标到屏幕坐标
        transformed_position = ((position - self.screen_offset) * self.direction_scale * self.scale_factor).astype(int)
        
        return transformed_position
    
    def transform_positions(self):
        """
        将所有世界坐标转换为屏幕坐标。

        返回:
        transformed_positions: 屏幕坐标中的所有点，形状为 (n, 2)。
        """
        # 只转换存活的智能体坐标
        alive_mask = np.hstack((self.red_alives, self.blue_alives))
        # 转换所有世界坐标到屏幕坐标
        self.transformed_positions[alive_mask] = ((self.positions[alive_mask] - self.screen_offset) * self.direction_scale * self.scale_factor).astype(int)
        
    def close(self):
        """
        关闭仿真
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self._generate_video()
            
        if self.save_sim_data and self.sim_data is not None:
            if len(self.sim_data) != 0:
                self._save_data_to_csv()
    
    def _generate_video(self):
        """
        基于保存的每一帧渲染，生成视频
        """   
        # 视频的保存路径
        save_dir = os.path.join(self.directory, f'result/videos/{self.scenario_name}/{self.map_name}/{self.timestamp}')
        os.makedirs(save_dir)
        
        # 视频文件名
        save_file = f'{save_dir}/{self.scenario_name}_video_{self.timestamp}.mp4'
                 
        # 生成视频的命令
        command = ['ffmpeg', '-f', 'image2', '-framerate', f'{self.framerate}', '-i', 
                   f'{self.frame_dir}/frame_%06d.png', 
                   f'{save_file}']

        # 执行命令
        result = subprocess.run(command, capture_output=True, text=True)

        # 打印标准输出
        print(result.stdout)

        # 打印错误输出
        print(result.stderr)
        
        # 打印文件存路径
        print(f"渲染视频：{save_file}")
        
    def _save_data_to_csv(self):
        # 轨迹数据保存数据
        save_dir = f'./result/data/{self.scenario_name}/{self.map_name}/{self.timestamp}'
        os.makedirs(save_dir)
        
        filename = os.path.join(save_dir, f'{self.scenario_name}_data_{self.timestamp}.csv')
        append_to_csv(self.sim_data, filename)
        
        # 打印文件存储路径
        print(f"轨迹数据: {filename}")
            
    def render_explode(self):
        """
        渲染红方和蓝方智能体的爆炸特效，包括自爆和被爆效果。
        """
        red_color = (255, 0, 0)
        blue_color = (0, 0, 225)
        
        # 处理红方智能体的爆炸效果
        for i in range(self.n_reds):
            self._render_explosion(i, red_color, self.red_self_destruction_mask[i], self.red_explode_mask[i],
                                    self.red_self_destruction_render, self.red_explosion_frames_remaining)
        
        # 处理蓝方智能体的爆炸效果
        for i in range(self.n_blues):
            self._render_explosion(i, blue_color, self.blue_self_destruction_mask[i], self.blue_explode_mask[i],
                                    self.blue_self_destruction_render, self.blue_explosion_frames_remaining, is_blue=True)
        
    def _render_explosion(self, index, color, self_destruction_mask, explode_mask, self_destruction_render, explosion_frames_remaining, is_blue=False):
        """
        处理单个智能体的爆炸效果，包括自爆和被爆效果。
        
        参数:
        index: 智能体的索引
        color: 智能体的颜色
        self_destruction_mask: 自爆掩码
        explode_mask: 被爆掩码
        self_destruction_render: 自爆渲染标志
        explosion_frames_remaining: 爆炸剩余帧数
        is_blue: 是否为蓝方，默认为False。
        """
        
        offset = self.n_reds if is_blue else 0
        
        # 自爆处理
        if self_destruction_mask:
            # 设置自爆渲染标志
            self_destruction_render[index] = True
            # 初始化自爆需要渲染的帧数
            explosion_frames_remaining[index] = self.explode_render_frames
            
        # 被爆处理
        if explode_mask:
            # 初始化被爆需要渲染的帧数
            explosion_frames_remaining[index] = self.explode_render_frames
            
        # 渲染爆炸效果
        if explosion_frames_remaining[index] > 0:
            # 渲染自爆范围
            if self_destruction_render[index]:
                pygame.draw.circle(self.screen, color, self.transformed_positions[index+offset], radius=self.explode_radius, width=2)
                
            # 渲染爆炸痕迹
            pygame.draw.circle(self.screen, color, self.transformed_positions[index+offset], radius=5)
            
            # 更新剩余帧数
            explosion_frames_remaining[index] -= 1
            
    def render_collide(self):
        """
        渲染红方和蓝方智能体的碰撞特效，包括撞击位置和碰撞效果。        
        """
        # 处理红方智能体的碰撞效果
        if self.red_collide_agent_ids.size > 0:
            self._process_collision(self.red_collide_agent_ids, self.red_collide_target_ids + self.n_reds)
        
        # 处理蓝方智能体的碰撞效果
        if self.blue_collide_agent_ids.size > 0:
            self._process_collision(self.blue_collide_agent_ids + self.n_reds, self.blue_collide_target_ids)
        
        # 渲染智能体的碰撞效果
        for i in range(self.n_agents):
            if self.collision_frames_remaining[i] > 0:
                self._render_collision(i)
                self.collision_frames_remaining[i] -= 1
        
    def _process_collision(self, agent_ids, target_ids):
        """
        处理单个阵营的碰撞效果，包括更新碰撞位置和帧数
        
        参数:
        agent_ids: 撞击的智能体ID列表
        target_ids: 被撞击的目标智能体ID列表
        is_blue: 是否为蓝方，默认为False。
        """
        self.collide_success_target_id[agent_ids] = target_ids
        self.collision_frames_remaining[agent_ids] = self.collide_render_frames
            
    def _render_collision(self, index):
        """
        渲染单个智能体的碰撞效果，包括绘制碰撞位置和箭头
        
        参数：
        index: 智能体的索引
        """
        red_color = (255, 0, 0)
        blue_color = (0, 0, 255)
        
        agent_id, target_id = index, self.collide_success_target_id[index]
        agent_color = red_color if agent_id < self.n_reds else blue_color
        target_color = blue_color if agent_id < self.n_reds else red_color
        
        # 获取碰撞方和被碰撞方的位置 
        start_pos = self.transformed_positions[agent_id]
        end_pos = self.transformed_positions[target_id]
       
        # 绘制碰撞位置
        pygame.draw.circle(self.screen, agent_color, start_pos, radius=3)
        pygame.draw.circle(self.screen, target_color, end_pos, radius=3)
        
        # 绘制碰撞箭头
        draw_arrow(self.screen, start_pos, end_pos, color=agent_color)

    def render_soft_kill(self):
        """
        渲染于红方和蓝方智能体的软杀伤特效
        """
        red_color = (255, 0, 0)
        blue_color = (0, 0, 255)
        
        # 渲染红方智能体的软杀伤效果
        for i in range(self.n_reds):
            self._render_soft_kill(
                i,
                self.red_soft_kill_mask,
                self.red_soft_kill_frames_remaining,
                red_color,
                is_red=True
            )
        
        # 渲染蓝方智能体的软杀伤效果
        for i in range(self.n_blues):
            self._render_soft_kill(
                i,
                self.blue_soft_kill_mask,
                self.blue_soft_kill_frames_remaining,
                blue_color,
                is_red=False
            )
        
    def _render_soft_kill(self, index, soft_kill_mask, frames_remaining, color, is_red=False):
        """
        渲染单个智能体的软杀伤特效。
        
        参数：
        index: 智能体的索引
        soft_kill_mask: 软杀伤的掩码数组
        frames_remaining: 剩余渲染帧数组
        color: 渲染的颜色
        is_red: 是否为红方
        """
        
        offset = 0 if is_red else self.n_reds
        
        # 处理软杀伤的逻辑
        if soft_kill_mask[index]:
            if is_red:
                frames_remaining[index] = self.soft_kill_max_time
            else:
                frames_remaining[index] = self.explode_render_frames
                self.soft_kill_transformed_positions[index+offset] = self.transformed_positions[index + offset]
        
        # 渲染软杀伤的效果
        if frames_remaining[index] > 0:
            if is_red and self.red_alives[index]:
                start_angle = self.red_directions[index] - self.soft_kill_angle / 2
                end_angle = self.red_directions[index] + self.soft_kill_angle / 2
                
                # 渲染扇形
                draw_sector(self.screen, self.transformed_positions[index], self.soft_kill_distance, start_angle, end_angle, color)
            else:
                # 渲染位置
                pygame.draw.circle(self.screen, color, self.soft_kill_transformed_positions[index+offset], radius=5)
            
            # 更新渲染的帧数
            frames_remaining[index] -= 1
            
    def render(self, mode='human'):
        """
        使用 Pygame 渲染环境和智能体的状态
        """
        # 初始化 Pygame 相关内容
        if self.screen is None:
            self._init_pygame()
        
        # 填充背景颜色
        self.screen.fill((255, 255, 255))
        
        # 渲染场景相关的元素
        self._render_scenario()
        
        # 渲染智能体
        self._render_planes()
        
        # 渲染文本信息
        self._render_text()
        
        # 渲染动作效果
        self._render_action()
        
        # 更新屏幕
        pygame.display.flip()
        
        # 保存当前帧
        self._save_frame()
        
    def _init_pygame(self):
        """
        初始化 Pygame 相关的内容，只在第一次渲染时调用
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Swarm UAV Confrontation Environment on {self.scenario_name} Scenario")
        
        # 加载并缩放飞机贴图
        self._load_and_scale_images()
        
        # 将场景中的相关坐标转换到屏幕坐标
        self._transformer_coordinate()
        
        # 初始化字体
        self.font = pygame.font.SysFont(None, 36)

        # 创建帧保存目录
        self.frame_dir = os.path.join(self.directory, f'result/frames/{self.scenario_name}/{self.map_name}/{self.timestamp}')
        os.makedirs(self.frame_dir)
        
    def _load_and_scale_images(self):
        """
        加载并缩放红方和蓝方的飞机贴图
        """
        # 加载飞机贴图
        red_plane_img = pygame.image.load(f"{self.directory}/png/red_plane_s.png").convert_alpha()
        blue_plane_img = pygame.image.load(f"{self.directory}/png/blue_plane_s.png").convert_alpha()
        
        # 缩放飞机贴图
        scale_factor = 0.2
        self.red_plane_img = pygame.transform.scale(red_plane_img, (
            int(red_plane_img.get_width() * scale_factor),
            int(red_plane_img.get_height() * scale_factor)
        ))
        self.blue_plane_img = pygame.transform.scale(blue_plane_img, (
            int(blue_plane_img.get_width() * scale_factor),
            int(blue_plane_img.get_height() * scale_factor)
        ))
        
    def _transformer_coordinate(self):
        """
        将场景中的相关坐标转换到屏幕坐标
        """
        pass
    
    
    def _render_scenario(self):
        """
        渲染跟场景相关的元素
        """
        pass
    
    def _render_planes(self):
        """
        渲染红方和蓝方的智能体（飞机）。
        """
        # 处理位置和角度变换
        self.transform_positions()
        angles = -np.degrees(self.directions)
        
        for i in range(self.n_agents):
            if self.alives[i]:
                image = self.red_plane_img if i < self.n_reds else self.blue_plane_img
                rotated_img = pygame.transform.rotate(image, -angles[i])
                new_rect = rotated_img.get_rect(center=self.transformed_positions[i])
                self.screen.blit(rotated_img, new_rect)
    
    def _render_text(self):
        """
        渲染屏幕上的文本信息
        """
        pass
    
    def _render_action(self):
        """
        渲染动作效果
        """
        self.render_explode()
        self.render_collide()
        self.render_soft_kill()
        
    def _save_frame(self):
        """
        保存当前帧为图像文件。
        """
        frame_path = os.path.join(self.frame_dir, f"frame_{self._total_steps:06d}.png")
        pygame.image.save(self.screen, frame_path)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{current_time}] save {frame_path}')
        
    def scripted_policy_red(self, actions):

        actions = self.script_explode_red(actions)
        actions = self.script_collide_red(actions)
        actions = self.script_soft_kill_red(actions) 
    
        return actions
    
    def script_explode_red(self, actions):

        # 计算红方智能体到蓝方智能体的距离
        distances_red2blue = distance.cdist(self.red_positions[self.script_explode_ids], self.blue_positions, 'euclidean')

        # 若有蓝方飞机在红方爆炸范围内，则自爆
        explode_mask = np.any(distances_red2blue <= self.explode_radius, axis=1)
        actions[self.script_explode_ids[explode_mask], 2] = 1

        if np.any(explode_mask):
            return actions

        # 创建有效性掩码，只考虑存活的红方和蓝方智能体之间的距离
        valid_mask = self.red_alives[self.script_explode_ids, np.newaxis] & self.blue_alives[np.newaxis, :]

        # 将无效的距离设置为无限大
        distances_red2blue = np.where(valid_mask, distances_red2blue, np.inf)

        # 找到每个红方智能体最近的蓝方智能体
        nearest_blue_id = np.argmin(distances_red2blue, axis=1)

        # 如果没有蓝方智能体存活，目标设为-1
        nearest_blue_id[np.all(np.isinf(distances_red2blue), axis=1)] = -1

        # 更新红方智能体的目标
        red_targets = nearest_blue_id

        # 计算智能体的期望方向
        desired_directions = np.arctan2(self.blue_positions[red_targets, 1] - self.red_positions[self.script_explode_ids, 1],
                                        self.blue_positions[red_targets, 0] - self.red_positions[self.script_explode_ids, 0])    # (n)
        
        # 计算当前方向到期望方向的角度
        angles_diff = desired_directions - self.red_directions[self.script_explode_ids] # (n)
        angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi
        angles_diff_threshold = self.max_angular_vel * self.dt_time

        # 如果角度差大于angles_diff_threshold, 则只用-1, -2号动作
        mask_neg = angles_diff <= -angles_diff_threshold
        actions[self.script_explode_ids[mask_neg], 1] = -2

        # 如果角度差小于-angles_diff_threshold，则只用0, 1号动作
        mask_pos = angles_diff >= angles_diff_threshold
        actions[self.script_explode_ids[mask_pos], 1] = 2

        return actions
    
    def script_collide_red(self, actions):
        # 计算红方智能体到蓝方智能体的距离
        distances_red2blue = distance.cdist(self.red_positions[self.script_collide_ids], self.blue_positions, 'euclidean')

        # 若有蓝方飞机在红方撞击范围内，则撞击
        collide_mask = np.any(distances_red2blue <= self.attack_radius, axis=1)
        actions[self.script_collide_ids[collide_mask], 2] = 2

        if np.any(collide_mask):
            return actions

        # 创建有效性掩码，只考虑存活的红方和蓝方智能体之间的距离
        valid_mask = self.red_alives[self.script_collide_ids, np.newaxis] & self.blue_alives[np.newaxis, :]

        # 将无效的距离设置为无限大
        distances_red2blue = np.where(valid_mask, distances_red2blue, np.inf)

        # 找到每个红方智能体最近的蓝方智能体
        nearest_blue_id = np.argmin(distances_red2blue, axis=1)

        # 如果没有蓝方智能体存活，目标设为-1
        nearest_blue_id[np.all(np.isinf(distances_red2blue), axis=1)] = -1

        # 更新红方智能体的目标
        red_targets = nearest_blue_id

        # 计算智能体的期望方向
        desired_directions = np.arctan2(self.blue_positions[red_targets, 1] - self.red_positions[self.script_collide_ids, 1],
                                        self.blue_positions[red_targets, 0] - self.red_positions[self.script_collide_ids, 0])    # (n)
        
        # 计算当前方向到期望方向的角度
        angles_diff = desired_directions - self.red_directions[self.script_collide_ids] # (n)
        angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi
        angles_diff_threshold = self.max_angular_vel * self.dt_time

        # 如果角度差大于angles_diff_threshold, 则只用-1, -2号动作
        mask_neg = angles_diff <= -angles_diff_threshold
        actions[self.script_collide_ids[mask_neg], 1] = -2

        # 如果角度差小于-angles_diff_threshold，则只用0, 1号动作
        mask_pos = angles_diff >= angles_diff_threshold
        actions[self.script_collide_ids[mask_pos], 1] = 2

        return actions

    def script_soft_kill_red(self, actions):

        # 计算红方智能体软杀伤范围内的蓝方智能体
        distances_red2blue = distance.cdist(self.red_positions[self.script_soft_kill_ids], self.blue_positions, 'euclidean')
        angles_red2blue = self.update_angles_diff()

        blue_in_soft_kill_zone = (distances_red2blue <= self.soft_kill_distance) & (angles_red2blue[self.script_soft_kill_ids] <= self.soft_kill_angle / 2)

        # 若有蓝方飞机在红方软杀伤范围内，并且红方智能体开启软杀伤次数未达到上限，则执行软杀伤
        soft_kill_mask = np.any(blue_in_soft_kill_zone, axis=1) & (self.red_soft_kill_num[self.script_soft_kill_ids] < self.soft_kill_max_num)
        actions[self.script_soft_kill_ids[soft_kill_mask], 2] = 3

        if np.any(soft_kill_mask):
            return actions

        return actions
    
    def dump_data(self):
        """
        存储红方智能体的数据，包括存活状态、位置、角度、速度以及执行的动作。
        """
        for i in range(self.n_reds):
            # 创建数据字典，填充通用字段
            data = {
                '时间步': self._episode_steps,
                '阵营': '红方',
                'id': i,
                '存活状态': '存活' if self.red_alives[i] else '死亡',
            }

            if self.red_alives[i]:
                # 存活时，记录位置、角度、速度及动作
                data.update({
                    '位置': np.round(self.red_positions[i], 2).tolist(),
                    '角度': np.round(np.degrees(self.red_directions[i]), 2).tolist(),
                    '速度': np.round(self.red_velocities[i], 2).tolist(),
                    '打击': self.red_action[i, 2],
                    '航向': self.red_action[i, 1],
                    '加速度': self.red_action[i, 0],
                })
            else:
                # 死亡时，填充NaN值
                data.update({
                    '位置': np.nan,
                    '角度': np.nan,
                    '速度': np.nan,
                    '打击': np.nan,
                    '航向': np.nan,
                    '加速度': np.nan,
                })

            # 将数据追加到 agent_data 列表中
            self.sim_data.append(data)
   
def draw_arrow(screen, start_pos, end_pos, color):
    """
    在屏幕上绘制从 start_pos 指向 end_pos 的箭头
    
    参数:
    screen: pygame 显示的屏幕。
    start_pos: 箭头的起始位置 (x,y)。
    end_pos: 箭头的终点位置 (x, y)。
    color: 箭头的颜色
    """
    width = 3 # 箭头线条宽度
    arrow_length = 10  # 箭头三角形的边长
    arrow_angle = math.pi / 6   # 箭头三角形的角度

    # 计算箭头的角度
    angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
    
    # 计算箭头三角形的顶点位置
    arrow_points = [
        end_pos,
        (
            end_pos[0] - arrow_length * math.cos(angle - arrow_angle),
            end_pos[1] - arrow_length * math.sin(angle - arrow_angle)
        ),
        (
            end_pos[0] - arrow_length * math.cos(angle + arrow_angle),
            end_pos[1] - arrow_length * math.sin(angle + arrow_angle)
        )
    ]

    # 画箭头主线条
    pygame.draw.line(screen, color, start_pos, end_pos, width)

    # 画箭头的三角形部分
    pygame.draw.polygon(screen, color, arrow_points)
            
def draw_sector(screen, center, radius, start_angle, end_angle, color):
    """
    在屏幕上绘制一个扇形。

    参数:
    screen: pygame 显示的屏幕。
    center: 扇形的中心点坐标 (x, y)。
    radius: 扇形的半径。
    start_angle: 扇形的起始角度（弧度制）。
    end_angle: 扇形的结束角度（弧度制）。
    color: 扇形的颜色。
    """
    
    # 线条宽度
    width = 3 

    # 扇形的顶点计算
    points = [center]
    angle_range = math.degrees(end_angle - start_angle)
    steps = max(1, int(angle_range))  # 动态计算步数，确保至少绘制一个点
    for step in range(steps + 1):
        angle = start_angle + (end_angle - start_angle) * step / steps
        x = center[0] + int(radius * math.cos(angle))
        y = center[1] - int(radius * math.sin(angle))
        points.append((x, y))

    # 绘制扇形
    pygame.draw.polygon(screen, color, points)

    # 绘制边框（弧形和边线）
    pygame.draw.arc(screen, color, (center[0] - radius, center[1] - radius, radius * 2, radius * 2), start_angle, end_angle, width=width)
    pygame.draw.line(screen, color, center, points[1], width=width)
    pygame.draw.line(screen, color, center, points[-1], width=width)
