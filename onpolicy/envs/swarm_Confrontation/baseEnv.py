# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-20
# @Description: Implementation of base Environment

import os
import sys
import pygame
import numpy as np
import re
import datetime
import subprocess
import csv

from gym import spaces

sys.path.append("/home/ubuntu/sunfeng/MARL/on-policy/")
os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

from scipy.spatial import distance
from onpolicy.utils.multi_discrete import MultiDiscrete
from onpolicy.envs.swarm_Confrontation.plane_params import get_plane_params
from onpolicy.envs.swarm_Confrontation.utils import append_to_csv, draw_arrow, draw_sector

# 定义攻击模式常量
EXPLODE_MODE = 1
SOFTKILL_MODE = 2
INTERFERE_MODE = 3
COLLIDE_MODE = 4

class BaseEnv():
    def __init__(self, args, name=None):
        
        # 初始化场景参数
        self.map_name = args.map_name
        self.plane_name = args.plane_name
        self.scenario_name = args.scenario_name
        self.size_x, self.size_y = 8000, 5000
        self.n_reds, self.n_blues = map(int, re.findall(r'\d+', self.map_name))
        self.n_agents = self.n_reds + self.n_blues
        
        self.use_group = args.use_group     # 是否对智能体按照载荷进行分组
        self.share_action = args.share_action   # 是否将不同载荷的动作当作一个
        
        self.obs_attack = args.obs_attack        # 每个智能体的局部观测中是否包含载荷类型
        self.state_attack = args.state_attack    # 全局的state中是否包含载荷类型
        self.reward_base = args.reward_base      # 打击基地是否给惩罚
        
        self.name = name if name is not None else "Debug"# 实验名称
        
        self.debug = args.debug
        self.save_log = args.save_log
        
        self.only_explode = args.only_explode   # 是否只包含自爆
        self.shuffle = args.shuffle             # 是否打乱飞机类型
        
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
        
        # 设置高价值格子和低价值格子的数量
        self.core_grids_num = 0
        self.comm_grids_num = 0
        
        # 初始化时间步长
        self._episode_count = self._episode_steps = self._total_steps = 0
        
        # 设置仿真时间
        self.dt_time = 1.0
        
        # 初始化距离和角度参数
        self._init_capability_params(args)
        
        # 初始化动作参数
        self._init_action_params()
               
        # 初始化观测和状态参数
        self._init_observation_and_state_params()
        # self._init_obs_space()
        # self._init_state_space()
        
        # 初始化战斗统计参数
        self.battles_game = self.battles_won = self.timeouts = 0
        
        # 初始化仿真数据
        self.sim_data = None
        
        # 初始化屏幕参数
        self._init_screen_params()
        
        # 初始化边界参数
        self._init_boundaries()    
        
        # 初始化攻击方式/载荷
        self._init_attack_mode()
        
        # 初始化log文件
        if self.save_log:
            self.log_file = None
            
    def _init_capability_params(self, args):
        """
        初始化能力参数
        """
        plane_params = get_plane_params(args.plane_name)
        
        can_attack_factor = args.can_attack_factor
        
        # 设置速度参数
        self.red_min_vel, self.red_max_vel = plane_params['red_vel']
        self.blue_min_vel, self.blue_max_vel = plane_params['blue_vel']
        self.max_angular_vel = plane_params['max_angular_vel']
        self.max_turn = self.max_angular_vel * self.dt_time # 每个仿真时间步的最大转向角度
        
        # 圆形通信范围和视野角度
        self.detection_radius = plane_params['detection_radius']
        self.view_angle = plane_params['view_angle']
        
        # 扇形撞击范围和允许碰撞的半径
        self.collide_radius = plane_params['collide_radius']
        self.collide_angle = plane_params['collide_angle']
        self.can_collide_radius = self.collide_radius * can_attack_factor
        
        # 允许自爆的距离和自爆的半径
        self.explode_radius = plane_params['explode_radius']
        self.can_explode_radius = self.explode_radius * can_attack_factor
        
        # 软杀伤的半径和允许软杀伤的距离、次数
        self.softkill_radius = plane_params['softkill_radius']
        self.softkill_prob = plane_params['softkill_prob']
        self.softkill_time = plane_params['softkill_time']
        self.can_softkill_radius = self.softkill_radius * can_attack_factor
        
        # 干扰的范围和持续时间
        self.interfere_radius = plane_params['interfere_radius']
        self.interfere_angle = plane_params['interfere_angle']
        self.interfere_duration = plane_params['interfere_duration']
        self.can_interfere_radius = self.interfere_radius
        self.can_interfere_angle = self.interfere_angle + self.view_angle
        
    def _init_observation_and_state_params(self):
        """
        初始化观测和状态参数
        """
        self.obs_own_feats_size = 4 + (self.attack_mode_num if self.obs_attack else 0)         # x, y, v, phi
        self.obs_ally_feats_size = 4 + (self.attack_mode_num if self.obs_attack else 0)        # x, y, d, v
        self.obs_enemy_feats_size = 5 + (self.attack_mode_num if self.obs_attack else 0)       # x, y, d, theta, phi
        self.obs_size = (self.obs_own_feats_size +
                         self.obs_ally_feats_size +
                         self.obs_enemy_feats_size)
        
        self.red_state_size = 4 + (self.attack_mode_num if self.state_attack else 0)             # x, y, v, phi
        self.blue_state_size = 4 + (self.attack_mode_num if self.state_attack else 0)            # x, y, v, phi
        
        # 局部观测空间
        self.observation_space = [self.get_obs_size()] * self.n_reds
        # 状态空间
        self.share_observation_space = [self.get_state_size()] * self.n_reds
        
    def _init_obs_space(self):
        """
        初始化观测空间
        """    
        self.obs_own_feats_size = 4         # x, y, v, phi
        self.obs_ally_feats_size = 4        # x, y, d, v
        self.obs_enemy_feats_size = 5       # x, y, d, theta, phi
        self.obs_size = (self.obs_own_feats_size +
                         self.obs_ally_feats_size +
                         self.obs_enemy_feats_size)
        
        # 局部观测空间
        self.observation_space = [self.get_obs_size()] * self.n_reds
       
    def _init_state_space(self):
        """
        初始化状态空间
        """
        self.red_state_size = 4             # x, y, v, phi
        self.blue_state_size = 4            # x, y, v, phi
        
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
        # 攻击类动作数目
        self.attack_action_num = 3 if self.share_action else 5
        # self.attack_action_num = 5 # 不攻击、自爆/软毁伤/干扰、碰撞
        # 攻击载荷数量
        self.attack_mode_num = 3 # 自爆/软毁伤/干扰
        
        self.acc_action_mid_id = self.acc_action_num // 2
        self.heading_action_mid_id = self.heading_action_num // 2
        
        self.acc_action_max = 5.0
        self.heading_action_max = 1.0
        
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
        
        # # 多头多离散动作空间
        self.action_space = [MultiDiscrete([[0, self.acc_action_num-1],
                                            [0, self.heading_action_num-1],
                                            [0, self.attack_action_num-1]])] * self.n_reds
        
        # self.action_space = [spaces.Discrete(self.attack_action_num)] * self.n_reds        

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
        
        # 初始化攻击渲染参数
        self.screen_explode_radius = self.explode_radius * self.scale_factor
        self.screen_softkill_radius = self.softkill_radius * self.scale_factor
        self.screen_interfere_radius = self.interfere_radius * self.scale_factor
        self.screen_collide_radius = self.collide_radius * self.scale_factor
        
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
    
    def _init_attack_mode(self):
        """
        为智能体配置不同的攻击模式(自爆、软毁伤、干扰)。
        """
        # 配置比例
        if self.only_explode:
            mode_ratios = {
                EXPLODE_MODE: 1.0,
                SOFTKILL_MODE: 0.0,
                INTERFERE_MODE: 0.0
            }
        else:
            mode_ratios = {
                EXPLODE_MODE: 0.8,
                SOFTKILL_MODE: 0.1,
                INTERFERE_MODE: 0.1
            }

        # 初始化红方和蓝方的攻击模式
        self.red_attack_mode_code, \
        self.red_explode_mode_mask, self.red_softkill_mode_mask, self.red_interfere_mode_mask, \
        self.red_explode_mode_num, self.red_softkill_mode_num, self.red_interfere_mode_num = \
            self._assign_attack_mode(self.n_reds, mode_ratios)
        
        self.blue_attack_mode_code, \
        self.blue_explode_mode_mask, self.blue_softkill_mode_mask, self.blue_interfere_mode_mask, \
        self.blue_explode_mode_num, self.blue_softkill_mode_num, self.blue_interfere_mode_num = \
            self._assign_attack_mode(self.n_blues, mode_ratios)
            
        self.red_attack_mode_mask = self.red_attack_mode_code.astype(bool).T
        
    def _assign_attack_mode(self, num_agents, mode_ratios):
        """
        为指定数量的智能体分配攻击模式。
        
        参数:
        - num_agents: 智能体的数量。
        - mode_ratios: 模式配置比例的字典。
        
        返回:
        - mode_code: 攻击类型的One_hot编码
        - explode_mode_mask: 自爆模式的掩码布尔数组。
        - softkill_mode_mask: 软毁伤模式的掩码布尔数组。
        - interfere_mode_mask: 干扰模式的掩码布尔数组。
        - explode_num: 分配给自爆模式的智能体数量。
        - softkill_num: 分配给软毁伤模式的智能体数量。
        - interfere_num: 分配给干扰模式的智能体数量。
        """
        # 计算每种模式的智能体数量
        explode_num = int(num_agents * mode_ratios[EXPLODE_MODE])
        softkill_num = int(num_agents * mode_ratios[SOFTKILL_MODE])
        interfere_num = num_agents - explode_num - softkill_num  # 剩余数量分配给干扰模式

        # 初始化攻击模式数组并分配
        attack_mode = np.array(
            [EXPLODE_MODE] * explode_num +
            [SOFTKILL_MODE] * softkill_num +
            [INTERFERE_MODE] * interfere_num
        )
        
        # 打乱模式顺序
        if self.shuffle:
            np.random.shuffle(attack_mode)
        
        explode_mode_mask = attack_mode == EXPLODE_MODE
        softkill_mode_mask = attack_mode == SOFTKILL_MODE
        interfere_mode_mask = attack_mode == INTERFERE_MODE
        
        # 类型编码
        mode_code = np.zeros((num_agents, self.attack_mode_num))
        mode_code[explode_mode_mask, 0] = 1
        mode_code[softkill_mode_mask, 1] = 1
        mode_code[interfere_mode_mask, 2] = 1

        # 返回掩码和对应数量
        return (
            mode_code,
            explode_mode_mask, softkill_mode_mask, interfere_mode_mask,
            explode_num, softkill_num, interfere_num
        )
    
    def reset(self):
        """
        重置环境
        """
        # 重置基本计数器和状态变量
        self._reset_basic_state()
                       
        # 重置观测相关变量
        self._reset_observations()
        
        # 重置距离和角度矩阵
        self.update_dist_and_angles() 
        
        # 重置对抗结果
        self.win_counted = False
        self.defeat_counted = False
        
        # 重置攻击统计参数
        self._reset_attack_stats()
        
        # 数据存储
        self.red_action = np.zeros((self.n_reds, 3))
        self.blue_action = np.zeros((self.n_blues, 3))  # 加速度，航向，攻击
        self.sim_data = []
    
    def _reset_render_params(self):
        """
        初始化渲染参数
        """
        self._reset_explosion_rendering()
        self._reset_softkill_rendering()
        self._reset_interfere_rendering()
        self._reset_collision_rendering()
        self._reset_threat_rendering()
        
    def _reset_explosion_rendering(self):
        """
        初始化爆炸渲染参数
        """
        # 爆炸特效持续5帧
        self.explode_render_frames = 5
        
        # 自爆特效还需要渲染的帧数
        self.red_explosion_frames_remaining = np.zeros(self.n_reds)
        self.blue_explosion_frames_remaining = np.zeros(self.n_blues)
        
        # 是否需要渲染主动自爆
        self.red_explode_render = np.zeros(self.n_reds, dtype=bool)
        self.blue_explode_render = np.zeros(self.n_blues, dtype=bool)
        
    def _reset_softkill_rendering(self):
        """
        初始化软杀伤渲染参数
        """
        # 软杀伤特效持续5帧
        self.softkill_render_frames = 5
        
        # 软杀伤特效还需要渲染的帧数
        self.red_softkill_frames_ramaining = np.zeros(self.n_reds)
        self.blue_softkill_frames_remaining = np.zeros(self.n_blues)
        
        # 是否需要渲染软杀伤开启
        self.red_softkill_render = np.zeros(self.n_reds, dtype=bool)
        self.blue_softkill_render = np.zeros(self.n_blues, dtype=bool)
        
    def _reset_interfere_rendering(self):
        """
        初始化干扰渲染参数
        """
        pass
        
    def _reset_collision_rendering(self):
        """
        初始化碰撞渲染参数
        """
        # 碰撞特效需要持续渲染5帧
        self.collide_render_frames = 5
        
        # 碰撞特效还需要渲染的帧数
        self.collision_frames_remaining = np.zeros(self.n_agents)
        
        # 碰撞成功的目标
        self.collide_success_target_id = np.full(self.n_agents, -1, dtype=int)
        
    def _reset_threat_rendering(self):
        """
        初始化威胁区毁伤的渲染参数
        """
        # 需要连续渲染5帧
        self.threat_render_frames = 5
        
        # 碰撞特效还需要渲染的帧数
        self.threat_frames_ramaining = np.zeros(self.n_agents)
    
    def _reset_action_mask_and_counter(self):
        """
        重置攻击/被攻击的掩码及数量(每个时间步都要重置)
        """
        # 当前时间步主动自爆的智能体
        self.red_explode_mask = np.zeros(self.n_reds, dtype=bool)
        self.blue_explode_mask = np.zeros(self.n_blues, dtype=bool)
        
        # 当前时间步被自爆炸死的智能体
        self.red_explode_damage_mask = np.zeros(self.n_reds, dtype=bool)
        self.blue_explode_damage_mask = np.zeros(self.n_blues, dtype=bool)
        
        # 当前时间步开启软杀伤的智能体
        self.red_softkill_mask = np.zeros(self.n_reds, dtype=bool)
        self.blue_softkill_mask = np.zeros(self.n_blues, dtype=bool)
        
        # 当前时间步被软杀伤的智能体
        self.red_softkill_damage_mask = np.zeros(self.n_reds, dtype=bool)
        self.blue_softkill_damage_mask = np.zeros(self.n_blues, dtype=bool)
        
        # 当前时间步开启干扰的智能体
        self.red_interfere_mask = np.zeros(self.n_reds, dtype=bool)
        self.blue_interfere_mask = np.zeros(self.n_blues, dtype=bool)
        
        # 当前时间步被干扰的智能体
        self.red_interfere_damage_mask = np.zeros(self.n_reds, dtype=bool)
        self.blue_interfere_damage_mask = np.zeros(self.n_blues, dtype=bool)
        
        # 当前时间步在威胁区死掉的红方智能体掩码
        self.red_threat_damage_mask = np.zeros(self.n_reds, dtype=bool)
        self.blue_threat_damage_mask = np.zeros(self.n_blues, dtype=bool)
        
        # 当前时间步扫描到高价值/普通格子的智能体掩码
        self.red_scout_core_mask = np.zeros(self.n_reds, dtype=bool)
        self.red_scout_comm_mask = np.zeros(self.n_reds, dtype=bool)
        
        # 当前时间步重复扫描的智能体掩码
        self.red_scout_repeat_mask = np.zeros(self.n_reds, dtype=bool)
        
        # 红方发动撞击的智能体id和目标id
        self.red_collide_agent_ids = np.array([])
        self.red_collide_target_ids = np.array([])

        # 蓝方发动撞击的智能体id和目标id
        self.blue_collide_agent_ids = np.array([])
        self.blue_collide_target_ids = np.array([])
        
        # 红方采用的攻击方式的数量
        self.red_explode_count = 0
        self.red_invalide_explode_count = 0
        self.red_softkill_count = 0
        self.red_interfere_count = 0
        self.red_collide_count = 0
        
        # 蓝方采用攻击方式的数量
        self.blue_explode_count = 0
        self.blue_softkill_count = 0
        self.blue_interfere_count = 0
        self.blue_collide_count = 0
        
        # 被攻击方式毁伤的红方智能体数量
        self.red_explode_damage_count = 0
        self.red_softkill_damage_count = 0
        self.red_interfere_damage_count = 0
        self.red_collide_damage_count = 0
        
        # 被攻击方式毁伤的蓝方智能体数量
        self.blue_explode_damage_count = 0
        self.blue_softkill_damage_count = 0  
        self.blue_interfere_damage_count = 0
        self.blue_collide_damage_count = 0
        
        # 在威胁区死掉的红方智能体数量
        self.red_threat_damage_count = 0
        
        # 在威胁区死掉的蓝方智能体数量
        self.blue_threat_damage_count = 0
        
        # 初始化红方核心区域的被攻击次数
        self.attack_core_num = 0            # 每个时间步红方核心区域被攻击的次数
        
        # 侦察高价值区域/普通区域的格子数
        self.scout_core_num = 0
        self.scout_comm_num = 0
        
        # 摧毁高价值区域的数量
        self.destropy_core_num = 0
    
    def _reset_basic_state(self):
        """
        重置计数器和基本状态变量
        """
        # 重置时间步
        self._episode_steps = 0

        # 重置位置、方向和速度
        self.red_positions, self.red_directions, self.red_velocities, \
            self.blue_positions, self.blue_directions, self.blue_velocities = self.init_positions()
            
        # 重置屏幕坐标
        self.update_transformed_positions()
        
        # 重置存活状态
        self.red_alives = np.ones(self.n_reds, dtype=bool)
        self.blue_alives = np.ones(self.n_blues, dtype=bool)
        
        
    def _reset_observations(self):
        """
        重置观测相关变量
        """
        self.observed_allies = -np.ones((self.n_reds, self.max_observed_allies), dtype=int)
        self.observed_enemies = -np.ones((self.n_reds, self.max_observed_enemies), dtype=int)
        self.distance_observed_allies = np.zeros((self.n_reds, self.max_observed_allies))
        self.distance_observed_enemies = np.zeros((self.n_reds, self.max_observed_enemies))
        
    def _reset_attack_stats(self):
        """
        重置攻击统计参数
        """
        # 红方采用的攻击方式的总数
        self.red_explode_total = 0
        self.red_softkill_total = 0
        self.red_interfere_total = 0
        self.red_collide_total = 0
        
        # 蓝方采用攻击方式的总数
        self.blue_explode_total = 0
        self.blue_softkill_total = 0
        self.blue_interfere_total = 0
        self.blue_collide_total = 0
        
        # 被攻击方式毁伤的红方智能体总数
        self.red_explode_damage_total = 0
        self.red_softkill_damage_total = 0
        self.red_interfere_damage_total = 0
        self.red_collide_damage_total = 0
        
        # 被攻击方式毁伤的蓝方智能体总数
        self.blue_explode_damage_total = 0
        self.blue_softkill_damage_total = 0  
        self.blue_interfere_damage_total = 0
        self.blue_collide_damage_total = 0
        
        # 在威胁区死掉的红方智能体总数
        self.red_threat_damage_total = 0
        
        # 在威胁区死掉的蓝方智能体总数
        self.blue_threat_damage_total = 0
        
        # 初始化当前回合红方核心区域被打击的总次数
        self.attack_core_total = 0          # 当前回合红方高价值区域被打击的总次数
        
        # 侦察高价值区域/普通区域的格子总数
        self.scout_core_total = 0
        self.scout_comm_total = 0
        
        # 重置每个智能体干扰的时长
        self.red_interfere_duration = np.zeros(self.n_reds)
        self.blue_interfere_duration = np.zeros(self.n_blues)
        
        # 重置每个智能体软杀伤的次数
        self.red_softkill_time = np.zeros(self.n_reds)
        self.blue_softkill_time = np.zeros(self.n_blues)
        
        # 记录开启软杀伤时的屏幕位置
        self.softkill_positions = np.zeros((self.n_agents, 2), dtype=int)
        
        self._reset_action_mask_and_counter()
        
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

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
        
    def _perform_attack_actions(self, attack_t):
        """
        执行红方的攻击动作,包括自爆,碰撞和软杀伤
        """
        # 处理攻击动作的掩码
        if self.share_action:
            explode_mask = (attack_t == 1) & self.red_explode_mode_mask         # 自爆
            softkill_mask = (attack_t == 1) & self.red_softkill_mode_mask       # 软杀伤
            interfere_mask = (attack_t == 1) & self.red_interfere_mode_mask     # 干扰
            collide_mask = (attack_t == 2)
        else:
            explode_mask = (attack_t == 1) & self.red_explode_mode_mask         # 自爆
            softkill_mask = (attack_t == 2) & self.red_softkill_mode_mask       # 软杀伤
            interfere_mask = (attack_t == 3) & self.red_interfere_mode_mask     # 干扰
            collide_mask = (attack_t == 4)                                      # 撞击
        
        # 执行攻击动作
        self.red_explode(explode_mask)
        self.red_softkill(softkill_mask)
        self.red_interfere(interfere_mask)
        
        self.red_collide(collide_mask)
        
    def _update_red_position_and_direction(self, at, pt):
        """
        基于运动学模型更新红方智能体的位置和方向。
        仅更新存活的智能体。
        """
        # 只更新存活的智能体
        alive_mask = self.red_alives
        
        # 更新方向：确保角度在[-pi, pi]区间内
        self.red_directions[alive_mask] = (
            (self.red_directions[alive_mask] + pt[alive_mask] * self.max_turn + np.pi) \
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
        # 如果回合未结束，提前返回简化的信息
        if not dones:
            return {
                "battles_won": self.battles_won,
                "battles_game": self.battles_game,
                "battles_draw": self.timeouts,
                'bad_transition': bad_transition,
                'won': self.win_counted,
            }
        
        # 汇总红方和蓝方的损伤信息
        red_kill_total = (
            self.blue_explode_damage_total +
            self.blue_softkill_damage_total +
            self.blue_collide_damage_total
        )
        
        red_damage_total = (
            self.red_explode_damage_total +
            self.red_softkill_damage_total +
            self.red_collide_damage_total
        )

        # 计算核心信息字典
        info =  {
            "episode_count": self._episode_count,
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
            'scout_core_ratio': self.scout_core_total / self.core_grids_num if self.core_grids_num > 0 else 0.0,  # 高价值区域被侦察的比例
            'scout_comm_ratio': self.scout_comm_total / self.comm_grids_num if self.comm_grids_num > 0 else 0.0,  # 普通区域被侦察的比例
            'won': self.win_counted,
            "other": res
        }
        
        # Debug 模式下输出调试信息
        if self.debug and self.name.endswith("eval"):
            # print(f"[Debug in SCE {self.name}/{self.timestamp}]: \n {info} \n")
            
            if self._episode_count == 1:
                self.red_explode_total_list = []
            
            self.red_explode_total_list.append(self.red_explode_damage_total)
            if self._episode_count % 32 == 0:
                print('*'*50, '\n')
                print(np.mean(np.array(self.red_explode_total_list)))
                self.red_explode_total_list = []
        
        # 日志保存
        if self.save_log:
            if self.log_file is None:
                save_dir = os.path.join(self.directory, f'result/log/{self.scenario_name}/{self.map_name}/{self.name}')
                os.makedirs(save_dir, exist_ok=True)
                self.log_file = os.path.join(save_dir, f'{self.scenario_name}_log_{self.timestamp}.csv')
                
                # 写入表头
                with open(self.log_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=info.keys())
                    writer.writeheader()
            
            # 追加写入信息
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=info.keys())
                writer.writerow(info)
            
        return info
    
    def red_explode(self, explode_mask):
        """
        处理红方智能体的自爆逻辑，包括有效和无效自爆的统计，以及对蓝方智能体的影响
        
        参数：
        explode_mask: 表示哪些红方智能体选择自爆的布尔掩码
        """
        # 过滤掉已死亡的红方智能体，获取有效的自爆掩码
        red_explode_mask = explode_mask & self.red_alives         # (nreds,)
        
        if not np.any(red_explode_mask):
            return
        
        # print(f"Step: {self._episode_steps} Explode_mask: {np.where(red_explode_mask)[0]}")
        
        # 判断蓝方智能体是否在自爆范围
        blue_in_explode_zone = (self.distances_red2blue[red_explode_mask] < self.explode_radius) & self.blue_alives
            
        # 更新红方自爆统计和状态
        self.red_explode_mask = red_explode_mask
        self.red_explode_count = np.sum(red_explode_mask)
        self.red_explode_total += self.red_explode_count
        
        self.red_alives[red_explode_mask] = False
        
        # 更新蓝方毁伤统计和状态
        self.blue_explode_damage_mask = np.any(blue_in_explode_zone, axis=0)
        self.blue_explode_damage_count = np.sum(self.blue_explode_damage_mask)
        self.blue_explode_damage_total += self.blue_explode_damage_count
        
        self.blue_alives[self.blue_explode_damage_mask] = False
        
        # 统计无效自爆的数量
        self.red_invalide_explode_count = np.sum(~np.any(blue_in_explode_zone, axis=1))
        
    def red_softkill(self, softkill_mask):
        """
        处理红方智能体的软杀伤逻辑
        
        参数:
        softkill_mask: 表示哪些红方智能体选择软杀伤的布尔掩码
        """
        # 过滤掉已经死亡的红方智能体，获取有效的软杀伤掩码
        red_softkill_mask = softkill_mask & self.red_alives
        
        if not np.any(red_softkill_mask):
            return
        
        # print(f"Step: {self._episode_steps} Softkill_mask: {np.where(red_softkill_mask)[0]}")
        
        # 判断蓝方智能体是否在软杀伤范围内
        blue_in_softkill_zone = (self.distances_red2blue[red_softkill_mask] < self.softkill_radius) & self.blue_alives
        
        # 更新红方软杀伤统计和状态
        self.red_softkill_mask = red_softkill_mask
        self.red_softkill_count = np.sum(red_softkill_mask)
        self.red_softkill_total += self.red_softkill_count
        self.red_softkill_time[red_softkill_mask] += 1
        
        # 更新蓝方毁伤统计和状态
        random_prob = np.random.rand(self.n_blues)
        self.blue_softkill_damage_mask = np.any(blue_in_softkill_zone, axis=0) & (random_prob < self.softkill_prob) # 以60%的概率被软毁伤
        self.blue_softkill_damage_count = np.sum(self.blue_softkill_damage_mask)
        self.blue_softkill_damage_total += self.blue_softkill_damage_count
        
        self.blue_alives[self.blue_softkill_damage_mask] = False
        
        # 记录软杀伤的位置
        mask = np.hstack((self.red_softkill_mask, self.blue_softkill_damage_mask))
        self.softkill_positions[mask] = self.transformed_positions[mask]
        
    def red_interfere(self, interfere_mask):
        """
        处理红方只能提的干扰逻辑
        
        参数:
        interfere_mask: 表示哪些红方智能体选择干扰的布尔掩码
        """
        # 过滤掉已经死亡的红方智能体，获取有效的干扰掩码
        red_interfere_mask = interfere_mask & self.red_alives
        
        if not np.any(red_interfere_mask):
            return
        
        # print(f"Step: {self._episode_steps} Interfere_mask: {np.where(red_interfere_mask)[0]}")
        
        # 判断蓝方智能体是否在干扰范围内
        blue_in_interfere_zone = (
            (self.distances_red2blue[red_interfere_mask] < self.interfere_radius) & 
            (np.abs(self.angles_diff_red2blue[red_interfere_mask]) < self.interfere_angle / 2) &
            self.blue_alives
        )
        
        # 更新红方干扰统计和状态
        self.red_interfere_mask = red_interfere_mask
        # self.red_interfere_count = np.sum(self.red_interfere_duration[red_interfere_mask] == 0) # 首次开干扰
        # self.red_interfere_total += self.red_interfere_count
        self.red_interfere_count = np.sum(self.red_interfere_mask)
        self.red_interfere_total += np.sum(self.red_interfere_duration[red_interfere_mask] == 0)
        
        self.red_interfere_duration[red_interfere_mask] += 1
        
        # 更新蓝方干扰统计和状态
        self.blue_interfere_damage_mask = np.any(blue_in_interfere_zone, axis=0)
        self.blue_interfere_damage_count = np.sum(self.blue_interfere_damage_mask)
        self.blue_interfere_damage_total += self.blue_interfere_damage_count
            
    def red_collide(self, collide_mask):
        """
        处理红方智能体与其目标之间的碰撞逻辑
        
        参数：
        collide_mask: 表示哪些红方智能体尝试与目标碰撞的布尔掩码。
        """
        
        # 过滤掉已经死亡的红方智能体，生成有效的碰撞掩码
        red_collide_mask = collide_mask & self.red_alives
        
        if not np.any(red_collide_mask):
            return
        
        # 获取有效的红方智能体和目标的索引
        agent_ids = np.where(red_collide_mask)[0]
        target_ids = self.red_collide_targets[red_collide_mask]

        # 计算有效红方智能体与其目标之间的距离
        valid_distances = self.distances_red2blue[red_collide_mask, target_ids]

        # 判断哪些红方智能体成功撞击到目标
        success_mask = (valid_distances < self.collide_radius) & self.blue_alives[target_ids]
        
        if not np.any(success_mask):
            return
        
        success_agent_ids = agent_ids[success_mask]
        success_target_ids = target_ids[success_mask]
        
        # 更新撞击成功的红方智能体和蓝方目标智能体的存活状态
        self.red_alives[success_agent_ids] = False
        self.blue_alives[success_target_ids] = False

        # 更新统计数据
        successful_collisions = success_agent_ids.size
        
        self.red_collide_count = successful_collisions
        self.red_collide_total += successful_collisions
        
        self.blue_collide_damage_count = successful_collisions
        self.blue_collide_damage_total += successful_collisions
        
        # 记录成功撞击的红方智能体和目标蓝方智能体的索引，用于渲染
        self.red_collide_agent_ids = success_agent_ids
        self.red_collide_target_ids = success_target_ids
    
    def step(self):
        """
        执行所有智能体的动作，包括攻击和移动，并更新各项状态。
        
        参数：
        actions: 包含所有智能体动作，形状为 (n,3)
                 其中第一列为加速度动作，第二列为航向动作，第三列为攻击动作。        
        """
        pass
    
    def is_red_in_threat_zone(self):
        """
        判断红方智能体是否在威胁区域
        """
        pass
    
    def red_step(self, actions):
        """
        执行所有智能体的动作，包括攻击和移动，并更新各项状态。
        
        参数：
        actions: 包含所有智能体动作，形状为 (n,3)
                 其中第一列为加速度动作，第二列为航向动作，第三列为攻击动作。        
        """
        # 解析红方智能体的动作
        at = self.acc_actions[actions[:, 0]]
        pt = self.heading_actions[actions[:, 1]]
        attack_t = actions[:, 2]
            
        # 执行攻击动作
        self._perform_attack_actions(attack_t)
        
        # 更新红方的位置和方向
        self._update_red_position_and_direction(at, pt)
        
        # 判断红方智能体是否在威胁区
        self.is_red_in_threat_zone()
        
        # 如果需要保存仿真数据，则记录红方的动作信息
        if self.save_sim_data:
            self.red_action = np.stack([at, pt * self.max_turn * 180 / np.pi, attack_t], axis=-1)
            
        # 更新距离矩阵和角度矩阵
        self.update_dist_and_angles()
    
    def update_observed_entities(self, distances, max_num):
        """
        更新红方智能体观测到的实体信息，返回每个红方智能体在通信范围内的最近实体的索引和距离。

        参数:
        distances: 红方与其它实体之间的距离矩阵，形状为 (nreds, M)。
        max_num: 每个红方智能体可以观测到的最大实体数量。

        返回:
        nearest_id: 每个红方智能体在通信范围内的最近实体的索引，形状为 (n_reds, max_num)。
                    若不足 max_num 个实体，则用 -1 填充。
        nearest_dist: 每个红方智能体到最近实体的距离，形状为 (n_reds, max_num)。
                    若不足 max_num 个实体，则用 np.inf 填充。
        """
        # 计算红方智能体与其它实体之间的欧几里得距离
        distance_red2entity = distances.copy()

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
    
    def _calculate_angles(self, delta_vectors):
        """
        计算给定方向向量的角度。

        参数:
        - delta_vectors: 方向向量的数组，形状为 (N, M, 2)，其中 N 和 M 是两个智能体群体的数量。

        返回:
        - angles: 对应于每个方向向量的角度，范围为 [-π, π]。
        """
        return (np.arctan2(delta_vectors[:, :, 1], delta_vectors[:, :, 0]) + np.pi) % (2 * np.pi) - np.pi
    
    def _calculate_angle_diff(self, angles, directions):
        """
        计算智能体当前方向与目标方向之间的角度差。

        参数:
        - angles: 目标方向的角度数组，形状为 (N, M)。
        - directions: 当前智能体的方向数组，形状为 (N,)。

        返回:
        - angles_diff: 角度差数组，形状为 (N, M)，范围为 [-π, π]。
        """
        angles_diff = angles - directions[:, np.newaxis]
            
        return (angles_diff + np.pi) % (2 * np.pi) - np.pi
    
    def _calculate_angles_and_diffs(self, delta_vectors, directions_A):
        """
        计算给定方向向量的角度以及角度差。

        参数:
        - delta_vectors: 方向向量的数组，形状为 (N, M, 2)。
        - directions_A: 当前智能体的方向数组，形状为 (N,)。

        返回:
        - angles: 对应于每个方向向量的角度，范围为 [-π, π]。
        - angles_diff: 角度差数组，形状为 (N, M)，范围为 [-π, π]。
        """
        angles = (np.arctan2(delta_vectors[:, :, 1], delta_vectors[:, :, 0]) + np.pi) % (2 * np.pi) - np.pi
        angles_diff = (angles - directions_A[:, np.newaxis] + np.pi) % (2 * np.pi) - np.pi
        return angles, angles_diff
    
    def _calculate_dist_and_angles(self, positions_A, positions_B, directions_A, directions_B, alives_A, alives_B, is_same=False):
        """
        计算智能体群体A到群体B的距离、角度和角度差。

        参数:
        - positions_A: 群体A的坐标数组，形状为 (N, 2)。
        - positions_B: 群体B的坐标数组，形状为 (M, 2)。
        - directions_A: 群体A的方向数组，形状为 (N,)。
        - directions_B: 群体B的方向数组，形状为 (M,)。
        - alives_A: 群体A的存活状态布尔数组，形状为 (N,)。
        - alives_B: 群体B的存活状态布尔数组，形状为 (M,)。

        返回:
        - distances_A2B: A到B的距离矩阵，形状为 (N, M)。
        - angles_A2B: A到B的角度矩阵，形状为 (N, M)。
        - angles_diff_A2B: A到B的角度差矩阵，形状为 (N, M)。
        """
        # 计算方向向量 delta_A2B 和距离矩阵
        delta_A2B = positions_B[np.newaxis, :, :] - positions_A[:, np.newaxis, :]   
        distances_A2B = np.linalg.norm(delta_A2B, axis=2)  # 计算欧几里得距离
        
        # 创建存活掩码，仅对存活的智能体进行计算
        mask = alives_A[:, np.newaxis] & alives_B[np.newaxis, :]
        distances_A2B[~mask] = np.inf
        
        # 计算A到B的角度和角度差
        angles_A2B, angles_diff_A2B = self._calculate_angles_and_diffs(delta_A2B, directions_A)
        
        if is_same:
            np.fill_diagonal(distances_A2B, np.inf)
            np.fill_diagonal(angles_A2B, np.inf)
            np.fill_diagonal(angles_diff_A2B, np.inf)
            
        
        return distances_A2B, angles_A2B, angles_diff_A2B
    
    def update_dist_and_angles(self):
        """
        计算红方到蓝方/蓝方到红方的距离、角度以及角度差
        """
        self.distances_red2blue, self.angles_red2blue, self.angles_diff_red2blue = self._calculate_dist_and_angles(
            self.red_positions, self.blue_positions, self.red_directions, self.blue_directions, self.red_alives, self.blue_alives
        )
        
        self.distances_blue2red, self.angles_blue2red, self.angles_diff_blue2red = self._calculate_dist_and_angles(
            self.blue_positions, self.red_positions, self.blue_directions, self.red_directions, self.blue_alives, self.red_alives
        )
        
        self.distances_red2red, self.angles_red2red, self.angles_diff_red2red = self._calculate_dist_and_angles(
            self.red_positions, self.red_positions, self.red_directions, self.red_directions, self.red_alives, self.red_alives, is_same=True
        )
        
        self.distances_blue2blue, self.angles_blue2blue, self.angles_diff_blue2blue = self._calculate_dist_and_angles(
            self.blue_positions, self.blue_positions, self.blue_directions, self.blue_directions, self.blue_alives, self.blue_alives, is_same=True
        )
            
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
            self.distances_red2red, self.max_observed_allies)
        self.observed_enemies, self.distance_observed_enemies = self.update_observed_entities(
            self.distances_red2blue, self.max_observed_enemies)

        # 初始化特征数组
        own_feats = np.zeros((self.n_reds, self.obs_own_feats_size), dtype=np.float32)
        ally_feats = np.zeros((self.n_reds, self.max_observed_allies, self.obs_ally_feats_size), dtype=np.float32)
        enemy_feats = np.zeros((self.n_reds, self.max_observed_enemies, self.obs_enemy_feats_size), dtype=np.float32)

        # 仅处理存活的智能体
        alive_mask = self.red_alives

        # 填充自身特征
        own_feats[alive_mask, 0:2] = self.red_positions[alive_mask] / np.array([self.size_x / 2, self.size_y / 2])
        own_feats[alive_mask, 2] = (self.red_velocities[alive_mask] - self.red_min_vel) / (self.red_max_vel - self.red_min_vel)
        own_feats[alive_mask, 3] = self.red_directions[alive_mask] / np.pi
        if self.obs_attack:
            own_feats[alive_mask, 4: ] = self.red_attack_mode_code[alive_mask]

        # 填充盟友特征
        valid_allies_mask = self.observed_allies != -1
        ally_ids = self.observed_allies[valid_allies_mask]
        agent_ids, ally_indices = np.where(valid_allies_mask)
        
        ally_positions = self.red_positions[ally_ids]
        ally_feats[agent_ids, ally_indices, 0:2] = (ally_positions - self.red_positions[agent_ids]) / self.detection_radius
        ally_feats[agent_ids, ally_indices, 2] = self.distance_observed_allies[valid_allies_mask] / self.detection_radius
        ally_feats[agent_ids, ally_indices, 3] = self.red_directions[ally_ids] / np.pi
        if self.obs_attack:
            ally_feats[agent_ids, ally_indices, 4: ] = self.red_attack_mode_code[ally_ids]

        # 填充敌人特征
        valid_enemies_mask = self.observed_enemies != -1
        enemy_ids = self.observed_enemies[valid_enemies_mask]
        agent_ids, enemy_indices = np.where(valid_enemies_mask)
        
        enemy_positions = self.blue_positions[enemy_ids]
        enemy_feats[agent_ids, enemy_indices, 0:2] = (enemy_positions - self.red_positions[agent_ids]) / self.detection_radius
        enemy_feats[agent_ids, enemy_indices, 2] = self.distance_observed_enemies[valid_enemies_mask] / self.detection_radius
        enemy_feats[agent_ids, enemy_indices, 3] = self.blue_directions[enemy_ids] / np.pi
        enemy_feats[agent_ids, enemy_indices, 4] = self.angles_diff_red2blue[agent_ids, enemy_ids] / (self.view_angle / 2)
        if self.obs_attack:
            enemy_feats[agent_ids, enemy_indices, 5: ] = self.blue_attack_mode_code[enemy_ids]

        # 将所有特征合并成一个单一的观测数组
        agents_obs = np.concatenate(
            (
                own_feats,
                ally_feats.reshape(self.n_reds, -1),
                enemy_feats.reshape(self.n_reds, -1)
            ),
            axis=1
        )

        # # 将被干扰的智能体的局部观测信息置0
        # agents_obs[self.red_interfere_damage_mask] = 0.0

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
        if self.state_attack:
            red_state[alive_reds, 4: ] = self.red_attack_mode_code[alive_reds]

        # 初始化状态数组并填充蓝方状态
        blue_state = np.zeros((self.n_blues, self.blue_state_size), dtype=np.float32)
        alive_blues = self.blue_alives.astype(bool)
        blue_state[alive_blues, 0:2] = normalized_blue_positions[alive_blues]
        blue_state[alive_blues, 2] = normalized_blue_velocities[alive_blues]
        blue_state[alive_blues, 3] = normalized_blue_directions[alive_blues]
        if self.state_attack:
            blue_state[alive_blues, 4: ] = self.blue_attack_mode_code[alive_blues]

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
        # assert np.sum(~np.any(available_acc_actions, axis=1)) == 0

        # 获取航向类动作的 available_actions
        available_heading_actions = self.get_avail_heading_actions()
        # assert np.sum(~np.any(available_heading_actions, axis=1)) == 0

        # 获取攻击类动作的 available_actions
        available_attack_actions = self.get_avail_attack_actions()
        # assert np.sum(~np.any(available_attack_actions, axis=1)) == 0

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
        available_actions[max_vel_mask, self.acc_action_mid_id + 1:] = False

        # 限制达到最小速度的智能体的减速动作（只能加速或保持）
        available_actions[min_vel_mask, :self.acc_action_mid_id] = False
            
        # 对于被干扰的智能体，只能保持匀速 (self.acc_action_mid_id 号动作可用)
        interfere_mask = self.red_interfere_damage_mask
        available_actions[interfere_mask] = False
        available_actions[interfere_mask, self.acc_action_mid_id] = True

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
        out_of_bounds = (
            (self.red_positions[:, 0] < -self.half_size_x) | 
            (self.red_positions[:, 0] > self.half_size_x) |
            (self.red_positions[:, 1] < -self.half_size_y) | 
            (self.red_positions[:, 1] > self.half_size_y)
        )
        
        # 获取超出边界的智能体索引
        out_of_bounds_indices = np.where(out_of_bounds)[0]

        # 如果没有智能体超出边界，返回默认的可用动作
        if out_of_bounds_indices.size == 0:
            return available_actions

        # 计算超出边界的智能体到每个边界线段的向量和投影点
        pos_vec = self.red_positions[out_of_bounds_indices, np.newaxis, :] - self.cache_bounds[:, 0, :] # 投影向量
        pos_unitvec = pos_vec / self.bounds_len[:, np.newaxis]  # 单位投影向量
        t = np.clip(np.einsum('nij,ij->ni', pos_unitvec, self.bounds_unitvec), 0.0, 1.0)    # 投影比例
        nearest = self.cache_bounds[:, 0, :] + t[:, :, np.newaxis] * self.bounds_vec[np.newaxis, :, :]  # 投影点

        # 计算智能体当前位置到最近点的距离和最近线段的索引
        nearest_dist = np.linalg.norm(self.red_positions[out_of_bounds_indices, np.newaxis, :] - nearest, axis=2)
        nearest_id = np.argmin(nearest_dist, axis=1)

        # 获取每个智能体最近的目标点
        nearest_target = nearest[np.arange(out_of_bounds_indices.size), nearest_id]

        # 计算智能体的期望方向和角度差
        desired_directions = np.arctan2(nearest_target[:, 1] - self.red_positions[out_of_bounds_indices, 1],
                                        nearest_target[:, 0] - self.red_positions[out_of_bounds_indices, 0])
        angles_diff = (desired_directions - self.red_directions[out_of_bounds_indices] + np.pi) % (2 * np.pi) - np.pi

        # 计算动作限制限，限制智能体的左转和右转
        # 1.如果角度差大于阈值，限制只能选择右转动作（负号表示逆时针）
        mask_pos = angles_diff >= self.max_turn
        available_actions[out_of_bounds_indices[mask_pos], :self.heading_action_mid_id + 1] = False

        # 2.如果角度差小于负的阈值，限制只能选择左转动作（正号表示顺时针）
        mask_neg = angles_diff <= -self.max_turn
        available_actions[out_of_bounds_indices[mask_neg], self.heading_action_mid_id:] = False
        
        # 对于被干扰的智能体，只能保持航向 (self.heading_action_mid_id 号动作可用)
        interfere_mask = self.red_interfere_damage_mask
        available_actions[interfere_mask] = False
        available_actions[interfere_mask, self.heading_action_mid_id] = True

        return available_actions

    def get_avail_attack_actions(self):
        """
        获取红方智能体的可用攻击动作。

        返回:
        available_actions: 布尔数组，形状为 (n_reds, attack_action_num)，
                        表示每个红方智能体的各个攻击动作是否可用。
        """
        # 干扰还未结束的掩码
        self.red_interfering = (
            (self.red_interfere_duration > 0) & 
            (self.red_interfere_duration < self.interfere_duration) & 
            self.red_interfere_mode_mask &
            self.red_alives
        )
        
        # 初始化有效动作数组
        available_actions = np.ones((self.n_reds, self.attack_action_num), dtype=bool)
        
        # 不攻击动作的可用性
        available_actions[:, 0] = self.get_avail_no_op_actions()

        # 自爆、软杀伤、干扰和碰撞动作的可用性
        if self.share_action:
            available_actions[:, 1] = (self.get_avail_explode_action() | self.get_avail_softkill_action() | self.get_avail_interfere_action())
            available_actions[:, 2] = self.get_avail_collide_action()
        else:
            available_actions[:, 1] = self.get_avail_explode_action() 
            available_actions[:, 2] = self.get_avail_softkill_action()
            available_actions[:, 3] = self.get_avail_interfere_action()
            available_actions[:, 4] = self.get_avail_collide_action()

        # if np.any(available_actions[:, 1]) and self._episode_count > 0:
        #     print()

        # available_actions[~self.red_alives, :] = 1

        return available_actions
    
    def get_avail_no_op_actions(self):
        """
        获取红方智能体不攻击动作的可用性
        
        返回:
        available_actions: 布尔数组，形状为 (n_reds, )，
                           表示每个智能体的不攻击动作是否可用
        """
        # 干扰中的智能体无法选择不攻击动作
        return ~self.red_interfering

    def get_avail_explode_action(self):
        """
        获取红方智能体的可用自爆动作
        
        返回:
        available_actions: 布尔数组，形状为 (n_reds, )，
                           表示每个智能体的自爆动作是否可用
        """
        # 红方智能体自爆范围内的蓝方智能体
        blue_in_red_explode_zone = self.distances_red2blue < self.can_explode_radius

        # 判断自爆动作的可用性
        available_actions = np.any(blue_in_red_explode_zone, axis=1) & self.red_explode_mode_mask
        
        # 被干扰/在干扰的智能体不能自爆
        available_actions[self.red_interfere_damage_mask | self.red_interfering] = False

        return available_actions
    
    def get_avail_softkill_action(self):
        """
        获取红方智能体可用的软杀伤动作
        
        返回：
        available_actions: 布尔数组，形状为 (n_reds, )，
                           表示每个智能体的软杀伤动作是否可用
        """
        # 红方智能体软杀伤范围内的蓝方智能体
        blue_in_red_softkill_zone = self.distances_red2blue < self.can_softkill_radius
        
        # 判断软杀伤动作的可用性
        available_actions = np.any(blue_in_red_softkill_zone, axis=1) & self.red_softkill_mode_mask & (self.red_softkill_time < self.softkill_time)
        
        # 被干扰/在干扰的智能体不能软杀伤
        available_actions[self.red_interfere_damage_mask | self.red_interfering] = False
        
        return available_actions
    
    def get_avail_interfere_action(self):
        """
        获取红方智能体可用的干扰动作
        
        返回：
        availables_actions: 布尔数组，形状为 (n_reds, )，表示每个智能体的干扰动作是否可用
        """
        # 红方智能体干扰范围附近的蓝方智能体
        blue_near_red_interfere_zone = (
            (self.distances_red2blue < self.can_interfere_radius) &
            (np.abs(self.angles_diff_red2blue) < self.can_interfere_angle / 2)
        )
        
        # 判断干扰动作的可用性 (干扰中的智能体仍然可以进行干扰动作)
        available_actions = (
            (
                np.any(blue_near_red_interfere_zone, axis=1) & 
                self.red_interfere_mode_mask & 
                (self.red_interfere_duration < self.interfere_duration)
            ) | self.red_interfering
        )
        
        return available_actions
        
    def get_avail_collide_action(self):
        """
        获取红方智能体的可用碰撞动作
        
        返回:
        available_actions: 布尔数组，形状为 (n_reds, )，
                           表示每个智能体的碰撞动作是否可用
        """
        # 判断蓝方智能体是否在红方智能体允许的撞击范围内
        blue_near_red_collide_zone = (
            (self.distances_red2blue < self.can_collide_radius) &
            (np.abs(self.angles_diff_red2blue) < self.collide_angle / 2)
        )
        
        # 将不在攻击范围内的智能体距离设置为无限大
        distances_red2blue = self.distances_red2blue.copy()
        distances_red2blue[~blue_near_red_collide_zone] = np.inf
        
        # 找个每个红方智能体最近的蓝方智能体
        nearest_blue_id = np.argmin(distances_red2blue, axis=1)
        
        # 如果红方智能体没有攻击范围内的蓝方智能体，目标设为-1
        nearest_blue_id[np.all(np.isinf(distances_red2blue), axis=1)] = -1
        
        # 获取红方智能体的目标索引
        self.red_collide_targets = nearest_blue_id
        
        # 判断可用的撞击动作
        available_actions = (self.red_collide_targets != -1)
        
        # 对于(被)干扰还未结束的智能体，不能进行碰撞
        available_actions[self.red_interfering] = False
        available_actions[self.red_interfere_damage_mask] = False
        
        # 对于还没有使用软杀伤或者干扰的智能体，不能碰撞
        no_softkill_mask = self.red_softkill_mode_mask & (self.red_softkill_time == 0)
        no_interfere_mask = self.red_interfere_mode_mask & (self.red_interfere_duration == 0)
        available_actions[no_softkill_mask | no_interfere_mask] = False
        
        return available_actions
    
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
    
    def transform_positions(self, positions):
        """
        将所有世界坐标转换为屏幕坐标。

        返回:
        transformed_positions: 屏幕坐标中的所有点，形状为 (n, 2)。
        """
        transformed_positions = ((positions - self.screen_offset) * self.direction_scale * self.scale_factor).astype(int)
        
        return transformed_positions
    
    def update_transformed_positions(self):
        """
        将所有智能体的坐标转成屏幕坐标
        """
        red_transformed_positions = self.transform_positions(self.red_positions)
        blue_transformed_positions = self.transform_positions(self.blue_positions)
        
        self.transformed_positions = np.vstack([red_transformed_positions, blue_transformed_positions])
        
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
        save_dir = os.path.join(self.directory, f'result/data/{self.scenario_name}/{self.map_name}/{self.timestamp}')
        os.makedirs(save_dir)
        
        filename = os.path.join(save_dir, f'{self.scenario_name}_data_{self.timestamp}.csv')
        append_to_csv(self.sim_data, filename)
        
        # 打印文件存储路径
        print(f"轨迹数据: {filename}")

    def render_explode_softkill_interfere(self):
        """
        1. 渲染红方和蓝方智能体的爆炸特效，包括自爆和被爆效果。
        2. 渲染红方和蓝方智能体的软杀伤特效，包括开启软杀伤和被软杀伤的效果
        
        """
        
        red_color_explode = (255, 0, 0)
        blue_color_explode = (0, 0, 255)
        
        red_color_softkill = (255, 0, 0)
        blue_color_softkill = (0, 0, 255)
        
        red_color_interfere = (255, 0, 0)
        blue_color_interfere = (0, 0, 255)
        
        # red_color_softkill = (0, 0, 0)  # 黑色
        # blue_color_softkill = (255, 255, 0)    # 黄色
        
        # red_color_interfere = (240, 128, 128)
        # blue_color_interfere = (173, 216, 230)
        
        # 处理红方智能体的攻击/被攻击效果
        for i in range(self.n_reds):
            # 渲染自爆/被爆
            self._render_explosion(i, red_color_explode, self.red_explode_mask[i], self.red_explode_damage_mask[i],
                                        self.red_explode_render, self.red_explosion_frames_remaining)

            # 渲染软杀伤/被软杀伤
            self._render_softkill(i, red_color_softkill, self.red_softkill_mask[i], self.red_softkill_damage_mask[i],
                                       self.red_softkill_render, self.red_softkill_frames_ramaining)
            
            # 渲染干扰
            self._render_interfere(red_color_interfere, self.red_interfere_mask[i], self.transformed_positions[i],
                                       self.red_directions[i])
        
        # 处理蓝方智能体的攻击/被攻击效果
        for i in range(self.n_blues):
            # 渲染自爆/被爆
            self._render_explosion(i, blue_color_explode, self.blue_explode_mask[i], self.blue_explode_damage_mask[i],
                                        self.blue_explode_render, self.blue_explosion_frames_remaining, is_blue=True) 
            
            # 渲染软杀伤/被软杀伤
            self._render_softkill(i, blue_color_softkill, self.blue_softkill_mask[i], self.blue_softkill_damage_mask[i],
                                      self.blue_softkill_render, self.blue_softkill_frames_remaining, is_blue=True)
                
            # 渲染干扰
            self._render_interfere(blue_color_interfere, self.blue_interfere_mask[i], self.transformed_positions[i+self.n_reds],
                                       self.blue_directions[i])
        
    def _render_explosion(self, index, color, explode_mask, explode_damage_mask, explode_render, explosion_frames_remaining, is_blue=False):
        """
        处理单个智能体的爆炸效果，包括自爆和被爆效果。
        
        参数:
        index: 智能体的索引
        color: 智能体的颜色
        explode_mask: 自爆掩码
        explode_damage_mask: 被爆掩码
        explode_render: 自爆渲染标志
        explosion_frames_remaining: 爆炸剩余帧数
        is_blue: 是否为蓝方，默认为False。
        """
        
        offset = self.n_reds if is_blue else 0
        
        # 自爆处理
        if explode_mask:
            # 设置自爆渲染标志
            explode_render[index] = True
            # 初始化自爆需要渲染的帧数
            explosion_frames_remaining[index] = self.explode_render_frames
            
        # 被爆处理
        if explode_damage_mask:
            # 初始化被爆需要渲染的帧数
            explosion_frames_remaining[index] = self.explode_render_frames
            
        # 渲染爆炸效果
        if explosion_frames_remaining[index] > 0:
            # 渲染自爆范围
            if explode_render[index]:
                pygame.draw.circle(self.screen, color, self.transformed_positions[index+offset], radius=self.screen_explode_radius, width=2)
                
            # 渲染爆炸痕迹
            pygame.draw.circle(self.screen, color, self.transformed_positions[index+offset], radius=5)
            
            # 更新剩余帧数
            explosion_frames_remaining[index] -= 1
    
    def _render_softkill(self, index, color, softkill_mask, softkill_damage_mask, softkill_render, softkill_frames_ramining, is_blue=False):
        """
        处理单个智能体的软杀伤效果，包括开启软杀伤和被软杀伤
        
        参数：
        index: 智能体的索引
        color: 智能体的颜色
        softkill_mask: 开启软杀伤的掩码
        softkill_damage_mask: 被软杀伤的掩码
        softkill_rendr: 开启软杀伤的渲染标志
        softkill_frames_ramining: 软杀伤剩余帧数
        is_blue: 是否为蓝方，默认为False
        """
        offset = self.n_reds if is_blue else 0
        
        # 开启软杀伤处理
        if softkill_mask:
            # 设置开启软杀伤标志
            softkill_render[index] = True
            # 初始化软杀伤需要渲染的帧数
            softkill_frames_ramining[index] = self.softkill_render_frames
            
        # 被软杀伤处理
        if softkill_damage_mask:
            # 初始化被软杀伤需要渲染的帧数
            softkill_frames_ramining[index] = self.softkill_render_frames
            
        # 渲染软杀伤效果
        if softkill_frames_ramining[index] > 0:
            # 渲染软杀伤范围
            if softkill_render[index]:
                pygame.draw.circle(self.screen, color, self.softkill_positions[index+offset], radius=self.screen_softkill_radius, width=2)
            
            # 渲染软杀伤痕迹
            pygame.draw.circle(self.screen, color, self.softkill_positions[index+offset], radius=5)
            
            # 更新剩余帧数
            softkill_frames_ramining[index] -= 1
    
    def _render_interfere(self, color, interfere_mask, position, direction):
        """
        处理单个智能体的干扰效果，主要包括开启干扰的效果
        
        参数：
        color: 智能体的颜色
        interfere_mask: 开启干扰的掩码
        position: 屏幕位置 
        """
        # 渲染开启干扰的效果
        if interfere_mask:
            # 计算干扰区域的起始和结束角度
            start_angle = direction - self.interfere_angle / 2
            end_angle = direction + self.interfere_angle / 2
            # 渲染软杀伤的扇形区域
            draw_sector(self.screen, position, self.screen_interfere_radius, start_angle, end_angle, color)
    
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

    def render_threat(self):
        """
        渲染红方和蓝方智能体在威胁区域内被毁伤的特效
        """
        mask = np.hstack((self.red_threat_damage_mask, self.blue_threat_damage_mask))
        self.threat_frames_ramaining[mask] = self.threat_render_frames
        
        for i in range(self.n_agents):
            if self.threat_frames_ramaining[i] > 0:
                color = (255, 0, 0) if i < self.n_reds else (0, 0, 255)
                pygame.draw.circle(self.screen, color, self.transformed_positions[i], radius=5)
                self.threat_frames_ramaining[i] -= 1
    
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
        
        # 初始化渲染参数
        self._reset_render_params()

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
        scale_factor = 0.15
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
        self.update_transformed_positions()
        directions = np.hstack([self.red_directions, self.blue_directions])
        alives = np.hstack([self.red_alives, self.blue_alives])
        
        angles = -np.degrees(directions)
        
        for i in range(self.n_agents):
            if alives[i]:
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
        self.render_explode_softkill_interfere()
        self.render_collide()
        self.render_threat()
        
    def _save_frame(self):
        """
        保存当前帧为图像文件。
        """
        frame_path = os.path.join(self.frame_dir, f"frame_{self._total_steps:06d}.png")
        pygame.image.save(self.screen, frame_path)
        
        if self.debug:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'[{current_time}] save {frame_path}')
        
    def red_random_policy(self, available_actions):
        """
        基于可用动作生成红方智能体的随机策略。
        
        参数:
        available_actions: 布尔数组，形状为 (n_reds, total_action_num)，表示每个智能体的各个动作是否可用。

        返回:
        actions: 整数数组，形状为 (n_reds, 3)，表示每个智能体的加速、航向和攻击动作。
        """
        available_actions = np.array(available_actions)
        
        acc_avail_actions = available_actions[:, :self.acc_action_num]
        heading_avail_actions = available_actions[:, self.acc_action_num:self.acc_action_num + self.heading_action_num]
        attack_avail_actions = available_actions[:, self.acc_action_num + self.heading_action_num:]

        # 随机选择加速动作
        acc_actions = np.array([np.random.choice(np.where(acc)[0]) for acc in acc_avail_actions])

        # 随机选择航向动作
        heading_actions = np.array([np.random.choice(np.where(heading)[0]) for heading in heading_avail_actions])

        # 随机选择攻击动作
        attack_actions = np.array([np.random.choice(np.where(attack)[0]) for attack in attack_avail_actions])

        # 将动作组合成完整的行动列表
        actions = np.column_stack((acc_actions, heading_actions, attack_actions))

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
