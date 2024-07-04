# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-20
# @Description: Implementation of base Environment

import os
import sys
import pygame
import imageio
import numpy as np

sys.path.append("/home/ubuntu/sunfeng/MARL/on-policy/")
os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

from scipy.spatial import distance
from onpolicy.envs.swarm_Confrontation.sce_maps import get_map_params
from onpolicy.utils.multi_discrete import MultiDiscrete
from onpolicy.envs.starcraft2.multiagentenv import MultiAgentEnv

class BaseEnv(MultiAgentEnv):
    def __init__(self, args):

        # Set the parameters of the map
        self.map_name = args.map_name
        map_params = get_map_params(self.map_name)
        self.n_reds = map_params["n_reds"]
        self.n_blues = map_params["n_blues"]
        self.size_x = map_params["size_x"]
        self.size_y = map_params["size_y"]

        self.episode_limit = args.episode_length

        # self.defender = map_params["defender"]
        if args.scenario_name == 'defense':
            self.defender = 'red'
        elif args.scenario_name == 'scout':
            self.defender = 'blue'
        else:
            raise ValueError("Unknown Scenario Name")

        # Calculate the number of agents
        self.n_agents = self.n_reds + self.n_blues

        # Set the max observed num
        self.max_observed_allies = 5
        self.max_observed_enemies = 5

        # Set the max velocity
        self.red_max_vel = 40.0 if self.defender == 'red' else 45.0
        self.blue_max_vel = 40.0 if self.defender == 'blue' else 45.0

        # Set the min velocity
        self.red_min_vel = 15.0 if self.defender == 'red' else 20.0
        self.blue_min_vel = 15.0 if self.defender == 'blue' else 20.0

        # Set the max 角速度
        # Set the max angular velocity
        self.max_angular_vel = 17.0 * np.pi / 180

        # Set the simulation time (seconds)
        self.dt_time = 1.0

        # Initialize episode and time step
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0

        # Circular detection radius (meters)
        self.detection_radius = 500.0
        # Radius of sector attacking zone (meters)
        self.attack_radius = 150.0
        # Angle of sector attacking zone (degrees)
        self.attack_angle = 34.0 * np.pi / 180

        self.can_explode_radius = 100.0
        # explode radius (meters)
        self.explode_radius = 60.0
        # collide distance (meters)
        self.collide_distance = 30.0
        # soft kill distance (meters)
        self.soft_kill_distance = 10.0
        # soft kill angle (degrees)
        self.soft_kill_angle = 20.0 * np.pi / 180
        # soft kill time (seconds)
        self.soft_kill_time = 0.0
        self.soft_kill_max_time = 3.0

        # parameters of observation
        self.obs_own_feats_size = 4   # x, y, v, heading
        self.obs_ally_feats_size = 4  # delta_x, delta_y, distance, heading
        self.obs_enemy_feats_size = 5 # delta_x, delta_y, distance, heading, angle
        self.obs_size = self.obs_own_feats_size + self.obs_ally_feats_size + self.obs_enemy_feats_size

        # parameters of state
        self.red_state_size = 4      # x, y, v, heading
        self.blue_state_size = 4     # x, y, v, heading

        # out of bound
        self.max_out_of_bounds_time = 10

        # action space
        self.acc_action_num = 5
        self.heading_action_num = 5
        self.attack_action_num = 4

        self.acc_action_mid_id = self.acc_action_num // 2
        self.heading_action_mid_id = self.heading_action_num // 2

        self.acc_action_max = 5.0
        self.heading_action_max = 1.0

        self.acc_actions = np.linspace(-self.acc_action_max, self.acc_action_max, self.acc_action_num)
        self.heading_actions = np.linspace(-self.heading_action_max, self.heading_action_max, self.heading_action_num)
        self.attack_actions = np.arange(0, self.attack_action_num) # ["no-op", "explode", "collide", "soft_kill"]

        # 定义动作空间 （多离线动作空间）
        self.action_space = [MultiDiscrete([[0, self.acc_action_num-1],
                                            [0, self.heading_action_num-1],
                                            [0, self.attack_action_num-1]])] * self.n_reds
        
        # 定义观测空间
        self.observation_space = [self.get_obs_size()] * self.n_reds

        # 定义状态空间
        self.share_observation_space = [self.get_state_size()] * self.n_reds

        # 初始化获胜次数
        self.battles_game = 0
        self.battles_won = 0
        self.timeouts = 0

        # Screen
        self.screen = None
        self.scale_factor = 0.5
        self.screen_width = self.size_x * self.scale_factor
        self.screen_height = self.size_y * self.scale_factor
        self.red_plane_img = None
        self.blue_plane_img = None
        self.red_img_cache = {}
        self.blue_img_cache = {}

        # 缓冲边界，当智能体飞出缓冲边界时，需要通过限制 heading action，使其飞回界内
        self.cache_fator = 0.92
        self.half_size_x = (self.size_x * self.cache_fator) / 2
        self.half_size_y = (self.size_y * self.cache_fator) / 2

        self.cache_bounds = np.array([
            [[ 1,  1], [-1,  1]],   # 上边界
            [[-1,  1], [-1, -1]],   # 左边界
            [[-1, -1], [ 1, -1]],   # 下边界
            [[ 1, -1], [ 1,  1]]    # 右边界
        ]) * np.array([self.half_size_x, self.half_size_y])

        # 边界向量、长度以及单位向量
        self.bounds_vec = self.cache_bounds[:, 1, :] - self.cache_bounds[:, 0, :]
        self.bounds_len = np.linalg.norm(self.bounds_vec, axis=1)
        self.bounds_unitvec = self.bounds_vec / self.bounds_len[:, np.newaxis]

    def reset(self):
        """
        Reset the environment.
        """
        # Reset time step
        self._episode_steps = 0

        # Reset num of alive agents
        self.n_agents = self.n_reds + self.n_blues

        # Reset positions, directions and velocities
        self.positions, self.directions, self.velocities = self.init_positions()

        # Reset alive array
        self.alives = np.ones(self.n_agents, dtype=bool)

        self.split_state()

        # Reset the transform positions matrix
        self.transformed_positions = np.zeros((self.n_agents, 2), dtype=int)

        # Reset out-of-bounds time
        self.out_of_bounds_time = np.zeros(self.n_agents)

        # Reset observed list and targets
        self.observed_allies = -np.ones((self.n_reds, self.max_observed_allies), dtype=int)
        self.observed_enemies = -np.ones((self.n_reds, self.max_observed_enemies), dtype=int)

        self.distance_observed_allies = np.zeros((self.n_reds, self.max_observed_allies))
        self.distance_observed_enemies = np.zeros((self.n_reds, self.max_observed_enemies))

        # 碰撞相关
        self.distances_red2blue = None
        self.angles_red2blue = None
        self.red_targets = None

        self.angles_diff_red2blue = None

        # 每个时间步，红方出界的智能体数量
        self.red_out_of_bounds_num = 0
        self.blue_out_of_bounds_num = 0

        self.red_out_of_bounds_total = 0
        self.blue_out_of_bounds_total = 0

        self.win_counted = False
        self.defeat_counted = False

        # 奖励相关的参数
        self.explode_red_num = 0 
        self.explode_blue_num = 0
        self.invalid_explode_red_num = 0
        self.collide_red_num = 0
        self.collide_blue_num = 0
        self.outofbound_red_num = 0
        self.outofbound_blue_num = 0
        
        self.explode_red_total = 0 
        self.explode_blue_total = 0
        self.invalid_explode_red_total = 0
        self.collide_red_total = 0
        self.collide_blue_total = 0
        self.outofbound_red_total = 0
        self.outofbound_blue_total = 0

        # 攻击相关的参数
        self.explode_flag = np.zeros(self.n_reds, dtype=bool)
        self.be_exploded_flag = np.zeros(self.n_reds, dtype=bool)
        self.collide_flag = np.zeros(self.n_reds, dtype=bool)
        self.be_collided_flag = np.zeros(self.n_reds, dtype=bool)
        
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def init_positions(self):
        # Random positions and directions
        positions = (np.random.rand(self.n_agents, 2) - 0.5) * np.array([self.size_x, self.size_y])
        directions = (np.random.rand(self.n_agents) - 0.5) * 2 * np.pi
        velocities = np.hstack([np.ones(self.n_reds) * self.red_max_vel, np.ones(self.n_blues) * self.blue_max_vel])

        return positions, directions, velocities
    
    def check_boundaries(self):
        # check if agents are out of boundaries
        half_size_x = self.size_x / 2
        half_size_y = self.size_y / 2

        out_of_bounds_x = (self.positions[:, 0] < -half_size_x) | (self.positions[:, 0] > half_size_x)
        out_of_bounds_y = (self.positions[:, 1] < -half_size_y) | (self.positions[:, 1] > half_size_y)
        
        out_of_bounds = out_of_bounds_x | out_of_bounds_y
        
        # Update out-of-bounds time
        self.out_of_bounds_time[out_of_bounds] += 1
        self.out_of_bounds_time[~out_of_bounds] = 0

        # Check for agents that are out of bounds for 10 time steps
        dead_or_not = self.out_of_bounds_time >= self.max_out_of_bounds_time 

        self.outofbound_red_num = np.sum(dead_or_not[:self.n_reds] & self.red_alives)
        self.outofbound_blue_num = np.sum(dead_or_not[self.n_reds:] & self.blue_alives)

        self.outofbound_red_total += self.outofbound_red_num
        self.outofbound_blue_total += self.outofbound_blue_num

        self.alives[dead_or_not] = False

    def explode(self, i):
        pass

    def collide(self, i):
        pass

    def soft_kill(self, i):
        pass
    
    def split_state(self):
        self.red_positions = self.positions[:self.n_reds, :]
        self.red_directions = self.directions[:self.n_reds]
        self.red_velocities = self.velocities[:self.n_reds]
        self.red_alives = self.alives[:self.n_reds]

        self.blue_positions = self.positions[self.n_reds:, :]
        self.blue_directions = self.directions[self.n_reds:]
        self.blue_velocities = self.velocities[self.n_reds:]
        self.blue_alives = self.alives[self.n_reds:]

    def merge_state(self):
        self.positions = np.vstack([self.red_positions, self.blue_positions])
        self.directions = np.hstack([self.red_directions, self.blue_directions])
        self.velocities = np.hstack([self.red_velocities, self.blue_velocities])
        self.alives = np.hstack([self.red_alives, self.blue_alives])

    def red_explode(self, explode_mask):
        # 更新 explode_mask， 排除已经死掉的智能体
        valid_explode_mask = explode_mask & self.red_alives

        # 计算每个红方智能体与每个蓝方智能体之间的距离
        distances_red2blue = distance.cdist(self.red_positions, self.blue_positions, 'euclidean')
        
        # 红方智能体自爆范围内的蓝方智能体
        blue_in_explode_zone = distances_red2blue < self.explode_radius

        # 统计无效自爆的智能体数量
        valid_blue_mask = blue_in_explode_zone & self.blue_alives
        self.invalid_explode_red_num = np.sum(np.sum(valid_blue_mask[valid_explode_mask], axis=1) == 0)
        self.invalid_explode_red_total += self.invalid_explode_red_num

        # 触发自爆的红方智能体将被标记为死亡
        self.red_alives[valid_explode_mask] = False
        self.explode_flag[valid_explode_mask] = (np.sum(valid_blue_mask[valid_explode_mask], axis=1) != 0)

        # 将自爆范围内的蓝方智能体标记为死亡，并统计有效毁伤的蓝方智能体数量
        self.blue_explode_mask = np.any(blue_in_explode_zone[valid_explode_mask], axis=0)
        self.explode_blue_num = np.sum(self.blue_explode_mask & self.blue_alives)
        self.explode_blue_total += self.explode_blue_num
        self.blue_alives[self.blue_explode_mask] = False

    def red_collide(self, collide_mask, pt):
        """
        红方智能体与其目标的碰撞
        """
        # # 计算红方智能体到蓝方智能体的方向向量
        # delta_red2blue = self.blue_positions[np.newaxis, :, :] - self.red_positions[:, np.newaxis, :]   # nred x nblue x 2

        # # 计算红方智能体到蓝方智能体的角度
        # angles_red2blue = np.arctan2(delta_red2blue[:, :, 1], delta_red2blue[:, :, 0])                  # nred x nblue

        # # 计算红方智能体当前方向与到蓝方智能体的方向的角度差
        # angles_diff_red2blue = angles_red2blue - self.red_directions[:, np.newaxis]                     # nred x nblue
        # angles_diff_red2blue = (angles_diff_red2blue + np.pi) % (2 * np.pi) - np.pi

        # # 计算红方智能体到蓝方智能体的距离
        # distances_red2blue = distance.cdist(self.red_positions, self.blue_positions, 'euclidean')

        # # 创建有效性掩码，只考虑存活的红方和蓝方智能体之间的距离
        # valid_mask = self.red_alives[:, np.newaxis] & self.blue_alives[np.newaxis, :]

        # # 将无效的距离设置为无限大
        # distances_red2blue = np.where(valid_mask, distances_red2blue, np.inf)

        # # 红方智能体攻击范围内的蓝方智能体
        # blue_in_attack_zone = (distances_red2blue < self.attack_radius) & (
        #     np.abs(angles_diff_red2blue) < self.attack_angle / 2
        # )

        # # 将不在攻击范围内的距离设置为无限大
        # distances_red2blue[~blue_in_attack_zone] = np.inf

        # # 找到每个红方智能体最近的蓝方智能体
        # nearest_blue_id = np.argmin(distances_red2blue, axis=1)

        # # 如果红方智能体没有攻击范围内的蓝方智能体，目标设为-1
        # nearest_blue_id[np.all(np.isinf(distances_red2blue), axis=1)] = -1

        # 更新红方智能体的目标
        red_targets = self.red_targets

        # 更新 collide_mask，排除没有 target 的智能体
        valid_collide_mask = collide_mask & (red_targets != -1) & self.red_alives

        # 获取有效的 target_id
        target_ids = red_targets[valid_collide_mask]
        agent_ids = np.where(valid_collide_mask)[0]

        # 获取红方智能体和其目标之间的距离
        valid_distances = self.distances_red2blue[valid_collide_mask, target_ids]

        # 判断撞击成功的情况
        collide_success_mask = valid_distances < (self.collide_distance + self.red_velocities[valid_collide_mask] * self.dt_time)

        # 获取撞击成功的 agent_id 和 target_id
        success_agent_ids = agent_ids[collide_success_mask]
        success_target_ids = target_ids[collide_success_mask]

        self.collide_blue_num = success_agent_ids.size
        self.collide_blue_total += self.collide_blue_num

        # 更新红方智能体和目标蓝方智能体的存活状态
        self.red_alives[success_agent_ids] = False
        self.blue_alives[success_target_ids] = False

        self.collide_flag = np.zeros(self.n_reds, dtype=bool)
        self.collide_flag[success_agent_ids] = True

        # 更新红方智能体的方向
        self.red_directions[valid_collide_mask] = self.angles_red2blue[valid_collide_mask, target_ids]
        pt[valid_collide_mask] = 0

        return pt

    def step(self, actions):
        # Get actions
        at = self.acc_actions[actions[:, 0]]
        pt = self.heading_actions[actions[:, 1]]
        attack_t = actions[:, 2]

        # Perform attack actions
        for i in range(self.n_agents):
            if not self.alives[i]:
                continue
            
            if attack_t[i] == 0:    # no attack
                continue
            elif attack_t[i] == 1:
                # self.explode(i)   # explode
                continue
            elif attack_t[i] == 2:  # collide
                flag = self.collide(i)
                if not flag:
                    pt[i] = 0
            elif attack_t[i] == 3: # soft kill
                self.soft_kill(i)
            else:
                raise ValueError

        # Perform move actions
        self.directions += pt * self.max_angular_vel
        self.directions = (self.directions + np.pi) % (2 * np.pi) - np.pi
        self.velocities += at * self.dt_time
        self.velocities[:self.n_reds] = np.clip(self.velocities[:self.n_reds], self.red_min_vel, self.red_max_vel)
        self.velocities[self.n_reds:] = np.clip(self.velocities[self.n_reds:], self.blue_min_vel, self.blue_max_vel)
        self.positions += np.column_stack((self.velocities * np.cos(self.directions), 
                                           self.velocities * np.sin(self.directions))) * self.dt_time
        
        self.split_state()
        self.check_boundaries()

        # Update step counter
        self._total_steps += 1
        self._episode_steps += 1     
    
    def update_observed_entities(self, positions, alives, max_num):
        # 计算红方智能体与实体之间的距离
        distance_red2entity = distance.cdist(self.red_positions, positions, 'euclidean')

        # 创建有效性掩码，只考虑存活的智能体之间的距离
        valid_mask = self.red_alives[:, np.newaxis] & alives[np.newaxis, :]

        # 将无效的距离设置为无限大
        distance_red2entity = np.where(valid_mask, distance_red2entity, np.inf)

        # 通信范围内的实体
        in_radius_entities = distance_red2entity < self.detection_radius

        # 将不在通信范围内的实体距离设置为无限大
        distance_red2entity[~in_radius_entities] = np.inf

        # 找到通信范围内max_num个最近的实体，如果不够则用-1补全
        # 使用 argsort 进行排序，并选取前max_num个最近的实体
        sorted_indices = np.argsort(distance_red2entity, axis=1)

        # 获取前 max_num 个最近的实体索引
        nearest_id = sorted_indices[:, :max_num]

        # 获取前 max_num 个最近的实体距离
        nearest_dist = distance_red2entity[np.arange(self.n_reds)[:, np.newaxis], nearest_id]

        # 创一个掩码来标记距离无限大的位置
        inf_mask = np.isinf(nearest_dist)

        # 将距离为无限大的位置索引替换为 -1
        nearest_id[inf_mask] = -1

        return nearest_id, nearest_dist

    def update_angles_diff(self):
        # 计算红方智能体到蓝方智能体的方向向量
        delta_red2blue = self.blue_positions[np.newaxis, :, :] - self.red_positions[:, np.newaxis, :]   # nred x nblue x 2

        # 计算红方智能体到蓝方智能体的角度
        angles_red2blue = np.arctan2(delta_red2blue[:, :, 1], delta_red2blue[:, :, 0])                  # nred x nblue

        # 计算红方智能体当前方向与到蓝方智能体的方向的角度差
        angles_diff_red2blue = angles_red2blue - self.red_directions[:, np.newaxis]                     # nred x nblue
        angles_diff_red2blue = (angles_diff_red2blue + np.pi) % (2 * np.pi) - np.pi

        return angles_diff_red2blue
    
    def get_obs_size(self):
        own_feats = self.obs_own_feats_size
        
        n_allies, n_ally_feats = self.max_observed_allies, self.obs_ally_feats_size
        n_enemies, n_enemy_feats = self.max_observed_enemies, self.obs_enemy_feats_size

        ally_feats = n_allies * n_ally_feats
        enemy_feats = n_enemies * n_enemy_feats

        all_feats = own_feats + ally_feats + enemy_feats

        return [all_feats, [1, own_feats], [n_allies, n_ally_feats], [n_enemies, n_enemy_feats]]
    
    def get_obs(self):
        # Update observed allies and enemies for all agents
        self.observed_allies, self.distance_observed_allies = self.update_observed_entities(
            self.red_positions, self.red_alives, self.max_observed_allies)
        self.observed_enemies, self.distance_observed_enemies = self.update_observed_entities(
            self.blue_positions, self.blue_alives, self.max_observed_enemies)
        
        # Update angles differences between red agents and blue agents
        self.angles_diff_red2blue = self.update_angles_diff()

        # Initialize feature arrays
        own_feats = np.zeros((self.n_reds, self.obs_own_feats_size), dtype=np.float32)
        ally_feats = np.zeros((self.n_reds, self.max_observed_allies, self.obs_ally_feats_size), dtype=np.float32)
        enemy_feats = np.zeros((self.n_reds, self.max_observed_enemies, self.obs_enemy_feats_size), dtype=np.float32)

        # Process only alive agents
        alive_mask = self.red_alives

        # Own features
        # own_feats[alive_mask, 0:2] = self.red_positions[alive_mask] / np.array([self.size_x / 2, self.size_y / 2])
        own_feats[alive_mask, 2] = (self.red_velocities[alive_mask] - self.red_min_vel) / (self.red_max_vel - self.red_min_vel)
        own_feats[alive_mask, 3] = self.red_directions[alive_mask] / np.pi

        # Ally features
        valid_allies_mask = self.observed_allies != -1
        ally_ids = self.observed_allies[valid_allies_mask]
        agent_ids, ally_indices = np.where(valid_allies_mask)
        
        ally_positions = self.red_positions[ally_ids]
        ally_feats[agent_ids, ally_indices, 0:2] = (ally_positions - self.red_positions[agent_ids]) / self.detection_radius
        ally_feats[agent_ids, ally_indices, 2] = self.distance_observed_allies[valid_allies_mask] / self.detection_radius
        ally_feats[agent_ids, ally_indices, 3] = self.red_directions[ally_ids] / np.pi

        # Enemy features
        valid_enemies_mask = self.observed_enemies != -1
        enemy_ids = self.observed_enemies[valid_enemies_mask]
        agent_ids, enemy_indices = np.where(valid_enemies_mask)
        
        enemy_positions = self.blue_positions[enemy_ids]
        enemy_feats[agent_ids, enemy_indices, 0:2] = (enemy_positions - self.red_positions[agent_ids]) / self.detection_radius
        enemy_feats[agent_ids, enemy_indices, 2] = self.distance_observed_enemies[valid_enemies_mask] / self.detection_radius
        enemy_feats[agent_ids, enemy_indices, 3] = self.blue_directions[enemy_ids] / np.pi
        enemy_feats[agent_ids, enemy_indices, 4] = self.angles_diff_red2blue[agent_ids, enemy_ids] / (self.attack_angle / 2)

        # Combine all features into a single observation array
        agents_obs = np.concatenate(
            (
                own_feats,
                ally_feats.reshape(self.n_reds, -1),
                enemy_feats.reshape(self.n_reds, -1)
            ),
            axis=1
        )

        obs = [agents_obs[i, :] for i in range(self.n_reds)]

        return obs

    def get_state_size(self):
        nf_al = self.red_state_size
        nf_en = self.blue_state_size

        size = self.n_reds * nf_al + self.n_blues * nf_en

        return [size, [self.n_reds, nf_al], [self.n_blues, nf_en]]
    
    def get_state(self):
        # Initialize the red and blue state arrays
        red_state = np.zeros((self.n_reds, self.red_state_size), dtype=np.float32)
        blue_state = np.zeros((self.n_blues, self.blue_state_size), dtype=np.float32)

        # Calculate the normalized positions for reds and blues
        normalized_red_positions = self.red_positions / np.array([self.size_x / 2, self.size_y / 2])
        normalized_blue_positions = self.blue_positions / np.array([self.size_x / 2, self.size_y / 2])

        # Calculate the normalized velocities for reds and blues
        normalized_red_velocities = (self.red_velocities - self.red_min_vel) / (self.red_max_vel - self.red_min_vel)
        normalized_blue_velocities = (self.blue_velocities - self.blue_min_vel) / (self.blue_max_vel - self.blue_min_vel)

        # Calculate the normalized directions for reds and blues
        normalized_red_directions = self.red_directions / np.pi
        normalized_blue_directions = self.blue_directions / np.pi

        # Populate the red_state array
        alive_reds = self.red_alives.astype(bool)
        red_state[alive_reds, 0:2] = normalized_red_positions[alive_reds]
        red_state[alive_reds, 2] = normalized_red_velocities[alive_reds]
        red_state[alive_reds, 3] = normalized_red_directions[alive_reds]

        # Populate the blue_state array
        alive_blues = self.blue_alives.astype(bool)
        blue_state[alive_blues, 0:2] = normalized_blue_positions[alive_blues]
        blue_state[alive_blues, 2] = normalized_blue_velocities[alive_blues]
        blue_state[alive_blues, 3] = normalized_blue_directions[alive_blues]

        # Flatten and concatenate the state arrays
        state = np.concatenate((red_state.flatten(), blue_state.flatten()))

        return state
    
    def get_avail_actions(self):
        # 获取 available_actions

        # 获取加速类动作的 available_actions
        available_acc_actions = self.get_avail_acc_actions()

        # 获取航向类动作的 available_actions
        available_heading_actions = self.get_avail_heading_actions()

        # 获取攻击类动作的 available_actions
        available_attack_actions = self.get_avail_attack_actions()

        # 将三类动作拼起来
        agent_avail_actions = np.hstack((available_acc_actions, available_heading_actions, available_attack_actions))

        available_actions = [agent_avail_actions[i, :].tolist() for i in range(self.n_reds)]

        return available_actions
        

    def get_avail_acc_actions(self):
        # 初始化有效动作数组
        available_actions = np.ones((self.n_reds, self.acc_action_num), dtype=bool)

        # 判断速度是否出界
        max_vel_indices = self.red_velocities >= self.red_max_vel
        min_vel_indices = self.red_velocities <= self.red_min_vel

        # 限制速度已经出界的智能体的加速动作
        available_actions[max_vel_indices, self.acc_action_mid_id + 1:] = 0
        available_actions[min_vel_indices, :self.acc_action_mid_id] = 0

        return available_actions
    
    def get_avail_heading_actions(self):
        # 初始化有效动作数组
        available_actions = np.ones((self.n_reds, self.heading_action_num), dtype=bool)

        # 判断智能体是否出缓冲边界
        out_of_bounds_x = (self.red_positions[:, 0] < -self.half_size_x) | (self.red_positions[:, 0] > self.half_size_x)
        out_of_bounds_y = (self.red_positions[:, 1] < -self.half_size_y) | (self.red_positions[:, 1] > self.half_size_y)
        out_of_bounds = out_of_bounds_x | out_of_bounds_y

        # 找到出界的智能体索引
        out_of_bounds_indices = np.where(out_of_bounds)[0]

        if out_of_bounds_indices.size == 0:
            return available_actions

        # 计算智能体当前位置到每个边界起点的向量
        pos_vec = self.red_positions[out_of_bounds_indices, np.newaxis, :] - self.cache_bounds[:, 0, :]  # (n, 4, 2)

        # 计算点向量的单位向量
        pos_unitvec = pos_vec / self.bounds_len[:, np.newaxis]

        # 计算每个智能体位置在每条线段上的投影比例
        t = np.einsum('nij,ij->ni', pos_unitvec, self.bounds_unitvec)   # (n, 4)

        # 将 t 限制在 [0, 1] 范围内，确保投影点在线段上
        t = np.clip(t, 0.0, 1.0)

        # 计算线段上距离智能体最近的点的坐标 (投影点)
        nearest = self.cache_bounds[:, 0, :] + t[:, :, np.newaxis] * self.bounds_vec[np.newaxis, :, :]  # (n, 4, 2)

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

        # 如果角度差大于angles_diff_threshold, 则只用-1, -2号动作
        mask_pos = angles_diff >= angles_diff_threshold
        available_actions[out_of_bounds_indices[mask_pos], :self.heading_action_mid_id+1] = 0

        # 如果角度差小于-angles_diff_threshold，则只用0, 1号动作
        mask_neg = angles_diff <= -angles_diff_threshold
        available_actions[out_of_bounds_indices[mask_neg], self.heading_action_mid_id:] = 0


        return available_actions
    
    def get_avail_attack_actions(self):
        # 初始化有效动作数组
        available_actions = np.ones((self.n_reds, self.attack_action_num), dtype=bool)

        # 判断自爆动作是否满足条件
        explode_mask = self.get_avail_explode_action()
        available_actions[:, 1] = explode_mask

        # 判断撞击动作是否满足条件
        collide_mask = self.get_avail_collide_action()
        available_actions[:, 2] = collide_mask

        # 软杀伤动作默认不满足条件
        available_actions[:, 3] = 0

        return available_actions

    def get_avail_collide_action(self):
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

        available_actions = (red_targets != -1)

        self.distances_red2blue = distances_red2blue
        self.angles_red2blue = angles_red2blue
        self.red_targets = red_targets

        return available_actions
    

    def get_avail_explode_action(self):
        # 计算每个红方智能体与每个蓝方智能体之间的距离
        distances_red2blue = distance.cdist(self.red_positions, self.blue_positions, 'euclidean') # (nr, nb, 2)
        
        # 红方智能体自爆范围内的蓝方智能体
        blue_in_red_explode_zone = distances_red2blue < self.can_explode_radius # (nr, nb)

        # 在红方自爆范围内的存活的蓝方智能体
        blue_in_red_explode_zone = blue_in_red_explode_zone & self.blue_alives[np.newaxis, :]

        available_actions = np.any(blue_in_red_explode_zone, axis=1)
        # available_actions = np.ones(self.n_reds, dtype=)

        return available_actions


    def scripted_policy(self):
        acc_action = np.random.randint(0, self.acc_action_num, size=self.n_agents)
        heading_action = np.random.randint(0, self.heading_action_num, size=self.n_agents)
        # attack_action = np.random.randint(0, self.attack_action_num, size=self.n_agents)
        attack_action = np.random.choice(np.arange(self.attack_action_num), size=self.n_agents, p=np.array([0.9, 0.05, 0.05, 0]))

        actions = np.column_stack((acc_action, heading_action, attack_action))

        return actions 

    def scripted_policy_blue(self):
        acc_action = np.random.randint(0, self.acc_action_num, size=self.n_blues)
        heading_action = np.random.randint(0, self.heading_action_num, size=self.n_blues)
        # attack_action = np.random.randint(0, self.attack_action_num, size=self.n_blues)
        attack_action = np.random.choice(np.arange(self.attack_action_num), size=self.n_blues, p=np.array([0.9, 0.05, 0.05, 0]))

        actions = np.column_stack((acc_action, heading_action, attack_action))

        return actions 
    
    def scripted_policy_red(self):
        acc_action = np.random.randint(0, self.acc_action_num, size=self.n_reds)
        heading_action = np.random.randint(0, self.heading_action_num, size=self.n_reds)
        # attack_action = np.random.randint(0, self.attack_action_num, size=self.n_reds)
        attack_action = np.random.choice(np.arange(self.attack_action_num), size=self.n_reds, p=np.array([0.9, 0.02, 0.08, 0]))

        actions = np.column_stack((acc_action, heading_action, attack_action))

        return actions
    
    def transform_position(self, position):
        new_center = np.array([-self.size_x / 2, self.size_y/2])
        new_dir = np.array([1, -1])

        transformed_position = ((position - new_center) * new_dir * self.scale_factor).astype(int)
        
        return transformed_position
    
    def transform_positions(self):
        # 将世界坐标转换为屏幕坐标
        new_center = np.array([-self.size_x / 2, self.size_y/2])
        new_dir = np.array([1, -1])

        self.transformed_positions = ((self.positions - new_center) * new_dir * self.scale_factor).astype(int)
        
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def get_rotated_image(self, image, angle, cache, i):
        angle = round(angle)
        if angle not in cache:
            rotated_img = pygame.transform.rotate(image, -angle)
            cache[angle] = rotated_img
        
        return cache[angle]
    
    def render_circle(self, center, radius, color=(255, 0, 0), width=2):
        center_screen = self.transform_position(center)
        radius_screen = int(radius / self.size_x * self.screen_width)
        pygame.draw.circle(self.screen, color, center_screen, radius_screen, width=width)

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


        self.screen.fill((255, 255, 255))
        self.transform_positions()
        self.angles = np.degrees(self.directions)

        for i in range(self.n_agents):
            if self.alives[i]:
                image = self.red_plane_img if i < self.n_reds else self.blue_plane_img
                cache = self.red_img_cache if i < self.n_reds else self.blue_img_cache

                # rotated_img = pygame.transform.rotate(image, -self.angles[i])
                rotated_img = self.get_rotated_image(image, self.angles[i], cache, i)
                new_rect = rotated_img.get_rect(center=self.transformed_positions[i])
                self.screen.blit(rotated_img, new_rect)

        pygame.display.flip()

        if save_frames:
            frame_path = f"./frames/frame_{frame_num:04d}.png"
            pygame.image.save(self.screen, frame_path)
            

def create_gif(frame_folder, output_path,  fps=10):
    images = []
    for file_name in sorted(os.listdir(frame_folder)):
        if file_name.endswith('.png'):
            file_path = os.path.join(frame_folder, file_name)
            images.append(imageio.imread(file_path))

    imageio.mimsave(output_path, images, fps=fps)

            
if __name__ == "__main__":

    world = BaseEnv()

    import time
    world.reset()
    num_frames = 100

    time_list = []
    
    for i in range(num_frames):
        print('-'* 30)
        start_time = time.time()
        last_time = time.time()

        world.get_obs()
        print("获取观测: {:.5f}".format(time.time() - last_time))
        last_time = time.time()

        world.get_state()
        print("获取状态: {:.5f}".format(time.time() - last_time))
        last_time = time.time()

        actions = world.scripted_policy()
        print("脚本策略: {:.5f}".format(time.time() - last_time))
        last_time = time.time()
        
        world.step(actions)
        print("环境更新: {:.5f}".format(time.time() - last_time))
        last_time = time.time()
        
        world.render(frame_num=i, save_frames=False)
        print("环境渲染: {:.5f}".format(time.time() - last_time))
        time_list.append(time.time() - start_time)
    
    time_list = np.array(time_list)

    print(time_list.mean(), time_list.std())

    world.close()

    # create_gif("frames", "output.gif", fps=100)