# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-20
# @Description: Implementation of base Environment

import os
import pygame
import imageio
import numpy as np

from scipy.spatial import distance
from smac.env.multiagentenv import MultiAgentEnv

os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

class BaseEnv(MultiAgentEnv):
    def __init__(self):
        
        # Set the parameters of the map
        self.map_name = "map_1"
        self.n_reds = 100
        self.n_blues = 100
        self.episode_limit = 400
        self.size_x = 8000
        self.size_y = 5000
        self.defender = 'red'

        # Calculate the number of agents
        self.n_agents = self.n_reds + self.n_blues

        # Set the max observed num
        self.max_observed_allies = 5
        self.max_observed_enemies = 5

        # Set the max velocity
        self.red_max_vel = 45.0 if self.defender == 'red' else 40.0
        self.blue_max_vel = 45.0 if self.defender == 'blue' else 40.0

        # Set the min velocity
        self.red_min_vel = 20.0 if self.defender == 'red' else 15.0
        self.blue_min_vel = 20.0 if self.defender == 'blue' else 15.0

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
        self.attack_radius = 300.0
        # Angle of sector attacking zone (degrees)
        self.attack_angle = 34.0 * np.pi / 180

        # explode radius (meters)
        self.explode_radius = 30.0
        # collide distance (meters)
        self.collide_distance = 3.0
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

        self.acc_action_max = 5.0
        self.heading_action_max = 1.0

        self.acc_actions = np.linspace(-self.acc_action_max, self.acc_action_max, self.acc_action_num)
        self.heading_actions = np.linspace(-self.heading_action_max, self.heading_action_max, self.heading_action_num)
        self.attack_actions = np.arange(0, self.attack_action_num) # ["no-op", "explode", "collide", "soft_kill"]

        # Screen
        self.screen = None
        self.scale_factor = 0.5
        self.screen_width = self.size_x * self.scale_factor
        self.screen_height = self.size_y * self.scale_factor
        self.red_plane_img = None
        self.blue_plane_img = None
        self.red_img_cache = {}
        self.blue_img_cache = {}


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
        
        self.red_targets = -np.ones(self.n_reds, dtype=int)
        self.blue_targets = -np.ones(self.n_blues, dtype=int)

        self.angles_diff_red2blue = None

        # 每个时间步，红方出界的智能体数量
        self.out_of_bounds_num = 0 

        obs = self.get_obs()

        return obs

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

        self.out_of_bounds_num = np.sum(dead_or_not[:self.n_reds] & self.alives)
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


    def get_obs_agent(self, agent_id):
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        ally_feats = np.zeros((self.max_observed_allies, self.obs_ally_feats_size), dtype=np.float32)
        enemy_feats = np.zeros((self.max_observed_enemies, self.obs_enemy_feats_size), dtype=np.float32)

        if self.red_alives[agent_id]:
            # Own features
            # own_feats[0:2] = self.red_positions[agent_id] / np.array([self.size_x / 2, self.size_y / 2])
            own_feats[0:2] = [0,0]
            own_feats[2] = (self.red_velocities[agent_id] - self.red_min_vel) / (self.red_max_vel - self.red_min_vel)
            own_feats[3] = self.red_directions[agent_id] / np.pi

            # Ally features
            valid_allies = self.observed_allies[agent_id]
            valid_allies = valid_allies[valid_allies != -1]

            if valid_allies.size > 0:
                ally_positions = self.red_positions[valid_allies]
                ally_feats[:valid_allies.size, 0:2] = (ally_positions - self.red_positions[agent_id]) / self.detection_radius
                ally_feats[:valid_allies.size, 2] = self.distance_observed_allies[agent_id, :valid_allies.size] / self.detection_radius
                ally_feats[:valid_allies.size, 3] = self.red_directions[valid_allies] / np.pi

            # enemy_feats
            valid_enemies = self.observed_enemies[agent_id]
            valid_enemies = valid_enemies[valid_enemies != -1]

            if len(valid_enemies) > 0:
                enemy_positions = self.blue_positions[valid_enemies]
                enemy_feats[:valid_enemies.size, 0:2] = (enemy_positions - self.red_positions[agent_id]) / self.detection_radius
                enemy_feats[:valid_enemies.size, 2] = self.distance_observed_enemies[agent_id, :valid_enemies.size] / self.detection_radius
                enemy_feats[:valid_enemies.size, 3] = self.blue_directions[valid_enemies] / np.pi
                enemy_feats[:valid_enemies.size, 4] = self.angles_diff_red2blue[agent_id, valid_enemies] / (self.attack_angle / 2)
        
        agent_obs = np.concatenate(
            (
                own_feats,
                ally_feats.flatten(),
                enemy_feats.flatten()
            )
        )

        return agent_obs
    
    def get_obs_size(self):
        own_feats = self.obs_own_feats_size
        
        n_allies, n_ally_feats = self.max_observed_allies, self.obs_ally_feats_size
        n_enemies, n_enemy_feats = self.max_observed_enemies, self.obs_enemy_feats_size

        ally_feats = n_allies * n_ally_feats
        enemy_feats = n_enemies * n_enemy_feats

        all_feats = own_feats + ally_feats + enemy_feats

        return [all_feats, [1, own_feats], [n_allies, n_ally_feats], [n_enemies, n_enemy_feats]]
        

    def get_obs_old(self):
        self.observed_allies, self.distance_observed_allies = self.update_observed_entities(
            self.red_positions, self.red_alives, self.max_observed_allies)
        self.observed_enemies, self.distance_observed_enemies = self.update_observed_entities(
            self.blue_positions, self.blue_alives, self.max_observed_enemies)
        
        self.angles_diff_red2blue = self.update_angles_diff()

        agents_obs = [self.get_obs_agent(i) for i in range(self.n_reds)]

        return agents_obs
    
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
        own_feats[alive_mask, 0:2] = self.red_positions[alive_mask] / np.array([self.size_x / 2, self.size_y / 2])
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
    
    def get_state_size(self):
        nf_al = self.red_state_size
        nf_en = self.blue_state_size

        size = self.n_reds * nf_al + self.n_blues * nf_en

        return [size, [self.n_reds, nf_al], [self.n_blues, nf_en]]
    
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
        half_size_x = self.size_x / 2
        half_size_y = self.size_y / 2

        transformed_x = ((position[0] + half_size_x) / self.size_x * self.screen_width).astype(int)
        transformed_y = ((position[1] + half_size_y) / self.size_y * self.screen_height).astype(int)
        return np.array([transformed_x, transformed_y])
    
    def transform_positions(self):
        # 将世界坐标转换为屏幕坐标
        half_size_x = self.size_x / 2
        half_size_y = self.size_y / 2

        self.transformed_positions[:, 0] = ((self.positions[:, 0] + half_size_x)
                                              / self.size_x * self.screen_width).astype(int)
        self.transformed_positions[:, 1] = ((self.positions[:, 1] + half_size_y)
                                              / self.size_y * self.screen_height).astype(int)
        
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
            frame_path = f"frames/frame_{frame_num:04d}.png"
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


    


        





        



