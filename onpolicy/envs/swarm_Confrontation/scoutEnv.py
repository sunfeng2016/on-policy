# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-17
# @Description: Implementation of Agent Class

import os
import pygame
import imageio
import numpy as np

from scipy.spatial import distance

os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

class World(object):
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

        # action space
        self.acc_action_num = 5
        self.heading_action_num = 5
        self.attack_action_num = 3

        self.acc_action_max = 5.0
        self.heading_action_max = 1.0

        self.acc_actions = np.linspace(-self.acc_action_max, self.acc_action_max, self.acc_action_num)
        self.heading_actions = np.linspace(-self.heading_action_max, self.heading_action_max, self.heading_action_num)
        self.attack_actions = np.arange(0, self.attack_action_num) # ["explode", "collide", "soft_kill"]

        # Screen
        self.screen = None
        self.screen_width = 4000
        self.screen_height = 2500
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
        self.n_red_alive = self.n_reds
        self.n_blue_alive = self.n_blues

        # Reset positions, directions and velocities
        self.positions, self.directions, self.velocities = self.init_positions()

        self.red_positions = None
        self.red_directions = None
        self.red_velocities = None

        self.blue_positions = None
        self.blue_directions = None
        self.blue_velocities = None

        # Reset the transform positions matrix
        self.transformed_positions = np.zeros((self.n_agents, 2), dtype=int)

        # Reset alive array
        self.alives = np.ones(self.n_agents, dtype=bool)
        self.red_alives = None
        self.blue_alives = None

        # Reset the distances matrix and angle matrix
        self.distances_matrix_red2blue = None
        self.distances_matrix_red2red = None
        self.directions_matrix_red2blue = None
        self.directions_matrix_blue2red = None
        self.angles_diff_matrix_red2blue = None
        self.angles_diff_matrix_blue2red = None

        # Reset observed list and targets
        self.red_observed_allies = None
        self.red_observed_enemies = None
        self.red_targets = None
        self.blue_targets = None


    def init_positions(self):
        # Random positions and directions
        positions = (np.random.rand(self.n_agents, 2) - 0.5) * np.array([self.size_x, self.size_y])
        # directions = (np.random.rand(self.n_agents) - 0.5) * 2 * np.pi
        directions = np.zeros(self.n_agents)
        directions[self.n_reds:] = np.pi
        velocities = np.ones(self.n_agents) * 15.0
        velocities[:self.n_reds] += 5.0

        return positions, directions, velocities
    
    def explode(self, i):
        if i < self.n_reds:
            in_radius_enemies_i = self.distances_matrix_red2blue[i, :] < self.explode_radius

            self.red_alives[i] = False
            self.blue_alives[in_radius_enemies_i] = False
        else:
            i -= self.n_reds
            in_radius_enemies_i = self.distances_matrix_blue2red[i, :] < self.explode_radius

            self.blue_alives[i] = False
            self.red_alives[in_radius_enemies_i] = False

    def collide(self, i):
        if i < self.n_reds:
            # assert self.red_targets[i] != -1
            if self.red_targets[i] == -1:
                return True

            if self.distances_matrix_red2blue[i, self.red_targets[i]]  <= self.collide_distance:
                self.red_alives[i] = False
                self.blue_alives[self.red_targets[i]] = False
                return True
            else:
                self.red_directions = self.directions_matrix_red2blue[i, self.red_targets[i]]
                return False
        else:
            i -= self.n_reds
            # assert self.blue_targets[i] == -1
            if self.blue_targets[i] == -1:
                return True

            if self.distances_matrix_blue2red[i, self.blue_targets[i]]  <= self.collide_distance:
                self.blue_alives[i] = False
                self.red_alives[self.blue_targets[i]] = False
                return True
            else:
                self.blue_directions = self.directions_matrix_blue2red[i, self.blue_targets[i]]
                return False

    def soft_kill(self, i):
        pass


    def step(self, actions):

        at = self.acc_actions[actions[:, 0]]
        pt = self.heading_actions[actions[:, 1]]
        # pt = np.zeros(self.n_agents)
        attack_t = actions[:, 2]

        for i in range(self.n_agents):
            if not self.alives[i]:
                continue
            if attack_t[i] == 0:
                # self.explode(i)     # explode
                continue
            elif attack_t[i] == 1:  # collide
                flag = self.collide(i)
                if not flag:
                    pt[i] = 0
            elif attack_t[i] == 2:
                self.soft_kill(i)
            else:
                raise ValueError

        # self.red_directions += pt * self.max_angular_vel
        # self.red_velocities += at * self.dt_time
        # self.red_positions += np.column_stack((self.red_velocities * np.cos(self.red_directions), 
        #                                        self.red_velocities * np.sin(self.red_directions))) * self.dt_time
 
        # self.red_positions += self.red_velocities * self.dt_time

        self.directions += pt * self.max_angular_vel
        self.velocities += at * self.dt_time
        self.velocities[:self.n_reds] = np.clip(self.velocities[:self.n_reds], self.red_min_vel, self.red_max_vel)
        self.velocities[self.n_reds:] = np.clip(self.velocities[self.n_reds:], self.blue_min_vel, self.blue_max_vel)
        self.positions += np.column_stack((self.velocities * np.cos(self.directions), 
                                           self.velocities * np.sin(self.directions))) * self.dt_time
            

    def update_matrices(self):
        self.red_positions = self.positions[:self.n_reds, :]
        self.red_directions = self.directions[:self.n_reds]
        self.red_velocities = self.velocities[:self.n_reds]
        self.red_alives = self.alives[:self.n_reds]

        self.blue_positions = self.positions[self.n_reds:, :]
        self.blue_directions = self.directions[self.n_reds:]
        self.blue_velocities = self.velocities[self.n_reds:]
        self.blue_alives = self.alives[self.n_reds:]

        # Calculate direction vectors from red agents to blue agents
        delta_red2blue = self.blue_positions[np.newaxis, :, :] - self.red_positions[:, np.newaxis, :]   # nred x nblue x 2

        # Calculate direction angles from red agents to blue agents
        directions_angles_red2blue = np.arctan2(delta_red2blue[:, :, 1], delta_red2blue[:, :, 0])       # nred x nblue
        
        # Calculate angles between the red and blue planes
        angles_diff_matrix_red2blue = directions_angles_red2blue - self.red_directions[:, np.newaxis]   # nred x nblue
        angles_diff_matrix_red2blue = (angles_diff_matrix_red2blue + np.pi) % (2 * np.pi) - np.pi

        # Calculate distances from red agents to blue agents
        distances_red2blue = distance.cdist(self.red_positions, self.blue_positions, 'euclidean')

        # Create a mask for the alive agents
        valid_mask = self.red_alives[:, np.newaxis] & self.blue_alives[np.newaxis, :]                   # nred x nblue

        # Update the directions and distances matrices
        self.distances_matrix_red2blue = np.where(valid_mask, distances_red2blue, np.inf)
        self.directions_matrix_red2blue = np.where(valid_mask, directions_angles_red2blue, np.inf)
        self.angles_diff_matrix_red2blue = np.where(valid_mask, angles_diff_matrix_red2blue, np.inf)
        
        # Reuse the delta_red2blue for blue to red by swapping the axes
        delta_blue2red = -delta_red2blue.transpose(1, 0, 2)                                             # nblue x nred x 2

        # Calculate direction angles from blue agents to red agents
        directions_angles_blue2red = np.arctan2(delta_blue2red[:, :, 1], delta_blue2red[:, :, 0])       # nblue x nred
        
        # Calculate angles between the red and blue planes
        angles_diff_matrix_blue2red = directions_angles_blue2red - self.blue_directions[np.newaxis, :]  # nblue x nred
        angles_diff_matrix_blue2red = (angles_diff_matrix_blue2red + np.pi) % (2 * np.pi) - np.pi

        # Transpose the distances matrix for red to blue to get blue to red
        distances_blue2red = distances_red2blue.T

        # Calculate distances from blue agents to red agents
        distances_blue2red = distance.cdist(self.blue_positions, self.red_positions, 'euclidean')

        # Update the directions and distances matrices
        self.distances_matrix_blue2red = np.where(valid_mask.T, distances_blue2red, np.inf)
        self.directions_matrix_blue2red = np.where(valid_mask.T, directions_angles_blue2red, np.inf)
        self.angles_diff_matrix_blue2red = np.where(valid_mask.T, angles_diff_matrix_blue2red, np.inf)

        # Calculate distances and update matrix for red to red
        distances_red2red = distance.cdist(self.red_positions, self.red_positions, 'euclidean')
        valid_mask = self.red_alives[:, np.newaxis] & self.red_alives[np.newaxis, :]
        self.distances_matrix_red2red = np.where(valid_mask, distances_red2red, np.inf)


    def update_observed_agents(self):
        # Reset the observed allies list and fill it with -1 values
        self.red_observed_allies = -np.ones((self.n_reds, self.max_observed_allies), dtype=int)
        # Reset the observed enemies list and fill it with -1 values
        self.red_observed_enemies = -np.ones((self.n_reds, self.max_observed_enemies), dtype=int)
        # Reset the attacking target
        self.red_targets = -np.ones(self.n_reds, dtype=int)
        self.blue_targets = -np.ones(self.n_blues, dtype=int)
        
        # Check if the allies are within the field of detection
        in_radius_allies = (self.distances_matrix_red2red < self.detection_radius) & (self.distances_matrix_red2red > 0)
        # Check if the enemies are within the field of detection
        in_radius_enemies = (self.distances_matrix_red2blue < self.detection_radius)
        # Check if the enemies are within the field of attack
        in_attack_range_red = (self.distances_matrix_red2blue < self.attack_radius) & (np.abs(self.angles_diff_matrix_red2blue) < self.attack_angle / 2)
        in_attack_range_blue = (self.distances_matrix_blue2red < self.attack_radius) & (np.abs(self.angles_diff_matrix_blue2red) < self.attack_angle / 2)

        # Create an array of indices for sorting purpose
        ally_indices = np.argsort(self.distances_matrix_red2red, axis=1)
        enemy_indices_red = np.argsort(self.distances_matrix_red2blue, axis=1)
        enemy_indices_blue = np.argsort(self.distances_matrix_blue2red, axis=1)

        # Update the observed allies list for each agent
        for i in range(self.n_reds):
            if not self.red_alives[i]:
                continue
    
            in_radius_allies_i = in_radius_allies[i]
            sorted_allies = ally_indices[i, in_radius_allies_i]
            observed_allies_num = min(len(sorted_allies), self.max_observed_allies)
            self.red_observed_allies[i, :observed_allies_num] = sorted_allies[:observed_allies_num]
            
            in_radius_enemies_i = in_radius_enemies[i]
            sorted_enemies = enemy_indices_red[i, in_radius_enemies_i]
            observed_enemies_num = min(len(sorted_enemies), self.max_observed_enemies)
            self.red_observed_enemies[i, :observed_enemies_num] = sorted_enemies[:observed_enemies_num]

            in_attack_range_i = in_attack_range_red[i]
            sorted_targets = enemy_indices_red[i, in_attack_range_i]
            if len(sorted_targets) > 0:
                self.red_targets[i] = sorted_targets[0]

        for i in range(self.n_blues):
            if not self.blue_alives[i]:
                continue
            in_attack_range_i = in_attack_range_blue[i]
            sorted_targets = enemy_indices_blue[i, in_attack_range_i]
            if len(sorted_targets) > 0:
                self.blue_targets[i] = sorted_targets[0]
            

    def get_obs_agent(self, agent_id):
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        ally_feats = np.zeros((self.max_observed_allies, self.obs_ally_feats_size), dtype=np.float32)
        enemy_feats = np.zeros((self.max_observed_enemies, self.obs_enemy_feats_size), dtype=np.float32)

        if self.red_alives[agent_id]:
            # own_feats
            own_feats[0:2] = self.red_positions[agent_id] / np.array([self.size_x / 2, self.size_y / 2])
            own_feats[2] = (self.red_velocities[agent_id] - self.red_min_vel) / (self.red_max_vel - self.red_min_vel)
            own_feats[3] = self.red_directions[agent_id] / np.pi

            # ally_feats
            valid_allies = self.red_observed_allies[agent_id][self.red_observed_allies[agent_id] != -1]
            if len(valid_allies) > 0:
                ally_positions = self.red_positions[valid_allies]
                ally_feats[:len(valid_allies), 0:2] = (ally_positions - self.red_positions[agent_id]) / self.detection_radius
                ally_feats[:len(valid_allies), 2] = self.distances_matrix_red2red[agent_id, valid_allies] / self.detection_radius
                ally_feats[:len(valid_allies), 3] = self.red_directions[valid_allies] / np.pi

            # enemy_feats
            valid_enemies = self.red_observed_enemies[agent_id][self.red_observed_enemies[agent_id] != -1]
            if len(valid_enemies) > 0:
                enemy_positions = self.blue_positions[valid_enemies]
                enemy_feats[:len(valid_enemies), 0:2] = (enemy_positions - self.red_positions[agent_id]) / self.detection_radius
                enemy_feats[:len(valid_enemies), 2] = self.distances_matrix_red2blue[agent_id, valid_enemies] / self.detection_radius
                enemy_feats[:len(valid_enemies), 3] = self.blue_directions[valid_enemies] / np.pi
                enemy_feats[:len(valid_enemies), 4] = self.angles_diff_matrix_red2blue[agent_id, valid_enemies] / (self.attack_angle / 2)
        
        agent_obs = np.concatenate(
            (
                own_feats,
                ally_feats.flatten(),
                enemy_feats.flatten()
            )
        )

        return agent_obs

    def get_obs(self):
        self.update_observed_agents()
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_reds)]

        return agents_obs
    
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
    
    def scripted_policy(self):
        acc_action = np.random.randint(0, self.acc_action_num, size=self.n_agents)
        heading_action = np.random.randint(0, self.heading_action_num, size=self.n_agents)
        attack_action = np.random.randint(0, self.attack_action_num, size=self.n_agents)

        actions = np.column_stack((acc_action, heading_action, attack_action))

        return actions 
    

    def scripted_policy_blue(self):
        acc_action = np.random.randint(0, self.acc_action_num, size=self.n_blues)
        heading_action = np.random.randint(0, self.heading_action_num, size=self.n_blues)
        attack_action = np.random.randint(0, self.attack_action_num, size=self.n_blues)

        actions = np.column_stack((acc_action, heading_action, attack_action))

        return actions 
    
    def scripted_policy_red(self):
        acc_action = np.random.randint(0, self.acc_action_num, size=self.n_reds)
        heading_action = np.random.randint(0, self.heading_action_num, size=self.n_reds)
        attack_action = np.random.randint(0, self.attack_action_num, size=self.n_reds)

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

    def get_rotated_image(self, image, angle, cache):
        angle = round(angle)
        if angle not in cache:
            rotated_img = pygame.transform.rotate(image, -angle)
            cache[angle] = rotated_img
        
        return cache[angle]
    
    def render_circle(self, center, radius, color):
        center_screen = self.transform_position(center)
        radius_screen = int(radius / self.size_x * self.screen_width)
        pygame.draw.circle(self.screen, color, center_screen, radius_screen, 2)

    def render(self, frame_num=0, save_frames=False):

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            red_plane_img = pygame.image.load('./png/red_plane_s.png').convert_alpha()
            blue_plane_img = pygame.image.load('./png/blue_plane_s.png').convert_alpha()

            # 缩放飞机贴图
            scale_factor = 0.25  # 调整缩放比例
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
                rotated_img = self.get_rotated_image(image, self.angles[i], cache)
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

    world = World()

    import time
    world.reset()
    num_frames = 500

    time_list = []
    
    for i in range(num_frames):
        print('-'* 30)
        start_time = time.time()
        last_time = time.time()

        world.update_matrices()
        print("更新矩阵: {:.5f}".format(time.time() - last_time))
        last_time = time.time()

        world.get_obs()
        print("获取观测: {:.5f}".format(time.time() - last_time))
        last_time = time.time()

        actions = world.scripted_policy()
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

    create_gif("frames", "output.gif", fps=100)


    


        





        



