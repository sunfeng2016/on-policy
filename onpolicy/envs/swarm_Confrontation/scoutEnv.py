# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-17
# @Description: Implementation of Agent Class

import os
import random
import pygame
import numpy as np

from baseEnv import BaseEnv
from scipy.spatial import distance

image_dir = "/home/ubuntu/sunfeng/MARL/on-policy/onpolicy/envs/swarm_Confrontation/"
os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

class ScoutEnv(BaseEnv):
    def __init__(self):
        super(ScoutEnv, self).__init__()

        # 蓝方待侦察区域
        self.scout_width = 3000
        self.scout_height = 4000

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
        self.core_ranges_num = 8
        self.core_ranges = random.sample(self.candidate_core, self.core_ranges_num)

        # 蓝方的威胁区域
        self.threat_ranges_num = 3
        self.threat_ranges = [
            {'center': np.array([-1250.0,   250.0]), 'radius': 250.0},
            {'center': np.array([-1100.0,  -700.0]), 'radius': 300.0},
            {'center': np.array([ 1000.0,  -800.0]), 'radius': 200.0},
        ]

    def transform_circles(self):
        self.transformed_circles_center = []
        self.transformed_circles_radius = []
        self.circles_width = [2] * self.core_ranges_num + [0] * self.threat_ranges_num

        circles = self.core_ranges + self.threat_ranges

        for circle in circles:
            self.transformed_circles_center.append(self.transform_position(circle['center']))
            self.transformed_circles_radius.append(circle['radius'] / self.size_x * self.screen_width)


    
    def render(self, frame_num=0, save_frames=False):
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

            self.transform_circles()
            self.transformed_rect = (0, 0, self.scout_width*self.scale_factor, self.scout_height*self.scale_factor)
            self.num_circles = self.core_ranges_num + self.threat_ranges_num

            # 初始化字体
            self.font = pygame.font.SysFont(None, 36)

        self.screen.fill((255, 255, 255))
        self.transform_positions()
        angles = np.degrees(self.directions)

        # 渲染 rect
        # pygame.draw.rect(self.screen, (255, 0, 0), self.transformed_rect)

        # 渲染 Circles
        for i in range(self.num_circles):
            print(i, self.transformed_circles_center[i])
            pygame.draw.circle(self.screen, (255, 0, 0), self.transformed_circles_center[i], 
                               self.transformed_circles_radius[i], width=self.circles_width[i])
        
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

        if save_frames:
            frame_path = f"{image_dir}/frames/frame_{frame_num:04d}.png"
            pygame.image.save(self.screen, frame_path)

            


if __name__ == "__main__":
    env = ScoutEnv()

    env.reset()

    env.render(frame_num=0, save_frames=True)
    



    


    


        





        



