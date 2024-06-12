# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-17
# @Description: Implementation of Agent Class

import os
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


    


    


        





        



