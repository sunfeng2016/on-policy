from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib
import numpy as np

class SMACMap(lib.Map):
    directory = "SMAC_Maps"
    download = "https://github.com/oxwhirl/smac#smac-maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


map_param_registry = {
    "100_vs_100": {
        "n_reds": 100,
        "n_blues": 100,
        "episode_limit": 400,
        "size_x": 8000,
        "size_y": 5000,
        "defender": 'red',
        "red_core":  {
            'center': np.array([2250.0, 0.0]),
            'radius': 25.0
        },
        "red_base": {
            'center': np.array([2250.0, 0.0]),
            'radius': 1250.0
        },
        "red_lines": np.array([
            [[1366.0,  884.0], [1750.0,  500.0]],
            [[1750.0,  500.0], [1750.0, -500.0]],
            [[1750.0, -500.0], [1366.0, -884.0]],
            [[3134.0,  884.0], [2750.0,  500.0]],
            [[2750.0,  500.0], [2750.0, -500.0]],
            [[2750.0, -500.0], [3134.0, -884.0]],
        ]),
        "red_square_size": 1000.0 / 2,

        # 左侧威胁区
        "left_sector_pos1": np.array([1366.0, 884.0]),
        "left_sector_pos1": np.array([1366.0, -884.0]),

        # 右侧威胁区
        "right_sector_pos1": np.array([3134.0, -884.0]),
        "right_sector_pos1": np.array([3134.0, 884.0]),

        # blue base
        "blue_bases": [
            {'center': np.array([1500.0, 1500.0]), 'radius': 500.0},    # 上右
            {'center': np.array([1500.0, -1500.0]), 'radius': 500.0},   # 下右
            {'center': np.array([500.0, 1500.0]), 'radius': 500.0},     # 上左
            {'center': np.array([500.0, -1500.0]), 'radius': 500.0},    # 下左
        ],

    },
}


def get_sce_map_registry():
    return map_param_registry


for name in map_param_registry.keys():
    globals()[name] = type(name, (SMACMap,), dict(filename=name))


def get_map_params(map_name):
    map_param_registry = get_sce_map_registry()
    return map_param_registry[map_name]
