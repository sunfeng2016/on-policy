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
    },
    "10_vs_10": {
        "n_reds": 10,
        "n_blues": 10,
        "episode_limit": 400,
        "size_x": 8000,
        "size_y": 5000,
        "defender": 'red',
    },
}


def get_sce_map_registry():
    return map_param_registry


for name in map_param_registry.keys():
    globals()[name] = type(name, (SMACMap,), dict(filename=name))


def get_map_params(map_name):
    map_param_registry = get_sce_map_registry()
    return map_param_registry[map_name]
