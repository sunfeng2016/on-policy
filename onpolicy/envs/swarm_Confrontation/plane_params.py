import numpy as np

plane_param_registry = {
    "plane_defense": {
        "red_vel": (15.0, 40.0),
        "blue_vel": (20.0, 45.0),
        "max_angular_vel": 17.0 * np.pi / 180,
        "detection_radius": 500.0,
        "attack_radius": 150.0,
        "attack_angle": 34.0 * np.pi / 180,
        "explode_radius": 60.0,
        "collide_distance": 30.0,
        "soft_kill_distance": 300.0,
        "soft_kill_angle": 20.0 * np.pi / 180,
    },
    
    "plane_scout": {
        "red_vel": (20.0, 45.0),
        "blue_vel": (15.0, 40.0),
        "max_angular_vel": 17.0 * np.pi / 180,
        "detection_radius": 500.0,
        "attack_radius": 150.0,
        "attack_angle": 34.0 * np.pi / 180,
        "explode_radius": 60.0,
        "collide_distance": 30.0,
        "soft_kill_distance": 300.0,
        "soft_kill_angle": 20.0 * np.pi / 180,
    }
}

def get_plane_params(plane_name):
    return plane_param_registry[plane_name]