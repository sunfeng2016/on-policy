import numpy as np

plane_param_registry = {
    "plane_defense": {
        "red_vel": (15.0, 40.0),                # 红方（防守方）的速度范围
        "blue_vel": (20.0, 45.0),               # 蓝方（进攻方）的速度范围
        "max_angular_vel": 17.0 * np.pi / 180,  # 角速度
        "detection_radius": 500.0,              # 通信半径
        "view_angle": 34.0 * np.pi / 180,       # 视野角度
        "collide_radius": 15.0,                 # 碰撞距离
        "collide_angle": 34.0 * np.pi / 180,    # 碰撞角度
        "explode_radius": 30.0,                 # 自爆半径
        "softkill_radius": 50.0,                # 软杀伤半径
        "softkill_prob": 0.60,                  # 软杀伤概率
        "softkill_time": 1,                     # 软杀伤次数
        "interfere_radius": 300.0,              # 干扰半径
        "interfere_angle": 20.0 * np.pi / 180,  # 干扰角度
        "interfere_duration": 5,                # 干扰持续时间
    },
    
    "plane_scout": {
        "red_vel": (20.0, 45.0),
        "blue_vel": (15.0, 40.0),
        "max_angular_vel": 17.0 * np.pi / 180,
        "detection_radius": 500.0,
        "view_angle": 34.0 * np.pi / 180,
        "collide_radius": 15.0,
        "collide_angle": 34.0 * np.pi / 180,
        "explode_radius": 30.0,
        "softkill_radius": 50.0,
        "softkill_prob": 0.60,
        "softkill_time": 1,
        "interfere_radius": 300.0,
        "interfere_angle": 20.0 * np.pi / 180,
        "interfere_duration": 5,
    }
}

def get_plane_params(plane_name):
    return plane_param_registry[plane_name]