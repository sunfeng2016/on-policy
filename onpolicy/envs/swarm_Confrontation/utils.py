import pandas as pd
import numpy as np
import pygame
import math

def append_to_csv(data, filename='output.csv'):

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Replace numeric values in '打击' with corresponding text
    df['打击'] = df['打击'].replace({np.nan: np.nan, 0: '不打击', 1: '软毁伤', 2: '硬毁伤'})
    
    # Replace numeric values in '航向' with corresponding text
    df['航向'] = df['航向'].apply(lambda x: np.nan if pd.isnull(x) else (f'顺时针转{-x}度' if x < 0 else (f'逆时针转{x}度' if x > 0 else f'保持航向不变')))
    
    # Replace numeric values in '加速度' with corresponding text
    df['加速度'] = df['加速度'].apply(lambda x: np.nan if pd.isnull(x) else (f'加速{x}' if x > 0 else (f'减速{-x}' if x < 0 else '保持' )))

    # Save DataFrame to CSV
    df.to_csv(filename, index=False, encoding='utf-8', mode='w')


def draw_arrow(screen, start_pos, end_pos, color):
    """
    在屏幕上绘制从 start_pos 指向 end_pos 的箭头
    
    参数:
    screen: pygame 显示的屏幕。
    start_pos: 箭头的起始位置 (x,y)。
    end_pos: 箭头的终点位置 (x, y)。
    color: 箭头的颜色
    """
    width = 3 # 箭头线条宽度
    arrow_length = 10  # 箭头三角形的边长
    arrow_angle = math.pi / 6   # 箭头三角形的角度

    # 计算箭头的角度
    angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
    
    # 计算箭头三角形的顶点位置
    arrow_points = [
        end_pos,
        (
            end_pos[0] - arrow_length * math.cos(angle - arrow_angle),
            end_pos[1] - arrow_length * math.sin(angle - arrow_angle)
        ),
        (
            end_pos[0] - arrow_length * math.cos(angle + arrow_angle),
            end_pos[1] - arrow_length * math.sin(angle + arrow_angle)
        )
    ]

    # 画箭头主线条
    pygame.draw.line(screen, color, start_pos, end_pos, width)

    # 画箭头的三角形部分
    pygame.draw.polygon(screen, color, arrow_points)
            
def draw_sector_old(screen, center, radius, start_angle, end_angle, color):
    """
    在屏幕上绘制一个扇形。

    参数:
    screen: pygame 显示的屏幕。
    center: 扇形的中心点坐标 (x, y)。
    radius: 扇形的半径。
    start_angle: 扇形的起始角度（弧度制）。
    end_angle: 扇形的结束角度（弧度制）。
    color: 扇形的颜色。
    """
    
    # 线条宽度
    width = 3 

    # 扇形的顶点计算
    points = [center]
    angle_range = math.degrees(end_angle - start_angle)
    steps = max(1, int(angle_range))  # 动态计算步数，确保至少绘制一个点
    for step in range(steps + 1):
        angle = start_angle + (end_angle - start_angle) * step / steps
        x = center[0] + int(radius * math.cos(angle))
        y = center[1] - int(radius * math.sin(angle))
        points.append((x, y))

    # 绘制扇形
    # pygame.draw.polygon(screen, color, points)

    # 绘制边框（弧形和边线）
    pygame.draw.arc(screen, color, (center[0] - radius, center[1] - radius, radius * 2, radius * 2), start_angle, end_angle, width=width)
    pygame.draw.line(screen, color, center, points[1], width=width)
    pygame.draw.line(screen, color, center, points[-1], width=width)
    
def draw_sector(screen, center, radius, start_angle, end_angle, color):
    """
    在屏幕上绘制一个扇形。

    参数:
    screen: pygame 显示的屏幕。
    center: 扇形的中心点坐标 (x, y)。
    radius: 扇形的半径。
    start_angle: 扇形的起始角度（弧度制）。
    end_angle: 扇形的结束角度（弧度制）。
    color: 扇形的颜色。
    """
    
    # 线条宽度
    width = 3 
    
    # 扇形的顶点计算
    points = [center]
    
    for angle in [start_angle, end_angle]:
        x = center[0] + int(radius * math.cos(angle))
        y = center[1] - int(radius * math.sin(angle))
        points.append((x, y))

    # 绘制边框（弧形和边线）
    pygame.draw.arc(screen, color, (center[0] - radius, center[1] - radius, radius * 2, radius * 2), start_angle, end_angle, width=width)
    pygame.draw.line(screen, color, center, points[1], width=width)
    pygame.draw.line(screen, color, center, points[-1], width=width)


def assign_target(agents, targets):
    # 分配目标比例
    target_ratios = [0.4, 0.2, 0.4]
    num_agents_per_target = np.array(target_ratios) * len(agents)  # 每个目标点的智能体数量
    num_agents_per_target = num_agents_per_target.astype(int)

    # Step 1: 计算每个智能体到每个目标点的距离
    distances = np.linalg.norm(agents[:, np.newaxis, :] - targets, axis=2)  # (100, 3) 每个智能体到3个目标点的距离

    # Step 2: 找到每个智能体最近的目标点
    initial_assignment = np.argmin(distances, axis=1)  # 每个智能体分配到的最近目标点 (长度为100的数组)

    # Step 3: 调整分配，保证每个目标点分配到的智能体数量符合比例
    final_assignment = np.zeros_like(initial_assignment)  # 用来存储最终的目标分配

    # 统计初始分配的结果
    for target_idx in range(len(targets)):
        num_assigned = np.sum(initial_assignment == target_idx)  # 初始分配给这个目标的智能体数量
        if num_assigned > num_agents_per_target[target_idx]:
            # 如果分配的数量超过比例限制，从分配给该目标的智能体中随机移除部分
            excess_agents = np.where(initial_assignment == target_idx)[0]  # 找到这些智能体的索引
            agents_to_remove = np.random.choice(excess_agents, num_assigned - num_agents_per_target[target_idx], replace=False)
            initial_assignment[agents_to_remove] = -1  # 暂时移除多余的分配
        
        # 现在符合比例的智能体分配给该目标
        final_assignment[initial_assignment == target_idx] = target_idx

    # Step 4: 将还未分配的智能体重新分配到其它目标，确保目标比例满足
    for target_idx in range(len(targets)):
        remaining_agents = np.where(initial_assignment == -1)[0]  # 还未分配的智能体
        needed_agents = num_agents_per_target[target_idx] - np.sum(final_assignment == target_idx)  # 还需要分配的数量
        if needed_agents > 0 and len(remaining_agents) > 0:
            selected_agents = np.random.choice(remaining_agents, needed_agents, replace=False)
            final_assignment[selected_agents] = target_idx  # 重新分配

    return final_assignment