# V2 reward function: curvature-based speed matching with penalties
# calculates turn sharpness from waypoints, rewards geometry-appropriate speed
# curvature thresholds: <15° straight, 15-50° moderate, >50° sharp
# includes penalties for wrong speed-geometry pairs (0.6-0.8x)
# result: catastrophic failure - 2.38 m/s speed but 60s laps
# issue: penalties too aggressive, short lookahead, restricted action space

import math

def reward_function(params):
    # extract param
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    steering_angle = params['steering_angle']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    all_wheels_on_track = params['all_wheels_on_track']
    
    # base reward: progress efficiency (scaled up)
    reward = (progress / 100.0) * 5.0  # 0-5
    
    # step efficiency
    if progress > 0:
        reward += (progress / (steps + 1)) * 3.0
    
    # calculate upcoming curvature
    next_point = waypoints[closest_waypoints[1]]
    further_point = waypoints[(closest_waypoints[1] + 3) % len(waypoints)]
    prev_point = waypoints[closest_waypoints[0]]
    
    # tack direction vectors
    track_direction = math.atan2(
        next_point[1] - prev_point[1],
        next_point[0] - prev_point[0]
    )
    future_direction = math.atan2(
        further_point[1] - next_point[1],
        further_point[0] - next_point[0]
    )
    
    # calc direction change (curvature proxy)
    direction_diff = abs(track_direction - future_direction)
    if direction_diff > math.pi:
        direction_diff = 2 * math.pi - direction_diff
    curvature_deg = math.degrees(direction_diff)
    
    # speed rewards based on curvature
    if curvature_deg < 15:  # straight
        if speed > 3.0:
            reward *= 1.5
        elif speed < 2.0:
            reward *= 0.7  # pnalize going slow on straights
    elif curvature_deg < 50:  # moderate turn
        if 2.0 < speed < 3.5:
            reward *= 1.3
        elif speed > 3.5:
            reward *= 0.8  # too quick for turn
    else:  # sharp turn
        if speed < 2.5:
            reward *= 1.4
        elif speed > 3.0:
            reward *= 0.6  # calm down bud
    
    # heading alignment
    heading_diff = abs(track_direction - math.radians(heading))
    if heading_diff > math.pi:
        heading_diff = 2 * math.pi - heading_diff
    heading_reward = 1 - (heading_diff / math.pi)
    reward *= (1.0 + heading_reward * 0.2)
    
    # minimal centerline bonus
    if distance_from_center <= 0.1 * track_width:
        reward *= 1.1
    elif distance_from_center <= 0.25 * track_width:
        reward *= 1.05
    
    # steering smoothness
    if abs(steering_angle) > 20:
        reward *= 0.8
    
    # off-track penalty
    if not all_wheels_on_track:
        reward *= 0.5
    
    return float(reward)
