# V1b reward function: progress + centerline (diagnostic)
# removes V1's speed multiplier to isolate generalization issue
# base: scaled progress rewards (0-8 range)
# centerline: reduced to 1.1-1.2x bonus
# result: fixed reinvent regression (32.8s vs V1's 37.3s)
# confirmed: progress/steps inherently rewards speed without explicit multiplier

import math

def reward_function(params):
    # extract parameters
    progress = params['progress']
    steps = params['steps']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    steering_angle = params['steering_angle']
    all_wheels_on_track = params['all_wheels_on_track']
    
    # base reward: progress efficiency (scaled up)
    reward = (progress / 100.0) * 5.0  # 0-5 range
    
    # step efficiency bonus
    if progress > 0:
        reward += (progress / (steps + 1)) * 3.0
    
    # centerline bonus (minimal)
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    
    if distance_from_center <= marker_1:
        reward *= 1.2
    elif distance_from_center <= marker_2:
        reward *= 1.1
    
    # steering smoothness
    if abs(steering_angle) > 15:
        reward *= 0.8
    
    # off-track penalty
    if not all_wheels_on_track:
        reward *= 0.5
    
    return float(reward)
