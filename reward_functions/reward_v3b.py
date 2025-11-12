# v3b reward function: multiplicative speed gradient test
# tests whether continuous speed multiplier fixes bimodal distribution
# reduced base progress (3+2 vs 5+3) to accommodate multiplier
# speed creates 1.125x to 2.0x gradient across 0.5-4.0 m/s range
# result: fixed extremes (ace -32%, kuei -46%) but broke re:invent (+62%)
# critical failure: speed-reward correlation +0.260 (flat, multiplier too weak)
# episode efficiency: +272% regression (learning inefficiency)
# action space: still 10% utilization (bimodal persists despite gradient)
# conclusion: multiplicative approach failed, reduced base reward hurt learning signal

import math

def reward_function(params):
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    steering_angle = params['steering_angle']
    all_wheels_on_track = params['all_wheels_on_track']
    
    # reduced base rewards (leave room for speed multiplier)
    reward = (progress / 100.0) * 3.0  # was 5.0
    if progress > 0:
        reward += (progress / (steps + 1)) * 2.0  # was 3.0
    
    # changed: multiplicative speed gradient instead of additive bonus
    # creates smooth incentive from 0.5 to 4.0 m/s
    if distance_from_center < 0.4 * track_width and all_wheels_on_track:
        speed_multiplier = 1.0 + (speed / 4.0)  # 1.125x at 0.5 m/s, 2.0x at 4.0 m/s
        reward *= speed_multiplier
    
    # centerline bonus
    if distance_from_center <= 0.1 * track_width:
        reward *= 1.2
    elif distance_from_center <= 0.25 * track_width:
        reward *= 1.1
    
    # steering smoothness
    if abs(steering_angle) > 15:
        reward *= 0.8
    
    # off-track penalty
    if not all_wheels_on_track:
        reward *= 0.5
    
    return float(reward)
