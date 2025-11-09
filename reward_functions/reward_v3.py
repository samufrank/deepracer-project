# v3 reward function: v1b base + conservative additive speed bonus
# returns to v1b's proven progress only approach, adds speed incentive
# base: scaled progress (0-8 range) for efficiency rewards
# new: additive speed bonus +1.0 max when on track near centerline
# result: best re:invent time (29.21s), 100% reliability, first training completion
# trade-off: cautious policy (1.84 m/s), poor on extremes kKuei 79s)


import math

def reward_function(params):
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    steering_angle = params['steering_angle']
    all_wheels_on_track = params['all_wheels_on_track']
    
    # v1b proven base: scaled progress rewards
    reward = (progress / 100.0) * 5.0
    if progress > 0:
        reward += (progress / (steps + 1)) * 3.0
    
    # new: cnoservative speed bonus (additive, not multiplicative)
    if distance_from_center < 0.4 * track_width and all_wheels_on_track:
        reward += (speed / 4.0) * 1.0  # Max +1.0 bonus at top speed
    
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
