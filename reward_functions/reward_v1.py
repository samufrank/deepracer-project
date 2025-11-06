# v1 reward function: progress + speed + centerline
# baseline (AWS default): pure centerline following, no speed incentive
# v1 changes:
#   - base reward: progress/steps (efficiency metric)
#   - speed multiplier: (1 + speed/max_speed) when near centerline
#   - centerline: reduced to bonus (1.1-1.2x) vs baseline's primary reward
#   - steering penalty: >15 deg gets 0.8x multiplier
# expected: faster lap times, maintained completion rate
# actual: +28% on barcelona, -16% on re:invent, 100% completion maintained

def reward_function(params):
    # base: progress efficiency (missing from examples)
    reward = params['progress'] / (params['steps'] + 1)
    
    # center line bonus (AWS ex 1 logic)
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    
    if distance_from_center <= marker_1:
        reward *= 1.2
    elif distance_from_center <= marker_2:
        reward *= 1.1
    
    # speed reward (NOT in AWS examples)
    speed = params['speed']
    if distance_from_center < 0.4 * track_width:
        reward *= (1.0 + speed / 4.0)
    
    # steering smoothness (AWS ex 2 logic)
    abs_steering = abs(params['steering_angle'])
    if abs_steering > 15:
        reward *= 0.8
    
    return float(reward)
