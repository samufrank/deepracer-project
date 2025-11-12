# v4 reward function: speed-primary with progress amplifier
# structural inversion test - makes speed the base reward, progress amplifies it
# base: raw speed (0.5-4.0 m/s range)
# multiplier: progress creates 1.0x to 3.0x amplifier
# additional: small additive progress term (0-0.5) for baseline
# result: speed-reward correlation +0.721 (3.3x better than V3's +0.217)
# trade-off: oscillatory behavior (0.5<->4.0 switching), re:Invent regressed to 46.7s (+60%)
# episode efficiency: +310% regression (better than v3's +512% but still learning inefficiency)
# action space: still 10% utilization (bimodal persists despite strong speed incentive)
# conclusion: speed-primary hypothesis validated but constraints conflict with execution
# maintains 100% completion across all tracks despite aggressive speed incentive

def reward_function(params):
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    steering_angle = params['steering_angle']
    all_wheels_on_track = params['all_wheels_on_track']
    
    # minimal reward w/o progress
    if progress < 1.0:
        return 1e-3
    
    # INVERTED STRUCTURE: speed is primary reward
    reward = speed  # 0.5 to 4.0 direct contribution
    
    # progress amplifies speed reward
    progress_multiplier = 1.0 + (progress / 100.0) * 2.0  # 1× to 3×
    reward *= progress_multiplier
    
    # step efficiency bonus
    # rewards completing track quickly
    if progress > 0:
        reward += (progress / (steps + 1)) * 1.0 # was 2.0 in v3b
    
    # centerline bonus (from V3b)
    # encourages optimal racing line
    if distance_from_center <= 0.1 * track_width:
        reward *= 1.2
    elif distance_from_center <= 0.25 * track_width:
        reward *= 1.1
    # no bonus/penalty for 0.4-0.5 (full track width)
    
    # steering smoothness (from V3b)
    # penalizes excessive steering
    if abs(steering_angle) > 15:
        reward *= 0.8
    
    # off-track penalty (from V3b)
    # strong penalty for going off-track
    if not all_wheels_on_track:
        reward *= 0.5
    
    return float(reward)
