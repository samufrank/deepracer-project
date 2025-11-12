import math

def reward_function(params):
    """
    v5: v3 base + curvature + track width geometry awareness
    
    strategy: teach agent when to use 0.5, 2.5, or 4.0 m/s based on track geometry
    - wide straights = reward high speed (3.0-4.0 m/s)
    - narrow turns = reward slow speed (<2.0 m/s)
    - moderate geometry = reward middle speeds (2.0-3.0 m/s)
    
    goal: break bimodal distribution by explicitly rewarding geometry-appropriate speeds
    """
    # param
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    steering_angle = params['steering_angle']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    
    # minimal reward for off-track or no progress
    if not all_wheels_on_track or progress < 1.0:
        return 1e-3
    
    ###########################################################################
    # V3 BASE REWARD (proven foundation)
    ###########################################################################
    
    # scaled progress rewards
    reward = (progress / 100.0) * 5.0
    
    # step efficiency (rewards speed implicitly)
    reward += (progress / (steps + 1)) * 3.0
    
    ###########################################################################
    # GEOMETRY CALCULATION (curvature + track width)
    ###########################################################################
    
    def calculate_curvature(waypoints, closest_index, lookahead=10):
        """
        calculate upcoming curvature using next N waypoints
        returns curvature value (0 = straight, higher = sharper turn)
        """
        # get future waypoints (wrap around if needed)
        num_waypoints = len(waypoints)
        future_indices = [(closest_index + i) % num_waypoints for i in range(lookahead + 1)]
        future_wps = [waypoints[i] for i in future_indices]
        
        # calculate heading changes between consecutive waypoints
        heading_changes = []
        for i in range(len(future_wps) - 1):
            x1, y1 = future_wps[i]
            x2, y2 = future_wps[i + 1]
            
            # calculate direction angle
            angle = math.atan2(y2 - y1, x2 - x1)
            if i > 0:
                # heading change from previous segment
                prev_angle = math.atan2(future_wps[i][1] - future_wps[i-1][1],
                                       future_wps[i][0] - future_wps[i-1][0])
                heading_change = abs(angle - prev_angle)
                
                # normalize to [0, pi]
                if heading_change > math.pi:
                    heading_change = 2 * math.pi - heading_change
                heading_changes.append(heading_change)
        
        # avg heading change = curvature
        if len(heading_changes) > 0:
            return sum(heading_changes) / len(heading_changes)
        return 0.0
    
    # calculate upcoming curvature
    closest_index = closest_waypoints[1]
    curvature = calculate_curvature(waypoints, closest_index, lookahead=10)
    
    # track width (absolute value, not normalized)
    # in meters
    width = track_width
    
    ###########################################################################
    # GEOMETRY-AWARE SPEED BONUSES (No penalties, only rewards)
    ###########################################################################
    
    # categorize track geometry
    # curvature thresholds (radians): 0.05 = gentle, 0.15 = moderate, 0.25+ = sharp
    # width thresholds (meters): 0.7 = narrow, 0.9 = moderate, 1.1+ = wide
    
    geometry_bonus = 1.0  # default
    
    # WIDE STRAIGHT: reward high speed (3.0-4.0 m/s)
    if curvature < 0.05 and width > 1.0:
        if speed >= 3.5:
            geometry_bonus = 1.5  # big reward for very quick
        elif speed >= 3.0:
            geometry_bonus = 1.3  # good reward for quick
        elif 2.0 <= speed < 3.0:
            geometry_bonus = 1.1  # small reward for moderate (could go quicker)
    
    # MODERATE STRAIGHT: reward moderate-high speed (2.5-3.5 m/s)
    elif curvature < 0.05 and 0.8 <= width <= 1.0:
        if 3.0 <= speed <= 3.5:
            geometry_bonus = 1.3
        elif 2.5 <= speed < 3.0:
            geometry_bonus = 1.2
        elif 2.0 <= speed < 2.5:
            geometry_bonus = 1.1
    
    # WIDE MODERATE TURN: reward middle speeds (2.0-3.0 m/s)
    elif 0.05 <= curvature < 0.15 and width > 0.9:
        if 2.5 <= speed <= 3.0:
            geometry_bonus = 1.3  # optimal for this geometry
        elif 2.0 <= speed < 2.5:
            geometry_bonus = 1.2
        elif 1.5 <= speed < 2.0:
            geometry_bonus = 1.1
    
    # MODERATE TURN: reward middle-low speeds (1.5-2.5 m/s)
    elif 0.05 <= curvature < 0.15:
        if 2.0 <= speed <= 2.5:
            geometry_bonus = 1.3
        elif 1.5 <= speed < 2.0:
            geometry_bonus = 1.2
        elif 1.0 <= speed < 1.5:
            geometry_bonus = 1.1
    
    # SHARP TURN (any width): reward slow speeds (<2.0 m/s)
    elif curvature >= 0.15:
        if speed <= 1.5:
            geometry_bonus = 1.4  # big reward for being slow in sharp turn
        elif 1.5 < speed <= 2.0:
            geometry_bonus = 1.2
        elif 2.0 < speed <= 2.5:
            geometry_bonus = 1.0  # Nnutral, no bonus or penalty
    
    # NARROW SECTION (regardless of curvature): xtra caution bonus
    if width < 0.7:
        if speed < 2.0:
            geometry_bonus *= 1.1  # extra 10% for being cautious
    
    # apply geometry bonus
    reward *= geometry_bonus
    
    ###########################################################################
    # V3 CONSTRAINTS (for stability)
    ###########################################################################
    
    # center line bonus (encourage racing line)
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    
    if distance_from_center <= marker_1:
        reward *= 1.2
    elif distance_from_center <= marker_2:
        reward *= 1.1
    
    # steering penalty (encourage smooth control)
    if abs(steering_angle) > 15:
        reward *= 0.8
    
    # off-track penalty (hard safety boundary)
    if not all_wheels_on_track:
        reward *= 0.5
    
    return float(reward)
