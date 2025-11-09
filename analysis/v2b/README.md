# V2b: Curvature bonuses only (no penalties)

Test whether V2's aggressive penalties caused performance collapse. Keeps curvature calculation and geometry-aware bonuses but removes all penalty multipliers. Restores 0.5-4.0 m/s action space.

## Implementation changes
Same curvature logic as V2  
Reduced bonus multipliers: 1.2-1.3x (down from 1.3-1.5x)  
REMOVED: All penalty multipliers (0.6x, 0.7x, 0.8x)  
Restored: 0.5-4.0 m/s action space  

## Training config
Track: Barcelona (60m, 16 turns, clockwise)  
Duration: 90 minutes (24 iterations)  
Action space: Speed 0.5-4.0 m/s, steering +/- 30 deg    

## Results

### Training metrics
Completion rate: 0%  
Best progress: 51.9%  
Mean speed: 2.08 m/s (dropped back to v1 level)  
Fewer failures: 468 vs v2's 785  

### Evaluation performance (5 trials per track)

| Track | V2 Best | V2b Best | Change |
|-------|---------|----------|--------|
| Barcelona | 59.90s | 47.30s | -21% |
| re:Invent 2022 | 60.43s | 35.04s | -42% |
| Ace Speedway | 42.41s | 49.45s | +17% |

Completion rate: 15/15 (100%)  

## Analysis

Partial recovery: Better than V2 but worse than V1/V1b  
Removing penalties removed crash behavior but also removed speed incentive  
Speed dropped to 2.08 m/s - curvature bonuses alone too weak  
Ace Speedway performance (49s on simple oval) suggests bonuses not shaping behavior  

Ace failures clustered at hairpin (WP53-60) - curvature logic not detecting/handling sharp turn properly  

Conclusion: Penalties were causing V2's crashes, but bonuses insufficient without them  
Curvature approach needs more tuning than timeline allows  

next: v3 returns to v1b proven base, adds conservative speed bonus  
