# V2: Curvature-based speed matching

Implement geometry-aware speed rewards using waypoint curvature calculation. Reward appropriate speed for turn sharpness: quick on straights, slow in sharp turns. Includes penalties for wrong speed-geometry pairs.

## Implementation changes
Base reward: Scaled progress `(progress/100) * 5.0 + (progress/steps) * 3.0`
Curvature calculation: Uses waypoints to compute upcoming turn angle (+3 waypoint lookahead)
Speed-curvature matching:
  - <15° (straight): 1.5x bonus if speed >3.0, 0.7x penalty if speed <2.0
  - 15-50° (moderate): 1.3x bonus for 2.0-3.5 m/s, 0.8x penalty if >3.5
  - >50° (sharp): 1.4x bonus if speed <2.5, 0.6x penalty if >3.0
Heading alignment: Reward pointing toward track direction
Minimal centerline bonus: 1.05-1.1x

## Training config
Track: Barcelona (60m, 16 turns, clockwise)
Duration: 120 minutes (40 iterations)
Action space: Speed 1.0-4.0 m/s (accidentally changed from 0.5-4.0), steering ±30°

## Results

### Training metrics
Completion rate: 0%
Best progress: 51.6% (worse than V1's 68.2%)
Mean speed: 2.38 m/s (highest yet)
More failures: 785 vs V1's 511

### Evaluation performance (5 trials per track)

| Track | V1b Best | V2 Best | Change |
|-------|----------|---------|--------|
| Barcelona | 51.19s | 59.90s | +17% |
| re:Invent 2022 | 32.83s | 60.43s | +84% |
| Ace Speedway | N/A | 42.41s | N/A |

Completion rate: 15/15 (100%)

## Analysis

Catastrophic failure: Highest speed (2.38 m/s) but slowest lap times
Agent going quick but crashing/correcting constantly
Speed-progress efficiency dropped to 0.0394 (was 0.0643 in V1b)

Root causes:
1. Penalties too aggressive - agent trapped in lose-lose scenarios
2. Restricted action space (1.0-4.0 instead of 0.5-4.0) prevented slow cornering
3. Short lookahead (+3 waypoints ≈ 0.7m) insufficient for speed planning
4. Curvature thresholds (15°/50°) may be miscalibrated

Conclusion: Concept sound but implementation needs significant debugging
Next: V2b removes penalties to isolate whether they caused the failure
