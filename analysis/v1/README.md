# V1: Progress + speed rewards

Baseline's pure centerline following creates overly conservative behavior. Adding progress efficiency and speed incentives should reduce lap times while maintaining completion rate.

## Implementation changes
Base reward: `(progress/100) * 5.0 + (progress/steps) * 3.0` - scales reward magnitude, incentivizes efficiency
Speed multiplier: `(1 + speed/4.0)` when within 40% of track width - rewards higher throttle usage
Centerline: Reduced from primary reward to 1.1-1.2x bonus - maintains stability without over-constraining
Steering penalty: 0.8x for abs(angle) > 15° - encourages smooth control

## Training config
Track: Barcelona (60m, 16 turns, clockwise)
Duration: 120 minutes (26 iterations)
Action space: Speed 0.5 - 4.0 m/s, steering +/- 30°

## Results

### Training metrics
Completion rate: 0% (episode time limit, not policy failure)
Best progress: 68.2%
Mean speed: 2.08 m/s (vs baseline 1.74 m/s, +19%)
Speed distribution: Bimodal at 0.5 and 4.0 m/s

### Evaluation performance (5 trials per track)

| Track | Baseline Best | V1 Best | Change |
|-------|---------------|---------|--------|
| Barcelona (training) | 62.38s | 44.62s | -28% |
| re:Invent 2022 | 32.09s | 37.32s | +16% |
| Ace Speedway | N/A | 40.27s | N/A |

Completion rate: 15/15 (100%)

## Analysis

Successes:
Speed increased from 1.74 to 2.08 m/s (progress+speed rewards working)
Training track lap time improved 28%
Maintained 100% completion reliability

Failures:
Regressed on re:Invent 2022 despite being more "advanced"
Speed distribution shows binary behavior (slow OR quick, not adaptive)
Poor cross-track generalization

Root causes:
1. Speed reward not geometry-aware - rewards speed regardless of turn sharpness
2. Training on Barcelona may have learned track-specific behaviors
3. Bimodal speed suggests agent hasn't learned nuanced throttle control

## Next
V1b: isolate progress component (remove speed multiplier) to determine if speed rewards caused re:Invent regression
