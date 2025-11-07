# V1b: Progress-only (diagnostic)

Diagnostic test to isolate whether V1's speed multiplier caused re:Invent 2022 regression. Removes speed component while keeping scaled progress rewards and centerline bonuses.

## Implementation changes
Base reward: `(progress/100) * 5.0 + (progress/steps) * 3.0` - same as V1
Centerline: 1.1-1.2x bonus - same as V1
Steering penalty: 0.8x for abs(angle) > 15° - same as V1
REMOVED: Speed multiplier from V1

## Training config
Track: Barcelona (60m, 16 turns, clockwise)
Duration: 90 minutes (25 iterations)
Action space: Speed 0.5-4.0 m/s, steering ±30°

## Results

### Training metrics
Completion rate: 0%
Best progress: 61.4%
Mean speed: 2.11 m/s (virtually unchanged from V1's 2.08 m/s)
Speed distribution: Bimodal at 0.5 and 4.0 m/s

### Evaluation performance (5 trials per track)

| Track | V1 Best | V1b Best | Change |
|-------|---------|----------|--------|
| Barcelona | 44.62s | 51.19s | +15% |
| re:Invent 2022 | 37.32s | 32.83s | -12% |

Completion rate: 10/10 (100%)

## Analysis

Key finding: Removing speed multiplier fixed re:Invent regression
V1b matched baseline's 32.09s on re:Invent (V1 was 37.32s)
Speed stayed constant (2.11 vs 2.08 m/s) despite removing speed multiplier
Progress/steps ratio inherently rewards speed without explicit multiplier

Conclusion: V1's blind speed multiplier helped on training track but hurt generalization
Next: V2 attempts geometry-aware speed control via curvature calculation
