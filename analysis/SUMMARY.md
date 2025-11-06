# DeepRacer experiment summary

## Objective
Optimize reward function for time-trial racing on unknown evaluation track

## Iterations

| Model | Key features | Training track | Best lap times | Notes |
|-------|-------------|----------------|----------------|-------|
| Baseline | Centerline only | re:Invent 2022 | 32s (rI), 62s (Bar), 82s (Rogue) | too conservative |
| V1 | Progress + Speed | Barcelona | 37s (rI), 45s (Bar), 40s (Ace) | poor generalization |

## Key findings
1. Progress rewards increase speed (+19%) but don't guarantee generalization
2. Blind speed incentives create bimodal behavior (slow OR quick, not adaptive)
3. Track-specific training may limit cross-track performance
4. 100% completion maintained across all tests

## Next: geometry aware rewards (curvature-based speed matching)
