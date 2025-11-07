# DeepRacer experiments summary

## Objective
Optimize reward function for time-trial racing on unknown evaluation track

## Iterations

| Model | Key Features | Training Track | re:Invent | Barcelona | Ace | Mean Speed | Status |
|-------|-------------|----------------|-----------|-----------|-----|------------|--------|
| Baseline | Centerline only | re:Invent 2022 | 32.09s | 62.38s | N/A | 1.74 m/s | Conservative |
| V1 | Progress + Speed | Barcelona | 37.32s | 44.62s | 40.27s | 2.08 m/s | Generalization issue |
| V1b | Progress only | Barcelona | 32.83s | 51.19s | N/A | 2.11 m/s | Diagnostic success |
| V2 | Curvature + penalties | Barcelona | 60.43s | 59.90s | 42.41s | 2.38 m/s | Catastrophic failure |
| V2b | Curvature bonuses | Barcelona | 35.04s | 47.30s | 49.45s | 2.08 m/s | Partial recovery |
| V3 | V1b + speed bonus | Barcelona | TBD | TBD | TBD | TBD | In progress |

All models: 100% completion rate maintained

## Key findings

### What worked
Progress/steps ratio inherently rewards speed without explicit multipliers

Scaled rewards (0.5-2.0 range) enable learning in short training episodes

V1b matched baseline's best time (32.83s vs 32.09s) with simpler approach

### What failed
Blind speed multipliers (V1): Helped training track, hurt generalization

Aggressive penalties (V2): Created lose-lose scenarios, excessive crashes

Weak bonuses (V2b): Insufficient to shape behavior without penalties

Short curvature lookahead (+3 waypoints): Too brief for speed planning

### Insights
1. Training completion (0%) vs eval completion (100%) is episode time limit artifact, not policy failure
2. Cross-track generalization more important than training track optimization
3. Speed-progress efficiency metric reveals true performance (V2: 0.039, V1b: 0.064)
4. Conservative > aggressive for competition format prioritizing completion

## Methodology
- Systematic isolation: V1b removed single component to diagnose V1's regression
- Multi-track evaluation: 3+ tracks per iteration to assess generalization
- Failure spatial analysis: Identified geometry-specific crash patterns
- Iterative refinement: Each iteration informed by previous failure modes

## Next
V3: return to V1b proven base, add conservative additive speed bonus
final: 4-hour training of best-performing approach
