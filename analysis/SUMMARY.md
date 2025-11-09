# DeepRacer experiments summary

## Objective
Optimize reward function for time-trial racing on unknown evaluation track

## Iterations

| Model | Key Features | Training Track | re:Invent | Barcelona | Ace | Kuei | Mean Speed | Speed-Reward | Episode Eff | Action Space | Status |
|-------|-------------|----------------|-----------|-----------|-----|------|------------|--------------|-------------|--------------|--------|
| Baseline | Centerline only | re:Invent 2022 | 32.09s | 62.38s | N/A | N/A | 1.74 m/s | N/A | N/A | N/A | Conservative |
| V1 | Progress + speed | Barcelona | 37.32s | 44.62s | 40.27s | N/A | 2.08 m/s | N/A | N/A | N/A | Generalization fail |
| V1b | Progress only | Barcelona | 32.83s | 51.19s | N/A | N/A | 2.11 m/s | N/A | N/A | N/A | Diagnostic success |
| V2 | Curvature + penalties | Barcelona | 60.43s | 59.90s | 42.41s | N/A | 2.38 m/s | N/A | N/A | N/A | Catastrophic |
| V2b | Curvature bonuses | Barcelona | 35.04s | 47.30s | 49.45s | N/A | 2.08 m/s | N/A | N/A | N/A | Partial recovery |
| V3 | V1b + additive speed | Barcelona | 29.21s | 47.97s | 50.59s | 79.86s | 1.84 m/s | +0.217 | +512% | 10% | Best re:Invent |
| V3b | Multiplicative gradient | Barcelona | 47.44s | 45.02s | 34.51s | 42.88s | 2.02 m/s | +0.260 | +272% | 10% | Extreme optimization |

Metrics:
- Speed-Reward: correlation between speed and reward (higher = stronger incentive)  
- Episode Eff: change in episode length early -> late training (negative = improving, positive = regressing)  
- Action Space: % of speed bins used above mean frequency (higher = more continuous)  

All models: 100% evaluation completion rate  

## Key findings

### What worked
Reliability first: all models maintained 100% eval completion despite varying training (0-0.2%)  
Progress/steps ratio: inherently rewards speed efficiency without explicit speed terms  
Medium complexity optimization: V3 best re:Invent (29.21s, 9% better than baseline)  
Systematic isolation: V1b diagnostic identified V1's blind speed multiplier as regression cause  
Cross-track evaluation: 5 diverse tracks (23-60m, complexity 0.04-0.34) revealed generalization patterns  

### What failed
Speed incentive weakness: both additive (v3: +0.217) and multiplicative (v3b: +0.260) created weak speed-reward correlation  
Binary action selection: persistent 10% utilization (only 0.5 and 4.0 m/s) across all models despite continuous action space  
Episode efficiency regression: v3 (+512%) and v3b (+272%) showed episodes getting longer - learning to be slower not faster  
Progress dominance: in both V3 (5+3 progress vs 1 speed) and V3b (3+2 progress × 2 speed), progress terms dominated  
Aggressive penalties (v2): created lose-lose scenarios, excessive crashes  
Curvature approach (v2/v2b): short lookahead (+3 waypoints) insufficient, extensive tuning needed  

### Insights
1. Training vs eval disconnect: 0% training completion with 100% eval = episode time limit artifact not policy failure, AWS uses best checkpoint
2. Track complexity metric: curvature variance (0.04-0.34) correlates with performance, enables objective difficulty quantification
3. Speed-reward correlation diagnostic: reveals reward effectiveness independent of lap times, weak (<0.3) = speed not incentivized
4. Episode efficiency as learning signal: length change shows improving (decreasing) vs regressing (increasing), more informative than completion rate
5. Binary action space fundamental: persists across additive and multiplicative - suggests PPO policy or action space issue not reward function
6. Oscillatory speed behavior: V3b high mean (1.90 m/s) but poor times (47.44s) on technical tracks - oscillation wastes time in accel/decel
7. Track complexity tradeoff: V3 optimized medium (re:Invent 0.09), V3b optimized extremes (Ace 0.10, Kuei 0.13) - can't do both
8. Checkpoint selection critical: strong eval despite poor late training = AWS finds better intermediate policies
9. Cross-track patterns: V1 blind speed helped training hurt others, V3 cautious good medium poor extremes, V3b aggressive good extremes catastrophic medium

## Methodology
Systematic isolation: V1b removed single component to diagnose V1  
Multi-track evaluation: 3-5 tracks per iteration for generalization assessment  
Failure spatial analysis: identified geometry-specific crash patterns (Kuei WP46 chicane: 127 failures)  
Advanced metrics: speed-reward correlation, episode efficiency, action space utilization beyond lap times  
Iterative refinement: each iteration informed by previous failure modes  
Track complexity quantification: curvature variance enables objective comparison  

## Current state

V3: best medium complexity (re:Invent 29.21s), 100% reliable, cautious policy (1.84 m/s)
- Weakness: high complexity poor (Kuei 79.86s), episode efficiency +512%, weak speed-reward +0.217
- Success is checkpoint luck (1 lap in 440 episodes) not converged policy

V3b: best extremes (Ace 34.51s, Kuei 42.88s), 100% reliable, aggressive policy (2.02 m/s)
- Weakness: medium complexity catastrophic (re:Invent 47.44s, +62% vs V3), oscillatory speed behavior
- Episode efficiency +272% (better than V3 but still regression), weak speed-reward +0.260

Shared issues: binary action selection (10%), weak speed-reward correlation, episode length increasing, progress dominates  

## Next iteration: V4

Hypothesis: speed must be PRIMARY reward (not modifier) to create strong speed incentive  

Both additive (V3) and multiplicative (V3b) modifiers failed:
- V3: progress base + speed bonus = progress-dominant (+0.217 correlation)
- V3b: progress base × speed multiplier = progress-dominant (+0.260 correlation)

Proposed V4: invert structure
- Speed base × progress amplifier = speed-dominant
- Target: >0.5 speed-reward correlation (ideally >0.6)
- Target: <+100% or negative episode efficiency
- Target: >15% action space utilization

Decision criteria: if v4 shows similar binary selection and weak correlation, problem is PPO/action-space not reward functions
