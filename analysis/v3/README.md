# V3: V1b + conservative speed bonus

Return to V1b's proven base (progress only) and add conservative additive speed bonus. Goal: improve speed while maintaining v1b's cross-track generalization.  

## Implementation changes
Base reward: Scaled progress `(progress/100) * 5.0 + (progress/steps) * 3.0` (same as V1b)  
New: Additive speed bonus `+ (speed/4.0) * 1.0` when on track and <40% from center  
Centerline: 1.1-1.2x bonus (same as V1b)  
Steering penalty: 0.8x for abs(angle) > 15 deg (same as V1b)  

## Training config  
Track: Barcelona (60m, 16 turns, clockwise)  
Duration: 120 minutes (22 iterations)  
Action space: speed 0.5-4.0 m/s, steering +/- 30 deg  

## Results

### Training metrics
Completion rate: 0.2% (1 episode in 440 total - checkpoint luck)  
Best progress: 100.0%  
Mean speed: 1.84 m/s (lower than v1/v1b's ~2.1 m/s)  
Mean reward: 1.322  
Speed-reward correlation: +0.217 (weak - additive bonus barely working)  
Episode efficiency: +512% (early 29 steps -> late 178 steps - getting worse)  
action space utilization: 10% (2/20 bins, bimodal 0.5 and 4.0 m/s only)  
Speed-progress correlation: -0.466 (learned to go slow for safety)  

### Evaluation performance (5 trials per track)

| Track | Length | Complexity | V1b Best | V3 Best | Change | Track Speed |
|-------|--------|------------|----------|---------|--------|-------------|
| Smile | 22.5m | 0.038 | N/A | 20.70s | N/A | 1.09 m/s |
| re:Invent 2022 | 31.8m | 0.090 | 32.83s | 29.21s | -11% | 1.09 m/s |
| Ace Speedway | 50.7m | 0.060 | N/A | 50.59s | N/A | 1.00 m/s |
| Barcelona | 57.9m | 0.148 | 51.19s | 47.97s | -6% | 1.21 m/s |
| Kuei | 47.0m | 0.179 | N/A | 79.86s | N/A | 0.59 m/s |

Completion rate: 25/25 (100%)  
Track speed range: 0.59-1.21 m/s across complexity spectrum  

## Analysis

Best re:Invent time: 29.21s (9% faster than baseline, 11% faster than V1b)  
Perfect reliability: 100% completion across all 5 tracks  
First model to hit 100% training progress (once, likely early exploration not converged policy)  

Problems revealed by new metrics:
- Speed-reward correlation +0.217 = additive bonus too weak (only 12% of total reward range)
- Episode efficiency +512% = model getting slower over training, not faster
- Action space 10% = binary selection (0.5 or 4.0 m/s only) despite continuous space
- Progress terms (8.0 max) dominate over speed bonus (1.0 max) = 89% vs 11% contribution

Cross-track pattern:
- Medium complexity (0.038-0.148): 1.0-1.2 m/s track speed - good
- High complexity (0.179): 0.59 m/s - poor (Kuei WP46 chicane: 127 training failures)

Checkpoint vs convergence disconnect:
- Late training: +512% episode length increase (policy regressing)
- Evaluation: 100% completion, competitive times
- AWS uses best checkpoint, not final model - explains performance despite regression


Root cause: progress dominates decision-making, speed barely matters  
Agent learned "go slow for safety" not "go fast for bonus"  

Conclusion: best re:Invent performance and reliability, but:
- Success is checkpoint luck (1 lap in 440 episodes)
- Late training shows policy regressing toward extreme caution
- Additive approach failed to create strong speed incentive
- Binary action selection persists

Status: leading candidate for competition (29s competitive, 100% reliable)  
but: success despite poor learning dynamics, not because of good reward function  

Next: v3b tests multiplicative gradient to see if relative vs absolute bonuses help  
