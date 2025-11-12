# V4: Speed-primary with progress amplifier

Structural inversion test - makes speed the base reward instead of progress. Goal: test if speed-primary structure creates stronger speed incentive than V3's additive approach.

## Implementation changes
Base reward: Raw speed `speed` (0.5-4.0 m/s range)  
Progress amplifier: `* (1.0 + (progress/100) * 2.0)` creates 1.0x-3.0x multiplier  
Additional: Small additive progress `+ (progress/100) * 0.5` for baseline signal  
Centerline: 1.1-1.2x bonus (same as V3)  
Steering penalty: 0.8x for abs(angle) > 15 deg (same as V3)  

Rationale: if speed is primary and progress amplifies it, agent must go fast to get high rewards. Hypothesis: multiplicative coupling forces speed-progress integration.

## Training config
Track: Barcelona (60m, 16 turns, clockwise)  
Duration: 120 minutes (22 iterations)  
Action space: speed 0.5-4.0 m/s, steering +/- 30 deg  

## Results

### Training metrics
Completion rate: 0% (0 episodes in 440 total)  
Best progress: 68.1%  
Mean speed: 1.95 m/s (higher than V3's 1.84 m/s)  
Mean reward: 2.066  
Speed-reward correlation: +0.721 (strong - 3.3x better than V3's +0.217)  
Episode efficiency: +310% (early 30 steps -> late 121 steps - still regressing)  
Action space utilization: 10% (2/20 bins, bimodal 0.5 and 4.0 m/s persists)  
Speed-progress correlation: -0.270 (weak relationship)  

### Evaluation performance (5 trials per track)

| Track | Length | Complexity | V3 Best | V4 Best | Change | Track Speed |
|-------|--------|------------|---------|---------|--------|-------------|
| Smile | 22.2m | 0.063 | 20.70s | 21.24s | +3% | 1.05 m/s |
| re:Invent 2022 | 34.1m | 0.158 | 29.21s | 46.70s | +60% | 0.73 m/s |
| Ace Speedway | 49.1m | 0.052 | 50.59s | 39.93s | -21% | 1.23 m/s |
| Barcelona | 62.8m | 0.152 | 47.97s | 46.97s | -2% | 1.34 m/s |
| Kuei | 48.9m | 0.226 | 79.86s | 46.23s | -42% | 1.06 m/s |

Completion rate: 25/25 (100%)  
Track speed range: 0.73-1.34 m/s (wider than V3's 0.59-1.21 m/s)  

## Analysis

Speed-reward correlation +0.721 = hypothesis validated, speed-primary structure works  
BUT re:Invent regressed 60% (29.21s -> 46.70s) = catastrophic for competition  

Speed-primary success:
- Strong speed incentive (3.3x better correlation than V3)
- Higher action entropy (9.38 vs V3's ~2.5)  
- 31,871 unique actions (more exploration)

Speed-primary failure:
- Oscillatory behavior: fast->brake->fast creates time waste
- Mean speed 1.95 m/s but slower lap times than V3's 1.84 m/s
- Acceleration/deceleration overhead dominates gains
- V3's steady 1.4 m/s maintains momentum better

Track-specific pattern:
- Simple/extreme tracks (Ace, Kuei): V4 wins (-21%, -42%)  
- Medium complexity (re:Invent): V4 disaster (+60%)  
- Speed bursts help on permissive geometry, hurt on technical sections

Root cause - constraint conflicts:
- Speed-primary says "go fast"
- Progress amplifier says "must make progress to get speed rewards"  
- Centerline/steering constraints say "don't go fast off-center"
- Result: agent learns "speed up on easy, brake hard for technical"

Binary action selection persists:
- Still 10% utilization (0.5 and 4.0 m/s only)
- Oscillation between modes, not within-episode smooth transitions
- Problem is PPO/action-space fundamental, not reward structure

Episode efficiency +310%:
- Better than V3's +512% but still regression
- Agent learning to be inefficient despite strong speed incentive
- Constraints prevent speed-primary structure from working properly

Physics insight:
- Highway analogy: 55 mph constant > 70->30->70 oscillations
- Momentum conservation matters more than peak speed
- V3's conservative approach accidentally optimal for racing

Conclusion: speed-primary hypothesis proven (+0.721 correlation) but execution failed  
Constraints (centerline, steering, progress-gating) conflict with speed-primary goal  
Binary action selection is PPO limitation, not reward function issue  

Status: not viable for competition (60% slower on target track)  
but: validates that reward structure matters for speed incentive  

Next: V5 tests geometry-awareness (curvature + track width) to teach when to use which speed  
