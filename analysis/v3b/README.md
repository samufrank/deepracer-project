# V3b: Multiplicative speed gradient

Test whether multiplicative speed rewards (vs V3's additive) create stronger speed incentive and fix bimodal speed distribution. Reduced base progress rewards to accommodate multiplier range.

## Implementation changes
Base reward: Reduced to `(progress/100) * 3.0 + (progress/steps) * 2.0` (was 5.0 + 3.0 in V3)  
New: Multiplicative speed gradient `reward *= (1.0 + speed/4.0)` when on track  
- Creates 1.125x at 0.5 m/s → 2.0x at 4.0 m/s continuous incentive  
Centerline: 1.1-1.2x bonus (same as V3)  
Steering penalty: 0.8x for abs(angle) > 15 deg (same as V3)  

## Training config
Track: Barcelona (60m, 16 turns, clockwise)  
Duration: 120 minutes (29 iterations)  
Action space: speed 0.5-4.0 m/s, steering +/- 30 deg 

## Results

### Training metrics
Completion rate: 0% (no training laps completed, regressed from V3's 0.2%)  
Best progress: 49.5% (regressed from V3's 100%)  
Mean speed: 2.02 m/s (increased from V3's 1.84 m/s)  
Mean reward: 0.639 (reduced from V3's 1.322 - 52% weaker signal)  
Speed-reward correlation: +0.260 (weak, only marginally better than V3's +0.217)  
Episode efficiency: +272% (early 28 steps -> late 105 steps - getting worse)  
Action space utilization: 10% (identical to V3 - binary 0.5/4.0 m/s only)  
Speed-progress correlation: -0.337 (learned to go slow for safety)

### Evaluation performance (5 trials per track)

| Track | Length | Complexity | V3 Best | V3b Best | Change | V3 Speed | V3b Speed |
|-------|--------|------------|---------|----------|--------|----------|-----------|
| Smile | 24.6m | 0.143 | 20.70s | 20.57s | -1% | 1.09 m/s | 1.20 m/s |
| re:Invent 2022 | 36.2m | 0.187 | 29.21s | 47.44s | +62% | 1.09 m/s | 0.76 m/s |
| Ace Speedway | 52.3m | 0.096 | 50.59s | 34.51s | -32% | 1.00 m/s | 1.51 m/s |
| Barcelona | 64.6m | 0.342 | 47.97s | 45.02s | -6% | 1.21 m/s | 1.43 m/s |
| Kuei | 47.2m | 0.126 | 79.86s | 42.88s | -46% | 0.59 m/s | 1.10 m/s |

Completion rate: 25/25 (100%)  
Track speed range: 0.76-1.51 m/s across complexity spectrum  

## Analysis

Trade-off revealed: improved extremes, catastrophic on medium complexity  
Ace improved 32% (50s -> 34s), Kuei improved 46% (79s -> 42s)  
re:Invent regressed 62% (29s -> 47s) - critical failure for competition-relevant complexity

Critical failures identified:  

1. Multiplicative gradient ineffective  
   Speed-reward correlation +0.260 vs V3's +0.217 = only marginal improvement  
   Progress base (3.0 + 2.0 = 5.0) still dominates despite 2x multiplier  
   At 50% progress: 2.5 base × 2.0 = 5.0 reward (speed contributes 50%)  
   At low progress: small base × 2.0 = still small (speed impact minimal)  
   Model still doesn't associate speed with reward improvement  

2. Weaker overall learning signal  
   Mean reward 0.639 vs V3's 1.322 = 52% reduction  
   Reducing base (5+3 --> 3+2) to "make room" for multiplier backfired  
   Speed multiplier (max 2x) didn't compensate for halved base  
   Weaker rewards --> slower learning → worse checkpoint quality  
   Episode efficiency +272% (better than V3's +512% but still regression)  

3. Binary action selection persists  
   10% utilization identical to V3's pattern  
   Multiplicative gradient didn't create continuous speed usage  
   Still only uses 0.5 and 4.0 m/s extremes  
   Problem deeper than reward function (likely PPO policy or action space)  

Speed profile analysis:
- High variance (oscillations 0.5 <-> 4.0 m/s)
- re:Invent paradox: mean speed 1.90 m/s (highest) but lap time 47.44s (worst)
- Oscillation wastes time in accel/decel on technical tracks
- Works on simple geometry (Ace, Smile) but fails medium complexity

Cross-track pattern:
- V3b optimized for extremes (simple/complex), catastrophic medium (re:Invent)
- V3 optimized for medium, poor on extremes
- Competition likely medium complexity → V3 profile more valuable


Root cause: reducing base to accommodate multiplier was net negative  
Weaker base hurt more than multiplier helped  
Multiplicative vs additive made minimal difference (+0.260 vs +0.217)  

Conclusion: V3b better at extremes but worse where it matters
- Multiplicative approach only marginally better than V3's additive
- Binary action selection fundamental (persists across both approaches)
- Oscillatory speed behavior (rapid 0.5 ↔ 4.0) hurts technical tracks
- 62% regression on competition-relevant complexity is critical failure

Status: rejected for competition use  
V3 remains best model (29s re:Invent, consistent medium-complexity performance)  


Next: V4 will invert structure - make speed PRIMARY reward, progress AMPLIFIER
- V3/V3b: progress base × speed modifier = progress-dominant
- V4: speed base × progress amplifier = speed-dominant
- Target: >0.5 speed-reward correlation (ideally >0.6)
- Decision: if V4 shows similar binary selection and weak correlation, problem is PPO/action-space not reward function
