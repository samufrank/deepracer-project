# V5: V3 base + geometry-aware speed bonuses

Geometry hypothesis test - adds curvature and track width analysis to V3's proven structure. Goal: teach agent appropriate speeds for different track sections using explicit geometry-based rewards.

## Implementation changes
Base reward: V3's proven structure (progress + steps efficiency + additive speed)  
New: 10-waypoint lookahead curvature calculation using coordinate derivatives  
New: Track width analysis from waypoint metadata  
Geometry bonuses: 1.1x to 1.5x multipliers based on 5 conditional branches:
- Straight + wide (curv <0.05, width > 1.0): 1.5x bonus, encourage high speed
- Straight + narrow (curv <0.05, width <= 1.0): 1.3x bonus, moderate speed
- Moderate curve (0.05 ≤ curv <0.15): 1.2x bonus, cautious speed  
- Sharp curve (0.15 ≤ curv <0.25): 1.1x bonus, slow speed
- Hairpin (curv >= 0.25): 1.0x (no bonus), crawl speed
Explicit middle-speed rewards: 2.0-3.0 m/s range gets additional 1.2x bonus  
Centerline/steering: same as V3  

Rationale: geometry determines optimal speed, explicit conditionals teach when to use each speed range. Hypothesis: geometry-aware bonuses break bimodal distribution.

## Training config
Track: Barcelona (60m, 16 turns, clockwise)  
Duration: 240 minutes (43 iterations)  
Action space: speed 0.5-4.0 m/s, steering +/- 30 deg  

## Results

### Training metrics
Completion rate: 0% (0 episodes in 860 total)  
Best progress: 68.5%  
Mean speed: 1.81 m/s (lower than V3's 1.84 m/s)  
Mean reward: 0.862  
Speed-reward correlation: -0.113 (NEGATIVE - geometry bonuses backfired)  
Episode efficiency: +370% (early 33 steps -> late 154 steps - worse regression than V3)  
Action space utilization: 10% (2/20 bins, bimodal 0.5 and 4.0 m/s persists)  
Speed-progress correlation: -0.340 (learned to go slow)  

### Evaluation performance (5 trials per track)

| Track | Length | Complexity | V3 Best | V5 Best | Change | Track Speed |
|-------|--------|------------|---------|---------|--------|-------------|
| Smile | 24.1m | 0.264 | 20.70s | 19.16s | -7% | 1.26 m/s |
| re:Invent 2022 | 36.6m | 0.262 | 29.21s | 37.12s | +27% | 0.99 m/s |
| Ace Speedway | 48.6m | 0.028 | 50.59s | 30.49s | -40% | 1.59 m/s |
| Barcelona | 60.5m | 0.242 | 47.97s | 49.04s | +2% | 1.23 m/s |
| Kuei | 46.6m | 0.200 | 79.86s | 42.41s | -47% | 1.10 m/s |

Completion rate: 25/25 (100%)  
Track speed range: 0.99-1.59 m/s  

## Analysis

Worse on re:Invent (+27%), better on extremes (Ace -40%, Kuei -47%)  
Competition track unknown = can't declare V5 "better" despite winning 3/5 tracks  

Geometry-aware failure:
- Negative speed-reward correlation (-0.113) = bonuses punishing speed
- 5 conditional branches + multiplicative stacking too complex for PPO
- 10-waypoint lookahead + curvature calculation created noisy signals
- Agent couldn't learn pattern, resulted in conflicting incentives

Cross-track pattern:
- Simple (Ace 0.028): -40% vs V3, geometry logic worked on permissive track
- Medium (re:Invent 0.262): +27% vs V3, complexity hurt on technical sections  
- Complex (Kuei 0.200): -47% vs V3, aggressive policy helped on difficult geometry
- V3 optimized medium, V5 optimized extremes - fundamental tradeoff

Speed profile analysis:
- re:Invent: 1.78 m/s mean, high variance (0.5-3.5 oscillations)
- Spatial map shows NO geometry awareness - slow sections don't match sharp turns
- Agent ignored curvature logic, fell back to random exploration
- Compare to V4's oscillations: V5 more consistent but still no geometry learning

Training failure analysis:
- 849 failures across 219 waypoints (widespread, not localized)
- Top 5 failures = only 9.7% of total (V3 had concentrated patterns)
- Random distribution suggests agent exploring without learning
- Curvature+width logic created noise, not signal

Binary action selection persists (7th iteration):
- Still 10% utilization (0.5 and 4.0 m/s only)
- Explicit middle-speed rewards (2.0-3.0 m/s with 1.2x bonus) ignored
- Geometry-aware bonuses didn't break bimodal
- Confirms pattern across additive (V3), multiplicative (V3b), speed-primary (V4), geometry-aware (V5)

Episode efficiency +370%:
- Worse than V3's +512% but still catastrophic regression
- Worse than V4's +310%
- Longest episodes of any iteration = learned to be slowest
- 240 minutes training vs 120 (V3/V4) didn't help

Why geometry failed:
- Too many decision branches: if curv<0.05 AND width>1.0 then 1.5x else if...
- Multiplicative stacking: base reward × centerline × steering × geometry × middle-speed
- Reward magnitude unpredictable with 5 multipliers
- PPO can't learn 5-way conditional in 4 hours

Hypothesis was sound, execution broke:
- IEEE paper approach valid (geometry determines optimal speed)
- Implementation too complex for time budget
- Simpler geometry (fewer branches, additive not multiplicative) might work
- But insufficient time for iteration given competition deadline

Comparison to V4:
- V4: +0.721 speed-reward correlation, oscillatory execution
- V5: -0.113 speed-reward correlation, reduced oscillations vs V4
- V5 more consistent than V4 but still slower than V3 on target track
- V5 learned moderate consistency, not optimal speeds

Conclusion: geometry approach failed on medium complexity (competition-relevant)  
Explicit middle-speed rewards still ignored = bimodal likely PPO/action-space fundamental  
Better on extremes but worse on medium = inverse of V3's profile  

Status: better than V3 on 2/3 track complexity types  
but: competition track unknown, can't declare winner  
Risk assessment: V3 safe for medium, V5 gamble on extremes  

Decision: ship V3 for risk-averse (proven re:Invent), V5 for risk-seeking (2/3 win rate)
