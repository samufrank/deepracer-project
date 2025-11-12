# DeepRacer experiments summary

## Objective
Optimize reward function for time-trial racing on unknown evaluation track

## Iterations

| Model | Key Features | Training Track | re:Invent | Barcelona | Ace | Kuei | Smile | Mean Speed | Speed-Reward | Episode Eff | Action Space | Status |
|-------|-------------|----------------|-----------|-----------|-----|------|-------|------------|--------------|-------------|--------------|--------|
| Baseline | Centerline only | re:Invent 2022 | 32.09s | 62.38s | N/A | N/A | N/A | 1.74 m/s | N/A | N/A | N/A | Conservative |
| V1 | Progress + speed | Barcelona | 37.32s | 44.62s | 40.27s | N/A | N/A | 2.08 m/s | N/A | N/A | N/A | Generalization fail |
| V1b | Progress only | Barcelona | 32.83s | 51.19s | N/A | N/A | N/A | 2.11 m/s | N/A | N/A | N/A | Diagnostic success |
| V2 | Curvature + penalties | Barcelona | 60.43s | 59.90s | 42.41s | N/A | N/A | 2.38 m/s | N/A | N/A | N/A | Catastrophic |
| V2b | Curvature bonuses | Barcelona | 35.04s | 47.30s | 49.45s | N/A | N/A | 2.08 m/s | N/A | N/A | N/A | Partial recovery |
| V3 | V1b + additive speed | Barcelona | 29.21s | 47.97s | 50.59s | 79.86s | 20.70s | 1.84 m/s | +0.217 | +512% | 10% | Best medium |
| V3b | Multiplicative gradient | Barcelona | 47.44s | 45.02s | 34.51s | 42.88s | 20.57s | 2.02 m/s | +0.260 | +272% | 10% | Extreme optimization |
| V4 | Speed-primary + progress amplifier | Barcelona | 46.70s | 46.97s | 39.93s | 46.23s | 21.24s | 1.95 m/s | +0.721 | +310% | 10% | Strong incentive, poor execution |
| V5 | V3 base + geometry (curvature + width) | Barcelona | 37.12s | 49.04s | 30.49s | 42.41s | 19.16s | 1.81 m/s | -0.113 | +370% | 10% | Complexity failed |

Metrics:
- Speed-Reward: correlation between speed and reward (higher = stronger incentive)  
- Episode Eff: change in episode length early -> late training (negative = improving, positive = regressing)  
- Action Space: % of speed bins used above mean frequency (higher = more continuous)  

All models: 100% evaluation completion rate  

## Key findings

### What worked
Reliability first: all models maintained 100% eval completion despite varying training (0-0.2%)  
Progress/steps ratio: inherently rewards speed efficiency without explicit speed terms  
Medium complexity optimization: v3 best re:Invent (29.21s, 9% better than baseline)  
Systematic isolation: v1b diagnostic identified v1's blind speed multiplier as regression cause  
Cross-track evaluation: 5 diverse tracks (22-62m, complexity 0.05-0.26) revealed generalization patterns  
Speed-primary structure: v4 proved reward structure matters (+0.721 correlation, 3.3x better than v3)  

### What failed
Speed incentive weakness: additive (v3: +0.217) and multiplicative (v3b: +0.260) created weak correlation  
Speed-primary execution: v4's +0.721 correlation undermined by oscillatory behavior (60% slower re:Invent)  
Geometry-aware complexity: v5's curvature+width approach created negative correlation (-0.113)  
Binary action selection: persistent 10% utilization (only 0.5 and 4.0 m/s) across all 7 iterations  
Episode efficiency regression: all models show episodes getting longer (v3 +512%, v3b +272%, v4 +310%, v5 +370%)  
Progress dominance: v3/v3b progress terms overwhelmed speed modifiers  
Oscillatory speed: v4's rapid speed changes (0.5→4.0→0.5) waste time in acceleration/deceleration  
Geometry conditional logic: v5's 5-branch conditionals + multiplicative stacking too complex for PPO to learn  
Aggressive penalties (v2): created lose-lose scenarios, excessive crashes  
Curvature approach (v2/v2b): short lookahead (+3 waypoints) insufficient, extensive tuning needed  

### Insights
1. Training vs eval disconnect: 0% training completion with 100% eval = episode time limit artifact not policy failure, AWS uses best checkpoint
2. Track complexity metric: curvature variance (0.05-0.26) correlates with performance, enables objective difficulty quantification
3. Speed-reward correlation diagnostic: reveals reward effectiveness independent of lap times, weak (<0.3) = speed not incentivized
4. Episode efficiency as learning signal: length change shows improving (decreasing) vs regressing (increasing), more informative than completion rate
5. Binary action space pattern: persists across additive, multiplicative, speed-primary, and geometry-aware approaches across 7 iterations - likely PPO policy or action space fundamental limitation
6. Oscillatory speed behavior: v4 high mean (1.95 m/s) but poor times (46.70s) on re:Invent - rapid speed changes waste time
7. Momentum conservation: v3's steady 1.84 m/s outperforms v4's oscillating 1.95 m/s - consistent speed beats peak speed in racing
8. Track complexity tradeoff: v3 optimized medium, v5 optimized extremes - no single model wins all complexity levels
9. Checkpoint selection critical: strong eval despite poor late training = AWS finds better intermediate policies
10. Reward structure validation: v4's +0.721 proves speed-primary creates strong incentive, but constraints (centerline, steering, progress-gating) conflict with execution
11. Geometry hypothesis sound, execution broke: v5's 10-waypoint lookahead + 5 conditionals too complex, created conflicting signals

## Methodology
Systematic isolation: v1b removed single component to diagnose v1  
Multi-track evaluation: 3-5 tracks per iteration for generalization assessment  
Failure spatial analysis: identified geometry-specific crash patterns (Kuei WP46 chicane: 127 failures)  
Advanced metrics: speed-reward correlation, episode efficiency, action space utilization beyond lap times  
Iterative refinement: each iteration informed by previous failure modes  
Track complexity quantification: curvature variance enables objective comparison  
Reward structure testing: isolated speed-primary (v4) vs geometry-aware (v5) hypotheses  

## Current state: track complexity determines optimal model

Competition track unknown - complexity could be simple, medium, or complex. No single "best" model.

### Model selection by track complexity

**Simple tracks (low curvature variance <0.06):**
- **V5 optimal**: Ace 30.49s (-40% vs v3), Smile 19.16s (-7% vs v3)
- Track speed: 1.49 m/s avg (Ace), 1.26 m/s avg (Smile)
- V3 performance: Ace 50.59s (1.0 m/s avg), Smile 20.70s (1.09 m/s avg)

**Medium tracks (curvature variance 0.15-0.16):**
- **V3 optimal**: re:Invent 29.21s
- Track speed: 1.25 m/s
- V5 performance: re:Invent 37.12s (+27% vs v3)
- V4 performance: re:Invent 46.70s (+60% vs v3)

**Complex tracks (curvature variance 0.20-0.26):**
- **V5 optimal**: Kuei 42.41s (-47% vs v3)
- Track speed: 1.10 m/s (Kuei)
- V3 performance: Kuei 79.86s (0.65 m/s)

### Model characteristics

**V3 (conservative, medium-optimized):**
- Strengths: reliable medium complexity, proven re:Invent performance (29.21s)
- Weaknesses: poor extremes (Kuei 79.86s), episode efficiency +512%, weak speed-reward +0.217
- Policy: cautious (1.84 m/s), checkpoint luck (1 lap in 440 episodes)
- Risk profile: safe for medium, fails on extremes

**V5 (geometry-aware, extreme-optimized):**
- Strengths: excellent extremes (Ace -40%, Kuei -47%), better episode efficiency than v3
- Weaknesses: medium complexity worse than v3 (re:Invent +27%), negative speed-reward correlation (-0.113)
- Policy: moderate consistency (1.81 m/s), geometry logic too complex but reduced v4's oscillations
- Risk profile: wins on simple/complex, loses on medium

**V4 (speed-primary, validation experiment):**
- Strengths: proved speed-primary structure works (+0.721 correlation)
- Weaknesses: oscillatory execution catastrophic (re:Invent +60% vs v3)
- Status: hypothesis validated, not competition viable

### Competition strategy decision

Unknown track = choose based on risk tolerance:
- **Risk-averse**: v3 (safe medium, acceptable if track is re:Invent-like)
- **Risk-seeking**: v5 (2/3 complexity types win, but +27% penalty if medium)
- **Expected value**: depends on track complexity probability distribution (unknown)

Grading structure: 50 pts completion + 2-20 pts ranking + 30 pts presentation
- Completion secured (both models 100% reliable)
- Ranking depends on track complexity
- Presentation: systematic methodology across 7 iterations valuable regardless

### Shared limitations across all models

Binary action selection (10%): likely PPO/action-space fundamental based on 7-iteration pattern across diverse reward structures (additive, multiplicative, speed-primary, geometry-aware)

Episode efficiency regression: all models learn to be slower over training (v3 +512%, v3b +272%, v4 +310%, v5 +370%)

No universal solution: track complexity creates fundamental tradeoff - models optimize for narrow complexity range, cannot win all track types simultaneously

## Final insights

Bimodal distribution: 7 iterations tested (v1, v1b, v3, v3b, v4, v5, plus v2/v2b) with fundamentally different reward structures (additive bonuses, multiplicative gradients, speed-primary, geometry-aware conditionals) - all show 10% action space utilization with peaks at 0.5 and 4.0 m/s. Pattern suggests PPO policy structure or continuous action space discretization issue rather than reward function design.

Speed-primary validation: v4 achieved +0.721 speed-reward correlation (3.3x better than v3's +0.217), proving reward structure matters for creating incentives. However, constraints (centerline adherence, steering penalties, progress-gating) conflicted with speed-primary goal, causing oscillatory behavior that wasted time.

Geometry hypothesis: v5's approach (curvature + track width → speed recommendations) was theoretically sound but execution too complex. 10-waypoint lookahead + 5 conditional branches + multiplicative stacking created conflicting signals, resulting in negative speed-reward correlation (-0.113). Simpler geometry approaches might work but insufficient time for iteration.

Track complexity is fundamental: 7 iterations confirmed no reward function optimizes all complexity levels. Each model trades performance on one complexity type for another. Competition success depends on unknown track matching model's optimization range.

Cross-track evaluation critical: single-track optimization (baseline on re:Invent) failed to generalize. Multi-track testing (5 tracks, 22-62m, complexity 0.05-0.26) revealed complexity-dependent performance patterns invisible in single-track metrics.
