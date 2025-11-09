# Baseline: AWS default centerline following

## Implementation
Pure centerline distance reward with three bands:  
<= 10% track width: 1.0  
<= 25% track width: 0.5  
<= 50% track width: 0.1  
Otherwise: 0.001  

No speed incentive, no progress reward  

## Training config
Track: re:Invent 2022 (35.87m, technical chicane)  
Duration: ~90 minutes (28 iterations)  
Action space: Speed 1.0-3.0 m/s, steering +/- 30 deg  

## Results

### Training metrics
Completion rate: 0%  
Best progress: 85.0%  
Mean speed: 1.74 m/s  
Right-turn bias: 51.3% vs. 39.2% left  

### Evaluation performance

| Track | Best | Average | Completion |
|-------|------|---------|------------|
| Barcelona | 62.38s | 66.12s | 5/5 |
| re:Invent 2022 | 32.09s | 38.77s | 5/5 |
| Rogue Raceway | 81.87s | 86.48s | 5/5 |

## Analysis
Conservative "slow for safety" policy. Completes tracks reliably but 2-3x slower than competitive benchmarks. Speed progress correlation of -0.340 indicates agent learned to prioritize safety over speed.
