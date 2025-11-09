#!/usr/bin/env python3
"""
AWS DeepRacer Training and Evaluation Log Analyzer
Analyzes training progression, failure modes, and performance metrics.


AWS DOWNLOAD STRUCTURE:

AWS packages training and evaluation logs separately. each downloads as a tar.gz
w/ the same folder name (model hash). extract and rename each:

Training download: test-training_job_*.tar.gz
     Extracts to: 18c345ab-3c07-457a-8d0c-74160849488a/
     Contains: logs/training/, metrics/training/, sim-trace/training/

Evaluation download: test-evaluation_job_*.tar.gz  
     Extracts to: 18c345ab-3c07-457a-8d0c-74160849488a/ (same name)
     Contains: logs/evaluation/, metrics/evaluation/, sim-trace/evaluation/


RECOMMENDED WORKFLOW:

extract each download and rename to avoid conflicts:

    tar -xzf test-training_job_*.tar.gz
    mv 18c345ab-3c07-457a-8d0c-74160849488a baseline_train
    
    tar -xzf test-evaluation_job_barcelona.tar.gz
    mv 18c345ab-3c07-457a-8d0c-74160849488a baseline_eval_barcelona
    
    tar -xzf test-evaluation_job_reinvent2022.tar.gz
    mv 18c345ab-3c07-457a-8d0c-74160849488a baseline_eval_reinvent2022


DIRECTORY ORGANIZATION OPTIONS:

Option 1 - flat:
    results/
    ├── baseline_train/
    │   ├── logs/training/...
    │   ├── metrics/training/...
    │   └── sim-trace/training/training-simtrace/*.csv
    ├── baseline_eval_barcelona/
    │   ├── logs/evaluation/...
    │   ├── metrics/evaluation/...
    │   └── sim-trace/evaluation/.../evaluation-simtrace/*.csv
    └── baseline_eval_reinvent2022/
        └── sim-trace/evaluation/.../evaluation-simtrace/*.csv

    Usage:
        # Manual specification
        python deepracer_analyzer.py results/baseline_train \
            --eval-dir results/baseline_eval_barcelona \
            --eval-dir results/baseline_eval_reinvent2022 --plot
        
        # Auto-discover all evaluations
        python deepracer_analyzer.py results/baseline_train --auto-eval --plot

Option 2 - nested:
    results/
    └── baseline/
        ├── baseline_train/
        ├── baseline_eval_barcelona/
        └── baseline_eval_reinvent2022/

    Usage:
        python deepracer_analyzer.py results/baseline/baseline_train --auto-eval --plot


MORE USAGE:

# txt report only (always runs)
python deepracer_analyzer.py results/baseline_train

# add plots (saves automatically)
python deepracer_analyzer.py results/baseline_train --plot

# auto-discover all evaluations with matching model name
python deepracer_analyzer.py results/baseline_train --auto-eval --plot

# manual specification of evals
python deepracer_analyzer.py results/baseline_train \\
    --eval-dir results/baseline_eval_barcelona --plot

# display plots interactively (default: only save to file)
python deepracer_analyzer.py results/baseline_train --plot --display

# save outputs to custom directory
python deepracer_analyzer.py results/baseline_train --plot --output-dir analysis/

# compare different models
python deepracer_analyzer.py results/baseline_train results/improved_train --compare


The --auto-eval flag automatically finds evaluation folders matching the model name:
- Extracts model name from training folder (e.g., "baseline" from "baseline_train")
- Searches parent directory for folders matching "{model}_eval_*"
- Works with flat structure (Option 1) and nested structure (Option 2)

"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import glob


class Logger:
    """Tee stdout to both console and file"""
    def __init__(self, filepath=None):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w') if filepath else None
    
    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()
    
    def close(self):
        if self.log:
            self.log.close()


class DeepRacerRun:
    """Container for a single training/evaluation run"""
    def __init__(self, run_dir, eval_dirs=None):
        self.run_dir = Path(run_dir)
        self.name = self.run_dir.name
        
        # Extract clean model name (strip _train/_eval suffixes)
        run_name = self.run_dir.name
        if run_name.endswith('_train'):
            self.model_name = run_name[:-6]
        elif run_name.endswith('_eval') or '_eval_' in run_name:
            self.model_name = run_name.split('_eval')[0]
        else:
            self.model_name = run_name
        
        # Auto-discover file paths
        self.training_csv_dir = self.run_dir / 'sim-trace' / 'training' / 'training-simtrace'
        self.eval_csv_dir = self.run_dir / 'sim-trace' / 'evaluation'
        self.training_metrics = self._find_file(self.run_dir / 'metrics' / 'training', '*.json')
        self.eval_metrics = self._find_file(self.run_dir / 'metrics' / 'evaluation', '*.json')
        
        # multiple evaluation directories
        self.additional_eval_dirs = [Path(d) for d in eval_dirs] if eval_dirs else []
        
        # load data lazily
        self._training_df = None
        self._eval_df = None
        self._training_metrics_df = None
        self._eval_metrics_df = None
        self._additional_evals = {}
    
    def _find_file(self, directory, pattern):
        """Find first file matching pattern in directory"""
        if not directory.exists():
            return None
        matches = list(directory.glob(pattern))
        return matches[0] if matches else None
    
    @property
    def training_df(self):
        """Lazy load training data"""
        if self._training_df is None and self.training_csv_dir.exists():
            print(f"Loading training CSVs from {self.training_csv_dir}...")
            dfs = []
            for csv_file in sorted(self.training_csv_dir.glob('*-iteration.csv')):
                iteration = int(csv_file.stem.split('-')[0])
                df = pd.read_csv(csv_file)
                df['iteration'] = iteration
                dfs.append(df)
            self._training_df = pd.concat(dfs) if dfs else None
            if self._training_df is not None:
                print(f"  Loaded {len(self._training_df)} training steps across {self._training_df['iteration'].nunique()} iterations")
        return self._training_df
    
    @property
    def eval_df(self):
        """Lazy load evaluation data"""
        if self._eval_df is None and self.eval_csv_dir.exists():
            # find timestamped subdirectory, filtering out hidden files
            eval_subdirs = [d for d in self.eval_csv_dir.glob('*') if not d.name.startswith('.')]
            if eval_subdirs:
                eval_trace_dir = eval_subdirs[0] / 'evaluation-simtrace'
                if eval_trace_dir.exists():
                    csv_files = list(eval_trace_dir.glob('*.csv'))
                    if csv_files:
                        print(f"Loading evaluation CSV from {csv_files[0]}...")
                        self._eval_df = pd.read_csv(csv_files[0])
                        print(f"  Loaded {len(self._eval_df)} evaluation steps")
        return self._eval_df
    
    @property
    def training_metrics_df(self):
        """Lazy load training metrics JSON"""
        if self._training_metrics_df is None and self.training_metrics:
            with open(self.training_metrics) as f:
                data = json.load(f)
            self._training_metrics_df = pd.DataFrame(data['metrics'])
        return self._training_metrics_df
    
    @property
    def eval_metrics_df(self):
        """Lazy load evaluation metrics JSON"""
        if self._eval_metrics_df is None and self.eval_metrics:
            with open(self.eval_metrics) as f:
                data = json.load(f)
            self._eval_metrics_df = pd.DataFrame(data['metrics'])
        return self._eval_metrics_df
    
    def get_additional_eval(self, eval_dir):
        """Load evaluation data from additional evaluation directory"""
        eval_path = Path(eval_dir)
        if eval_path in self._additional_evals:
            return self._additional_evals[eval_path]
        
        eval_data = None
        # look for eval .csv in the directory structure
        eval_csv_dir = eval_path / 'sim-trace' / 'evaluation'
        
        if eval_csv_dir.exists():
            # Filter out hidden files (like .DS_Store on macOS)
            eval_subdirs = [d for d in eval_csv_dir.glob('*') if not d.name.startswith('.')]
            if eval_subdirs:
                eval_trace_dir = eval_subdirs[0] / 'evaluation-simtrace'
                if eval_trace_dir.exists():
                    csv_files = list(eval_trace_dir.glob('*.csv'))
                    if csv_files:
                        print(f"Loading evaluation CSV from {csv_files[0]}...")
                        eval_data = pd.read_csv(csv_files[0])
                        print(f"  Loaded {len(eval_data)} evaluation steps")
        
        self._additional_evals[eval_path] = eval_data
        return eval_data
    
    def get_all_evaluations(self):
        """Return dict of all evaluation dataframes with track names"""
        evals = {}
        
        # primary evaluation (from training directory)
        if self.eval_df is not None:
            evals['primary'] = self.eval_df
        
        # additional evaluations
        for eval_dir in self.additional_eval_dirs:
            eval_data = self.get_additional_eval(eval_dir)
            if eval_data is not None:
                track_name = eval_dir.name
                evals[track_name] = eval_data
        
        return evals


def estimate_track_length(df):
    """
    Estimate track length from waypoint X,Y coordinates.
    Uses centerline waypoints to calculate total track distance.
    
    Args:
        df: DataFrame with 'closest_waypoint', 'X', 'Y' columns
    
    Returns:
        float: Estimated track length in meters
    """
    # Get average X,Y position for each waypoint (centerline)
    waypoint_coords = df.groupby('closest_waypoint').agg({
        'X': 'mean',
        'Y': 'mean'
    }).sort_index()
    
    if len(waypoint_coords) < 2:
        return None
    
    # Calculate Euclidean distance between consecutive waypoints
    distances = np.sqrt(
        np.diff(waypoint_coords['X'])**2 + 
        np.diff(waypoint_coords['Y'])**2
    )
    
    # Add closing distance (last waypoint back to first - closed loop)
    closing_distance = np.sqrt(
        (waypoint_coords['X'].iloc[-1] - waypoint_coords['X'].iloc[0])**2 +
        (waypoint_coords['Y'].iloc[-1] - waypoint_coords['Y'].iloc[0])**2
    )
    
    total_length = distances.sum() + closing_distance
    return total_length


def calculate_track_complexity(df):
    """
    Calculate track complexity metric based on curvature variance.
    Higher values indicate more complex tracks with tighter/variable turns.
    
    Args:
        df: DataFrame with 'closest_waypoint', 'X', 'Y' columns
    
    Returns:
        float: Track complexity score (curvature variance)
    """
    # Get average X,Y position for each waypoint (centerline)
    waypoint_coords = df.groupby('closest_waypoint').agg({
        'X': 'mean',
        'Y': 'mean'
    }).sort_index()
    
    if len(waypoint_coords) < 3:
        return None
    
    # Calculate direction angles between consecutive waypoints
    dx = np.diff(waypoint_coords['X'])
    dy = np.diff(waypoint_coords['Y'])
    angles = np.arctan2(dy, dx)
    
    # Calculate curvature as change in direction angle
    # Wrap angle differences to [-pi, pi]
    angle_diffs = np.diff(angles)
    angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
    
    # Curvature variance as complexity metric
    # Low variance = consistent curves (simple)
    # High variance = varying curvature (complex)
    complexity = np.var(angle_diffs)
    
    return complexity


def analyze_failure_points(df):
    """Find where car crashes most often"""
    failures = df[df['episode_status'] == 'off_track']
    if len(failures) == 0:
        return pd.Series(dtype=int), None
    
    failure_waypoints = failures.groupby('closest_waypoint').size()
    
    # get XY coordinates for waypoint visualization
    waypoint_coords = failures.groupby('closest_waypoint').agg({
        'X': 'mean',
        'Y': 'mean'
    })
    
    return failure_waypoints.sort_values(ascending=False), waypoint_coords


def analyze_action_distribution(df):
    """Check if using full action space"""
    action_counts = df['action'].value_counts()
    # calculate entropy
    probs = action_counts / len(df)
    action_entropy = -np.sum(probs * np.log(probs + 1e-10))
    return action_counts, action_entropy


def plot_track_with_failures(df, output_path=None):
    """Plot track layout with failure heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # plot track centerline using all waypoints
    track_points = df.groupby('closest_waypoint').agg({
        'X': 'mean',
        'Y': 'mean'
    }).sort_index()
    
    ax.plot(track_points['X'], track_points['Y'], 'k-', linewidth=2, 
            alpha=0.3, label='Track centerline')
    
    # plot failure locations
    failures = df[df['episode_status'] == 'off_track']
    if len(failures) > 0:
        failure_density = failures.groupby('closest_waypoint').size()
        failure_coords = failures.groupby('closest_waypoint').agg({
            'X': 'mean',
            'Y': 'mean'
        })
        
        # Normalize for color mapping
        max_failures = failure_density.max()
        
        scatter = ax.scatter(failure_coords['X'], failure_coords['Y'], 
                           c=failure_density.values, 
                           s=failure_density.values * 20,
                           cmap='Reds', alpha=0.6, edgecolors='darkred',
                           vmin=0, vmax=max_failures)
        
        # label top failure waypoints
        top_failures = failure_density.nlargest(5)
        for wp in top_failures.index:
            if wp in failure_coords.index:
                ax.annotate(f'WP{int(wp)}\n({int(top_failures[wp])})', 
                          xy=(failure_coords.loc[wp, 'X'], failure_coords.loc[wp, 'Y']),
                          xytext=(10, 10), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                          fontsize=8, ha='left')
        
        plt.colorbar(scatter, ax=ax, label='Number of failures')
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Track Layout with Failure Locations', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_progression(run, output_dir=None, display=True):
    """Visualize learning over time"""
    df = run.training_df
    if df is None:
        print("No training data available")
        return
    
    # Create main analysis figure - EXPANDED to 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f'Training Analysis: {run.model_name}', fontsize=16)
    
    # Progress over iterations
    iter_progress = df.groupby('iteration')['progress'].max()
    axes[0, 0].plot(iter_progress.index, iter_progress.values, 'o-', alpha=0.5, label='Raw progress')
    
    # Add rolling average if enough iterations
    if len(iter_progress) >= 5:
        rolling_avg = iter_progress.rolling(window=5, center=True, min_periods=3).mean()
        axes[0, 0].plot(rolling_avg.index, rolling_avg.values, '-', linewidth=3, 
                       color='blue', label='5-iter avg')
    
    axes[0, 0].axhline(y=100, color='r', linestyle='--', alpha=0.3, label='Complete lap')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Max Progress (%)')
    axes[0, 0].set_title('Best Progress Per Iteration')
    axes[0, 0].legend(loc='lower right', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Completion visualization - HYBRID: bars per iteration + cumulative line
    episode_progress = df.groupby(['iteration', 'episode'])['progress'].max()
    complete_episodes = episode_progress.groupby('iteration').apply(
        lambda x: (x == 100).sum()  # Count of completed laps per iteration
    )
    cumulative_completions = complete_episodes.cumsum()
    total_completions = int(cumulative_completions.iloc[-1]) if len(cumulative_completions) > 0 else 0
    
    # Primary axis: bars showing completions per iteration
    axes[0, 1].bar(complete_episodes.index, complete_episodes.values, 
                   color='green', alpha=0.6, edgecolor='darkgreen', width=0.8,
                   label='Per iteration')
    
    # Overlay cumulative line if meaningful (>3 total completions)
    if total_completions > 3:
        ax_twin = axes[0, 1].twinx()
        ax_twin.plot(cumulative_completions.index, cumulative_completions.values, '-',
                    linewidth=3, color='darkblue', marker='o', markersize=4,
                    label='Cumulative', alpha=0.8)
        ax_twin.set_ylabel('Cumulative Total', fontsize=9, color='darkblue')
        ax_twin.tick_params(axis='y', labelcolor='darkblue', labelsize=8)
        ax_twin.set_ylim([0, max(cumulative_completions.values) * 1.15])
        ax_twin.legend(loc='upper left', fontsize=8)
    
    # Add total completion count annotation
    axes[0, 1].text(0.98, 0.98, f'Total: {total_completions} laps',
                   transform=axes[0, 1].transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                   fontsize=10, ha='right', va='top', weight='bold')
    
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Laps Completed', fontsize=10)
    axes[0, 1].set_title('Track Completions Per Iteration')
    axes[0, 1].legend(loc='upper left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    # Auto-scale with minimum visibility
    max_per_iter = complete_episodes.max() if len(complete_episodes) > 0 else 1
    axes[0, 1].set_ylim([0, max(3, max_per_iter * 1.2)])
    axes[0, 1].set_ylim([0, max(cumulative_completions.values) * 1.1])
    
    # Row 1, left: failure heatmap by waypoint
    failures = df[df['episode_status'] == 'off_track']
    if len(failures) > 0:
        failure_waypoints = failures['closest_waypoint'].value_counts().sort_index()
        axes[1, 0].bar(failure_waypoints.index, failure_waypoints.values, 
                      color='red', alpha=0.6, width=1.0)
        axes[1, 0].set_xlabel('Waypoint')
        axes[1, 0].set_ylabel('Failure Count')
        axes[1, 0].set_title('Failure Locations (by Waypoint)')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 0].text(0.5, 0.5, 'No failures recorded', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Row 1, right: Episode length trends (MOVED from row 3)
    episode_lengths = df.groupby(['iteration', 'episode'])['steps'].max()
    mean_length = episode_lengths.groupby('iteration').mean()
    
    axes[1, 1].plot(mean_length.index, mean_length.values, 'o-', alpha=0.5, 
                   color='purple', label='Raw mean')
    
    # Add rolling average if enough iterations
    if len(mean_length) >= 5:
        rolling_avg = mean_length.rolling(window=5, center=True, min_periods=3).mean()
        axes[1, 1].plot(rolling_avg.index, rolling_avg.values, '-', linewidth=3,
                       color='indigo', label='5-iter avg')
    
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Mean Episode Length (steps)')
    axes[1, 1].set_title('Episode Efficiency Over Training')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add interpretation text
    if len(mean_length) >= 10:
        early_mean = mean_length.iloc[:5].mean()
        late_mean = mean_length.iloc[-5:].mean()
        change_pct = ((late_mean - early_mean) / early_mean * 100) if early_mean > 0 else 0
        trend_text = "Shorter" if change_pct < 0 else "Longer"
        axes[1, 1].text(0.02, 0.98, f'{trend_text} episodes:\n{abs(change_pct):.0f}% change',
                       transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round,pad=0.5', 
                                fc='lightgreen' if change_pct < 0 else 'lightyellow', alpha=0.7),
                       fontsize=9, ha='left', va='top')
    
    # Row 2: speed distribution spanning 2 columns
    # Create subplot that spans the full width
    gs = axes[2, 0].get_gridspec()
    # Remove the individual subplots
    axes[2, 0].remove()
    axes[2, 1].remove()
    # Add single subplot spanning row 2
    ax_speed = fig.add_subplot(gs[2, :])
    
    # Speed distribution with enhanced annotations (spanning row 2)
    ax_speed.hist(df['throttle'], bins=30, color='blue', alpha=0.6, edgecolor='black')
    
    # Add statistics
    mean_speed = df['throttle'].mean()
    median_speed = df['throttle'].median()
    std_speed = df['throttle'].std()
    
    # Add vertical lines for mean/median
    ax_speed.axvline(mean_speed, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_speed:.2f}')
    ax_speed.axvline(median_speed, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_speed:.2f}')
    
    # Add action space bounds (0.5 to 4.0 m/s)
    ax_speed.axvline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Action bounds')
    ax_speed.axvline(4.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Detect and annotate bimodal peaks if present
    # Check if there's clustering at low (0.5-1.0) and high (3.5-4.0) speeds
    low_speed_count = ((df['throttle'] >= 0.5) & (df['throttle'] <= 1.0)).sum()
    high_speed_count = ((df['throttle'] >= 3.5) & (df['throttle'] <= 4.0)).sum()
    total_count = len(df['throttle'])
    
    if low_speed_count > 0.15 * total_count and high_speed_count > 0.15 * total_count:
        # Bimodal distribution detected - position at bottom-right to avoid overlap
        ax_speed.annotate('Bimodal:\nLow/High\nspeed peaks', 
                           xy=(0.98, 0.05), xycoords='axes fraction',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           fontsize=9, ha='right', va='bottom')
    
    # Add statistics text box - moved to bottom left to avoid bar overlap
    stats_text = f'μ={mean_speed:.2f}\nσ={std_speed:.2f}'
    ax_speed.text(0.02, 0.40, stats_text,
                   transform=ax_speed.transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                   fontsize=9, ha='left', va='top')
    
    ax_speed.set_xlabel('Throttle (speed)')
    ax_speed.set_ylabel('Frequency')
    ax_speed.set_title('Speed Distribution')
    ax_speed.legend(loc='upper center', fontsize=8, ncol=2)  # Changed to upper center with 2 columns
    ax_speed.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{run.model_name}_training_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training analysis plot saved to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)
    
    # create separate track visualization
    track_output = Path(output_dir) / f"{run.model_name}_track_failures.png" if output_dir else None
    track_fig = plot_track_with_failures(df, track_output)
    if output_dir:
        print(f"Track failure map saved to {Path(output_dir) / f'{run.model_name}_track_failures.png'}")
    
    if display:
        plt.show()
    else:
        plt.close(track_fig)
    
    # NEW: Create action space utilization heatmap
    print("Generating action space utilization heatmap...")
    action_fig = plot_action_space_heatmap(df, run.model_name, output_dir, display)
    
    # NEW: Create reward distribution analysis
    print("Generating reward distribution analysis...")
    reward_fig = plot_reward_distribution(df, run.model_name, output_dir, display)


def plot_action_space_heatmap(df, model_name, output_dir=None, display=True):
    """Visualize 2D action space utilization: speed × steering"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Action Space Utilization: {model_name}', fontsize=16)
    
    # 2D histogram: speed vs steering - improved colormap
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create custom colormap: white for zero, then blue->red gradient
    colors = ['white', 'lightblue', 'yellow', 'orange', 'red', 'darkred']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('action_usage', colors, N=n_bins)
    
    h = ax1.hist2d(df['throttle'], df['steer'], bins=[20, 20], 
                   cmap=cmap, cmin=1)  # cmin=1 shows white for unused combos
    cbar = plt.colorbar(h[3], ax=ax1, label='Frequency')
    ax1.set_xlabel('Speed (m/s)')
    ax1.set_ylabel('Steering Angle (°)')
    ax1.set_title('Speed-Steering Combinations Used')
    ax1.grid(True, alpha=0.3)
    
    # Add action space boundaries
    ax1.axvline(0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Action bounds')
    ax1.axvline(4.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.axhline(-30, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.axhline(30, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Marginal distributions
    # Speed marginal with IMPROVED utilization metric
    speed_bins = np.linspace(0.5, 4.0, 21)
    speed_hist, _ = np.histogram(df['throttle'], bins=speed_bins)
    
    # Calculate effective utilization (bins with above-mean frequency)
    # This better captures bimodal distributions than % of peak
    mean_freq = speed_hist.mean()
    effective_bins = (speed_hist > mean_freq).sum()
    effective_utilization = effective_bins / len(speed_hist) * 100
    
    # Also show raw bin utilization
    any_usage_bins = (speed_hist > 0).sum()
    raw_utilization = any_usage_bins / len(speed_hist) * 100
    
    ax2.barh(range(len(speed_hist)), speed_hist, color='blue', alpha=0.6, edgecolor='black')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Speed Bin')
    ax2.set_title(f'Speed Range Utilization')
    ax2.set_yticks(range(0, len(speed_hist), 5))
    ax2.set_yticklabels([f'{speed_bins[i]:.1f}' for i in range(0, len(speed_hist), 5)])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add improved text annotation
    util_text = f'Effective: {effective_utilization:.0f}%\n({effective_bins}/{len(speed_hist)} bins >mean)\n\nAny usage: {raw_utilization:.0f}%'
    ax2.text(0.95, 0.95, util_text,
             transform=ax2.transAxes, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
             fontsize=9, ha='right', va='top')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{model_name}_action_space.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Action space heatmap saved to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_reward_distribution(df, model_name, output_dir=None, display=True):
    """Analyze reward distribution and evolution"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Reward Analysis: {model_name}', fontsize=16)
    
    # Panel 1: Reward distribution histogram
    axes[0, 0].hist(df['reward'], bins=50, color='purple', alpha=0.6, edgecolor='black')
    mean_reward = df['reward'].mean()
    median_reward = df['reward'].median()
    axes[0, 0].axvline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
    axes[0, 0].axvline(median_reward, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_reward:.2f}')
    axes[0, 0].set_xlabel('Reward Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Reward evolution over iterations
    iter_rewards = df.groupby('iteration')['reward'].mean()
    axes[0, 1].plot(iter_rewards.index, iter_rewards.values, 'o-', alpha=0.5, label='Raw mean')
    
    # Add rolling average
    if len(iter_rewards) >= 5:
        rolling_avg = iter_rewards.rolling(window=5, center=True, min_periods=3).mean()
        axes[0, 1].plot(rolling_avg.index, rolling_avg.values, '-', linewidth=3,
                    color='purple', label='5-iter avg')
    
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Mean Reward')
    axes[0, 1].set_title('Reward Evolution During Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 3 (bottom-left): Reward vs Progress correlation
    episode_data = df.groupby(['iteration', 'episode']).agg({
        'reward': 'sum',
        'progress': 'max'
    })
    
    axes[1, 0].scatter(episode_data['progress'], episode_data['reward'], 
                   alpha=0.3, s=10, c='purple')
    
    # Add trend line
    z = np.polyfit(episode_data['progress'], episode_data['reward'], 1)
    p = np.poly1d(z)
    progress_range = np.linspace(episode_data['progress'].min(), episode_data['progress'].max(), 100)
    axes[1, 0].plot(progress_range, p(progress_range), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    # Calculate correlation
    correlation = episode_data['progress'].corr(episode_data['reward'])
    axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:+.3f}',
                transform=axes[1, 0].transAxes, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                fontsize=10, ha='left', va='top')
    
    axes[1, 0].set_xlabel('Episode Progress (%)')
    axes[1, 0].set_ylabel('Episode Total Reward')
    axes[1, 0].set_title('Reward-Progress Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel 4 (bottom-right): NEW - Reward vs Speed
    axes[1, 1].scatter(df['throttle'], df['reward'], alpha=0.2, s=5, c='green')
    
    # Add mean reward per speed bin
    speed_bins = np.linspace(0.5, 4.0, 15)
    bin_indices = np.digitize(df['throttle'], speed_bins)
    mean_rewards = [df[bin_indices == i]['reward'].mean() if (bin_indices == i).sum() > 0 else np.nan 
                    for i in range(1, len(speed_bins))]
    bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2
    
    # Remove NaN values
    valid_mask = ~np.isnan(mean_rewards)
    valid_centers = bin_centers[valid_mask]
    valid_means = np.array(mean_rewards)[valid_mask]
    
    if len(valid_means) > 0:
        axes[1, 1].plot(valid_centers, valid_means, 'r-', linewidth=3, 
                       marker='o', markersize=6, label='Mean reward per speed')
        axes[1, 1].legend()
        
        # Calculate speed-reward correlation
        speed_reward_corr = df['throttle'].corr(df['reward'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {speed_reward_corr:+.3f}',
                       transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       fontsize=10, ha='left', va='top')
    
    axes[1, 1].set_xlabel('Speed (m/s)')
    axes[1, 1].set_ylabel('Step Reward')
    axes[1, 1].set_title('Reward vs Speed')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{model_name}_reward_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Reward analysis saved to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)
    
    return fig




def plot_evaluation_detailed(run, track_name, eval_df, output_dir=None, display=True):
    """Detailed visualization for single evaluation track"""
    if eval_df is None or len(eval_df) == 0:
        print(f"No evaluation data available for {track_name}")
        return
    
    print(f"Generating detailed evaluation plot for {track_name}...")
    
    # Create main analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Evaluation Analysis: {run.model_name} - {track_name}', fontsize=16)
    
    # Episode progress distribution
    episode_stats = eval_df.groupby('episode').agg({
        'progress': 'max',
        'steps': 'count',
        'episode_status': 'first'
    })
    
    axes[0, 0].bar(episode_stats.index, episode_stats['progress'], 
                   color=['green' if p == 100 else 'orange' for p in episode_stats['progress']],
                   alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Complete lap')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Progress (%)')
    axes[0, 0].set_title('Progress Per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 105])
    
    # Lap time distribution (for completed laps)
    completed_episodes = episode_stats[episode_stats['progress'] == 100]
    if len(completed_episodes) > 0:
        lap_times = completed_episodes['steps'] * 0.067  # convert steps to seconds
        axes[0, 1].hist(lap_times, bins=min(10, len(lap_times)), 
                       color='green', alpha=0.6, edgecolor='black')
        axes[0, 1].axvline(lap_times.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {lap_times.mean():.2f}s')
        axes[0, 1].axvline(lap_times.min(), color='blue', linestyle='--', 
                          linewidth=2, label=f'Best: {lap_times.min():.2f}s')
        axes[0, 1].set_xlabel('Lap Time (s)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Lap Time Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[0, 1].text(0.5, 0.5, 'No completed laps', 
                       ha='center', va='center', transform=axes[0, 1].transAxes,
                       fontsize=14)
        axes[0, 1].set_title('Lap Time Distribution')
    
    # Failure heatmap by waypoint
    failures = eval_df[eval_df['episode_status'] == 'off_track']
    if len(failures) > 0:
        failure_waypoints = failures['closest_waypoint'].value_counts().sort_index()
        axes[1, 0].bar(failure_waypoints.index, failure_waypoints.values,
                      color='red', alpha=0.6, width=1.0)
        axes[1, 0].set_xlabel('Waypoint')
        axes[1, 0].set_ylabel('Failure Count')
        axes[1, 0].set_title('Failure Locations (by Waypoint)')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 0].text(0.5, 0.5, 'No failures recorded',
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=14, color='green', weight='bold')
        axes[1, 0].set_title('Failure Locations (by Waypoint)')
    
    # Speed distribution
    axes[1, 1].hist(eval_df['throttle'], bins=30, color='blue', alpha=0.6, edgecolor='black')
    axes[1, 1].set_xlabel('Throttle (speed)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Speed Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{run.model_name}_eval_detailed.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Detailed evaluation plot saved to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_evaluation_comparison(run, all_evals, output_dir=None, display=True):
    """Multi-track comparison visualization"""
    if len(all_evals) == 0:
        print("No evaluation data available for comparison")
        return
    
    print(f"Generating multi-track comparison plot ({len(all_evals)} tracks)...")
    
    # Prepare data for all tracks
    track_names = []
    track_labels = []  # With length info
    track_lengths = []  # Store lengths for speed calculation
    completion_rates = []
    best_lap_times = []
    avg_lap_times = []
    lap_time_data = []  # for box plot
    best_speeds = []  # NEW: track speeds
    avg_speeds = []   # NEW: track speeds
    
    for track_name, eval_df in all_evals.items():
        episode_stats = eval_df.groupby('episode').agg({
            'progress': 'max',
            'steps': 'count'
        })
        
        track_names.append(track_name)
        
        # Clean track name - strip model_eval prefix
        clean_name = track_name
        if clean_name.startswith(f"{run.model_name}_eval_"):
            clean_name = clean_name[len(f"{run.model_name}_eval_"):]
        
        # Calculate track length for label
        track_length = estimate_track_length(eval_df)
        track_lengths.append(track_length)
        if track_length:
            track_labels.append(f"{clean_name}\n({track_length:.0f}m)")
        else:
            track_labels.append(clean_name)
        
        # Completion rate
        n_complete = (episode_stats['progress'] == 100).sum()
        n_total = len(episode_stats)
        completion_rates.append(n_complete / n_total * 100)
        
        # Lap times and track speeds
        completed = episode_stats[episode_stats['progress'] == 100]
        if len(completed) > 0 and track_length:
            lap_times = completed['steps'] * 0.067
            best_lap_times.append(lap_times.min())
            avg_lap_times.append(lap_times.mean())
            lap_time_data.append(lap_times.values)
            # Calculate track speeds (m/s)
            best_speeds.append(track_length / lap_times.min())
            avg_speeds.append(track_length / lap_times.mean())
        else:
            best_lap_times.append(None)
            avg_lap_times.append(None)
            lap_time_data.append([])
            best_speeds.append(None)
            avg_speeds.append(None)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Multi-Track Evaluation Comparison: {run.model_name}', fontsize=16)
    
    x_pos = np.arange(len(track_names))
    
    # Panel 1: Completion rates
    bars1 = axes[0].bar(x_pos, completion_rates, color='green', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Track')
    axes[0].set_ylabel('Completion Rate (%)')
    axes[0].set_title('Track Completion Rates')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(track_labels, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylim([0, 115])  # Increased from 105 to prevent label overlap
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars (positioned inside if at 100%)
    for bar, val in zip(bars1, completion_rates):
        height = bar.get_height()
        if val >= 100:
            # Place label inside bar for 100% to avoid overlap
            axes[0].text(bar.get_x() + bar.get_width()/2., height - 5,
                        f'{val:.0f}%', ha='center', va='top', fontsize=9, color='white', weight='bold')
        else:
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Panel 2: Track Speeds (CHANGED from lap times)
    valid_best_speeds = [s for s in best_speeds if s is not None]
    valid_avg_speeds = [s for s in avg_speeds if s is not None]
    valid_track_indices = [i for i, s in enumerate(best_speeds) if s is not None]
    
    if len(valid_best_speeds) > 0:
        x_valid = np.array(valid_track_indices)
        width = 0.35
        
        bars2 = axes[1].bar(x_valid - width/2, valid_best_speeds, width, 
                           label='Best', color='blue', alpha=0.7, edgecolor='black')
        bars3 = axes[1].bar(x_valid + width/2, valid_avg_speeds, width,
                           label='Average', color='orange', alpha=0.7, edgecolor='black')
        
        axes[1].set_xlabel('Track')
        axes[1].set_ylabel('Track Speed (m/s)')
        axes[1].set_title('Track Speed (Completed Tracks Only)')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(track_labels, rotation=45, ha='right', fontsize=9)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Set ylim with headroom to prevent label overlap
        max_speed = max(max(valid_best_speeds), max(valid_avg_speeds))
        axes[1].set_ylim([0, max_speed * 1.15])  # 15% headroom
        
        # Add value labels - position inside bar if near top
        for bar, val in zip(bars2, valid_best_speeds):
            height = bar.get_height()
            if height > max_speed * 0.9:  # If bar is >90% of max
                axes[1].text(bar.get_x() + bar.get_width()/2., height - 0.05,
                            f'{val:.2f}', ha='center', va='top', fontsize=8, color='white', weight='bold')
            else:
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.03,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars3, valid_avg_speeds):
            height = bar.get_height()
            if height > max_speed * 0.9:
                axes[1].text(bar.get_x() + bar.get_width()/2., height - 0.05,
                            f'{val:.2f}', ha='center', va='top', fontsize=8, color='white', weight='bold')
            else:
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.03,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    else:
        axes[1].text(0.5, 0.5, 'No completed laps across all tracks',
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=12)
        axes[1].set_title('Track Speed (Completed Tracks Only)')
    
    # Panel 3: Lap time distributions (box plot)
    valid_lap_data = [data for data in lap_time_data if len(data) > 0]
    valid_labels_with_length = [track_labels[i] for i, data in enumerate(lap_time_data) if len(data) > 0]
    
    if len(valid_lap_data) > 0:
        bp = axes[2].boxplot(valid_lap_data, tick_labels=valid_labels_with_length, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        axes[2].set_xlabel('Track')
        axes[2].set_ylabel('Lap Time (s)')
        axes[2].set_title('Lap Time Consistency')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Adjust x-axis labels
        axes[2].set_xticklabels(valid_labels_with_length, rotation=45, ha='right', fontsize=9)
    else:
        axes[2].text(0.5, 0.5, 'No completed laps for distribution analysis',
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=12)
        axes[2].set_title('Lap Time Consistency')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{run.model_name}_eval_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Multi-track comparison plot saved to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_speed_profile(run, track_name, eval_df, output_dir=None, display=True):
    """Plot speed vs waypoint progression around the track"""
    if eval_df is None or len(eval_df) == 0:
        print(f"  No evaluation data available for speed profile on {track_name}")
        return
    
    # Only use completed laps for clean speed profile
    completed_episodes = eval_df.groupby('episode')['progress'].max()
    completed_episode_ids = completed_episodes[completed_episodes == 100].index
    
    if len(completed_episode_ids) == 0:
        print(f"  No completed laps on {track_name} - skipping speed profile")
        return
    
    print(f"  Generating speed profile for {track_name}...")
    
    # Filter to completed laps only
    completed_laps = eval_df[eval_df['episode'].isin(completed_episode_ids)]
    
    # Calculate average speed at each waypoint
    waypoint_speeds = completed_laps.groupby('closest_waypoint').agg({
        'throttle': ['mean', 'std', 'min', 'max'],
        'X': 'mean',
        'Y': 'mean'
    }).sort_index()
    
    waypoint_speeds.columns = ['speed_mean', 'speed_std', 'speed_min', 'speed_max', 'X', 'Y']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Speed Profile Analysis: {run.model_name} - {track_name}', fontsize=16)
    
    # Panel 1: Speed vs Waypoint (line plot)
    waypoints = waypoint_speeds.index.values
    ax1.plot(waypoints, waypoint_speeds['speed_mean'], 'b-', linewidth=2, label='Mean speed')
    ax1.fill_between(waypoints, 
                      waypoint_speeds['speed_mean'] - waypoint_speeds['speed_std'],
                      waypoint_speeds['speed_mean'] + waypoint_speeds['speed_std'],
                      alpha=0.3, color='blue', label='±1 std dev')
    ax1.plot(waypoints, waypoint_speeds['speed_max'], 'g--', linewidth=1, alpha=0.5, label='Max observed')
    ax1.plot(waypoints, waypoint_speeds['speed_min'], 'r--', linewidth=1, alpha=0.5, label='Min observed')
    
    ax1.axhline(y=waypoint_speeds['speed_mean'].mean(), color='gray', linestyle=':', 
                linewidth=2, label=f"Track avg: {waypoint_speeds['speed_mean'].mean():.2f} m/s")
    
    ax1.set_xlabel('Waypoint')
    ax1.set_ylabel('Speed (m/s)')
    ax1.set_title('Speed vs Track Position')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([waypoints.min(), waypoints.max()])
    
    # Panel 2: Spatial speed map (track with speed coloring)
    scatter = ax2.scatter(waypoint_speeds['X'], waypoint_speeds['Y'],
                         c=waypoint_speeds['speed_mean'], s=100,
                         cmap='RdYlGn', alpha=0.8, edgecolors='black', linewidth=0.5,
                         vmin=waypoint_speeds['speed_mean'].min(),
                         vmax=waypoint_speeds['speed_mean'].max())
    
    # Draw track centerline
    ax2.plot(waypoint_speeds['X'], waypoint_speeds['Y'], 'k-', 
             linewidth=1, alpha=0.3, zorder=0)
    
    # Mark slow sections (bottom 20%)
    speed_threshold = waypoint_speeds['speed_mean'].quantile(0.20)
    slow_sections = waypoint_speeds[waypoint_speeds['speed_mean'] <= speed_threshold]
    if len(slow_sections) > 0:
        ax2.scatter(slow_sections['X'], slow_sections['Y'], 
                   s=200, facecolors='none', edgecolors='red', 
                   linewidth=2, label=f'Slowest 20% (<{speed_threshold:.2f} m/s)', zorder=3)
    
    plt.colorbar(scatter, ax=ax2, label='Mean Speed (m/s)')
    ax2.set_xlabel('X position (m)')
    ax2.set_ylabel('Y position (m)')
    ax2.set_title('Spatial Speed Distribution')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_dir:
        safe_track_name = track_name.replace(' ', '_').replace('/', '_')
        if safe_track_name.startswith(f"{run.model_name}_eval_"):
            safe_track_name = safe_track_name[len(f"{run.model_name}_eval_"):]
        output_path = Path(output_dir) / f"{run.model_name}_eval_{safe_track_name}_speed_profile.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Speed profile saved to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_evaluation_track_failures(run, track_name, eval_df, output_dir=None, display=True):
    """Plot spatial failure map for evaluation track"""
    failures = eval_df[eval_df['episode_status'] == 'off_track']
    
    if len(failures) == 0:
        print(f"  No failures on {track_name} - skipping failure map")
        return
    
    print(f"  Generating failure map for {track_name}...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot track centerline
    track_points = eval_df.groupby('closest_waypoint').agg({
        'X': 'mean',
        'Y': 'mean'
    }).sort_index()
    
    ax.plot(track_points['X'], track_points['Y'], 'k-', linewidth=2,
            alpha=0.3, label='Track centerline')
    
    # Plot failure locations
    failure_density = failures.groupby('closest_waypoint').size()
    failure_coords = failures.groupby('closest_waypoint').agg({
        'X': 'mean',
        'Y': 'mean'
    })
    
    max_failures = failure_density.max()
    
    scatter = ax.scatter(failure_coords['X'], failure_coords['Y'],
                        c=failure_density.values,
                        s=failure_density.values * 20,
                        cmap='Reds', alpha=0.6, edgecolors='darkred',
                        vmin=0, vmax=max_failures)
    
    # Label top failure waypoints
    top_failures = failure_density.nlargest(5)
    for wp in top_failures.index:
        if wp in failure_coords.index:
            ax.annotate(f'WP{int(wp)}\n({int(top_failures[wp])})',
                       xy=(failure_coords.loc[wp, 'X'], failure_coords.loc[wp, 'Y']),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       fontsize=8, ha='left')
    
    plt.colorbar(scatter, ax=ax, label='Number of failures')
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title(f'Evaluation Failure Map: {track_name}', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if output_dir:
        # Clean track name for filename - strip model prefix if present
        safe_track_name = track_name.replace(' ', '_').replace('/', '_')
        # Remove redundant model_name prefix from track_name (e.g., "v3b_eval_barcelona" -> "barcelona")
        if safe_track_name.startswith(f"{run.model_name}_eval_"):
            safe_track_name = safe_track_name[len(f"{run.model_name}_eval_"):]
        output_path = Path(output_dir) / f"{run.model_name}_eval_{safe_track_name}_failures.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Evaluation failure map saved to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)


def generate_report(run):
    """Generate text summary"""
    print("\n" + "=" * 70)
    print(f"DEEPRACER ANALYSIS REPORT: {run.model_name}")
    print("=" * 70)
    
    # training analysis
    df = run.training_df
    if df is not None:
        print("\nTRAINING SUMMARY")
        print("-" * 70)
        
        iterations = df['iteration'].nunique()
        total_episodes = df.groupby(['iteration', 'episode']).ngroups
        
        episode_progress = df.groupby(['iteration', 'episode'])['progress'].max()
        completed_episodes = (episode_progress == 100).sum()
        
        print(f"  Total iterations: {iterations}")
        print(f"  Total episodes: {total_episodes}")
        print(f"  Episodes completing track: {completed_episodes} ({completed_episodes/total_episodes*100:.1f}%)")
        print(f"  Average progress: {episode_progress.mean():.1f}%")
        print(f"  Best progress achieved: {episode_progress.max():.1f}%")
        
        # final iteration performance
        final_iter = df['iteration'].max()
        final_episodes = df[df['iteration'] == final_iter].groupby('episode')['progress'].max()
        final_completion = (final_episodes == 100).mean() * 100
        print(f"\n  Final iteration ({final_iter}) completion rate: {final_completion:.1f}%")
        
        # action diversity
        _, action_entropy = analyze_action_distribution(df)
        unique_actions = df['action'].nunique()
        print(f"\nACTION SPACE USAGE")
        print("-" * 70)
        print(f"  Unique actions used: {unique_actions}")
        print(f"  Action entropy: {action_entropy:.2f}")
        if action_entropy < 1.5:
            print(f"  [Low entropy - may be stuck using only a few actions]")
        elif action_entropy > 3.0:
            print(f"  [Good diversity in action selection]")
        
        # steering bias analysis
        print(f"\nSTEERING ANALYSIS")
        print("-" * 70)
        steering = df['steer']
        left_turns = (steering < -5).sum()
        right_turns = (steering > 5).sum()
        straight = ((steering >= -5) & (steering <= 5)).sum()
        total_steps = len(steering)
        
        print(f"  Left turns (< -5°):  {left_turns:6d} ({left_turns/total_steps*100:5.1f}%)")
        print(f"  Straight (-5° to 5°): {straight:6d} ({straight/total_steps*100:5.1f}%)")
        print(f"  Right turns (> 5°):  {right_turns:6d} ({right_turns/total_steps*100:5.1f}%)")
        
        # Check for bias
        turn_bias = abs(left_turns - right_turns) / total_steps * 100
        if turn_bias > 10:
            bias_direction = "left" if left_turns > right_turns else "right"
            print(f"  [WARNING: {turn_bias:.1f}% bias toward {bias_direction} turns - track-specific overfitting likely]")
        elif turn_bias > 5:
            bias_direction = "left" if left_turns > right_turns else "right"
            print(f"  [Slight {bias_direction} bias detected - may affect generalization]")
        else:
            print(f"  [Balanced steering - good for generalization]")
        
        # speed analysis
        print(f"\nSPEED STATISTICS")
        print("-" * 70)
        print(f"  Mean throttle: {df['throttle'].mean():.2f}")
        print(f"  Median throttle: {df['throttle'].median():.2f}")
        print(f"  Throttle range: [{df['throttle'].min():.2f}, {df['throttle'].max():.2f}]")
        
        # reward statistics
        print(f"\nREWARD STATISTICS")
        print("-" * 70)
        print(f"  Mean reward: {df['reward'].mean():.3f}")
        print(f"  Median reward: {df['reward'].median():.3f}")
        print(f"  Reward range: [{df['reward'].min():.3f}, {df['reward'].max():.3f}]")
        print(f"  Std dev: {df['reward'].std():.3f}")
        
        # Reward progression over iterations
        episode_rewards = df.groupby(['iteration', 'episode'])['reward'].sum()
        iter_avg_reward = episode_rewards.groupby('iteration').mean()
        
        if len(iter_avg_reward) >= 5:
            early_reward = iter_avg_reward.iloc[:5].mean()
            late_reward = iter_avg_reward.iloc[-5:].mean()
            reward_improvement = ((late_reward - early_reward) / abs(early_reward) * 100) if early_reward != 0 else 0
            print(f"  Early training avg (first 5 iter): {early_reward:.2f}")
            print(f"  Late training avg (last 5 iter):  {late_reward:.2f}")
            print(f"  Improvement: {reward_improvement:+.1f}%")
            
            if reward_improvement < 10:
                print(f"  [WARNING: Minimal reward improvement - model may not be learning effectively]")
        
        # NEW: Speed-reward correlation analysis
        print(f"\nSPEED-REWARD ANALYSIS")
        print("-" * 70)
        speed_reward_corr = df['throttle'].corr(df['reward'])
        print(f"  Speed-reward correlation: {speed_reward_corr:+.3f}")
        if speed_reward_corr < 0.3:
            print(f"  [WARNING: Weak speed incentive - reward function may not encourage speed]")
        elif speed_reward_corr > 0.6:
            print(f"  [Strong speed incentive in reward function]")
        else:
            print(f"  [Moderate speed incentive]")
        
        # Speed vs progress correlation
        print(f"\nBEHAVIOR ANALYSIS")
        print("-" * 70)
        episode_stats = df.groupby(['iteration', 'episode']).agg({
            'throttle': 'mean',
            'progress': 'max',
            'steps': 'max'
        })
        
        speed_progress_corr = episode_stats['throttle'].corr(episode_stats['progress'])
        print(f"  Speed-progress correlation: {speed_progress_corr:+.3f}")
        if speed_progress_corr < -0.3:
            print(f"  [Model learned to go SLOW for safety]")
        elif speed_progress_corr > 0.3:
            print(f"  [Model associates higher speed with better progress]")
        else:
            print(f"  [Weak speed-progress relationship]")
        
        # Episode length statistics
        episode_lengths = episode_stats['steps']
        print(f"\n  Episode length statistics:")
        print(f"    Mean steps per episode: {episode_lengths.mean():.1f}")
        print(f"    Median steps per episode: {episode_lengths.median():.1f}")
        print(f"    Max steps observed: {episode_lengths.max()}")
        
        # NEW: Episode efficiency change analysis
        if len(episode_lengths) >= 10:
            # Group by iteration to get mean episode length per iteration
            episode_lengths_per_iter = df.groupby(['iteration', 'episode'])['steps'].max().groupby('iteration').mean()
            early_length = episode_lengths_per_iter.iloc[:5].mean()
            late_length = episode_lengths_per_iter.iloc[-5:].mean()
            efficiency_change = ((late_length - early_length) / early_length * 100) if early_length > 0 else 0
            
            print(f"\n  Episode efficiency change:")
            print(f"    Early training avg length: {early_length:.1f} steps")
            print(f"    Late training avg length: {late_length:.1f} steps")
            print(f"    Change: {efficiency_change:+.1f}%")
            if efficiency_change > 50:
                print(f"    [WARNING: Episodes getting significantly longer - learning inefficiency]")
            elif efficiency_change < -20:
                print(f"    [Good: Episodes getting shorter - learning efficiency]")
            else:
                print(f"    [Moderate efficiency change]")
        
        if episode_lengths.mean() > 300:
            print(f"  [Long episodes suggest slow driving or inefficient exploration]")
        
        # NEW: Action space utilization metrics
        print(f"\nACTION SPACE UTILIZATION")
        print("-" * 70)
        speed_bins = np.linspace(0.5, 4.0, 21)
        speed_hist, _ = np.histogram(df['throttle'], bins=speed_bins)
        mean_freq = speed_hist.mean()
        effective_bins = (speed_hist > mean_freq).sum()
        print(f"  Speed bins with above-average usage: {effective_bins}/20 ({effective_bins/20*100:.0f}%)")
        if effective_bins < 5:
            print(f"  [WARNING: Limited speed diversity - possible bimodal distribution]")
        elif effective_bins > 15:
            print(f"  [Excellent: Broad speed range utilization]")
        else:
            print(f"  [Moderate speed range utilization]")
        
        # failure analysis
        failures, waypoint_coords = analyze_failure_points(df)
        if len(failures) > 0:
            print(f"\nFAILURE ANALYSIS")
            print("-" * 70)
            print(f"  Total off-track events: {len(df[df['episode_status'] == 'off_track'])}")
            print(f"  Unique failure waypoints: {len(failures)}")
            print(f"\n  Top 5 failure waypoints:")
            for wp, count in failures.head().items():
                pct = count / failures.sum() * 100
                if waypoint_coords is not None and wp in waypoint_coords.index:
                    x, y = waypoint_coords.loc[wp, 'X'], waypoint_coords.loc[wp, 'Y']
                    print(f"    Waypoint {int(wp):3d}: {int(count):4d} failures ({pct:5.1f}%) at position ({x:.2f}, {y:.2f})")
                else:
                    print(f"    Waypoint {int(wp):3d}: {int(count):4d} failures ({pct:5.1f}%)")
            
            # concentration analysis
            top_5_pct = failures.head().sum() / failures.sum() * 100
            print(f"\n  Failure concentration: {top_5_pct:.1f}% of failures occur at top 5 waypoints")
            if top_5_pct > 60:
                print(f"  [High concentration - failures are localized to specific track sections]")
    
    # evaluatin analysis
    all_evals = run.get_all_evaluations()
    
    if len(all_evals) > 0:
        print(f"\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        for track_name, eval_df in all_evals.items():
            print(f"\nTrack: {track_name}")
            print("-" * 70)
            
            # Estimate track length from waypoint coordinates
            track_length = estimate_track_length(eval_df)
            if track_length is not None:
                print(f"  Estimated track length: {track_length:.2f}m")
            
            # NEW: Calculate track complexity
            track_complexity = calculate_track_complexity(eval_df)
            if track_complexity is not None:
                print(f"  Track complexity: {track_complexity:.4f}")
                # Provide interpretation
                if track_complexity < 0.01:
                    print(f"    [Simple track - low curvature variance]")
                elif track_complexity < 0.05:
                    print(f"    [Moderate complexity]")
                else:
                    print(f"    [Complex track - high curvature variance]")
            
            eval_episodes = eval_df.groupby('episode').agg({
                'progress': 'max',
                'steps': 'count',
                'episode_status': 'first'
            })
            
            n_trials = len(eval_episodes)
            completed = (eval_episodes['progress'] == 100).sum()
            
            print(f"  Trials completing track: {completed}/{n_trials} ({completed/n_trials*100:.0f}%)")
            
            if completed > 0:
                complete_eps = eval_episodes[eval_episodes['progress'] == 100]
                # estimate lap time (steps * 0.067 seconds per step)
                lap_times = complete_eps['steps'] * 0.067
                print(f"\n  Lap Times:")
                print(f"    Best:    {lap_times.min():6.2f}s")
                print(f"    Average: {lap_times.mean():6.2f}s")
                print(f"    Worst:   {lap_times.max():6.2f}s")
                print(f"    Std Dev: {lap_times.std():6.2f}s")
                
                # Calculate and display track speed (m/s) if length is available
                if track_length is not None:
                    track_speeds = track_length / lap_times
                    print(f"\n  Track Speed (m/s):")
                    print(f"    Best:    {track_speeds.max():6.2f} m/s")
                    print(f"    Average: {track_speeds.mean():6.2f} m/s")
                    print(f"    Worst:   {track_speeds.min():6.2f} m/s")
                    print(f"    Std Dev: {track_speeds.std():6.2f} m/s")
                
                # performance assessment
                best_time = lap_times.min()
                if best_time < 15:
                    print(f"\n  [Excellent performance - sub-15s lap time]")
                elif best_time < 25:
                    print(f"  [Good performance - competitive lap times]")
                elif best_time < 40:
                    print(f"  [Moderate performance - room for optimization]")
                else:
                    print(f"  [Slow lap times - consider improving reward function]")
            else:
                print(f"\n  [Model did not complete any evaluation laps]")
                incomplete = eval_episodes[eval_episodes['progress'] < 100]
                if len(incomplete) > 0:
                    avg_progress = incomplete['progress'].mean()
                    print(f"  Average progress in failed attempts: {avg_progress:.1f}%")
                    
                    # analyze failure points on this track
                    failures = eval_df[eval_df['episode_status'] == 'off_track']
                    if len(failures) > 0:
                        failure_wps = failures['closest_waypoint'].value_counts().head(3)
                        print(f"  Top failure waypoints: {', '.join([str(int(wp)) for wp in failure_wps.index])}")
        
        # Generate comparative summary table if multiple tracks
        if len(all_evals) >= 2:
            print(f"\n" + "=" * 70)
            print("CROSS-TRACK PERFORMANCE SUMMARY")
            print("=" * 70)
            
            # Collect data for table
            summary_data = []
            for track_name, eval_df in all_evals.items():
                track_length = estimate_track_length(eval_df)
                track_complexity = calculate_track_complexity(eval_df)
                eval_episodes = eval_df.groupby('episode').agg({
                    'progress': 'max',
                    'steps': 'count'
                })
                
                completed = (eval_episodes['progress'] == 100).sum()
                n_trials = len(eval_episodes)
                completion_rate = completed / n_trials * 100
                
                if completed > 0:
                    lap_times = eval_episodes[eval_episodes['progress'] == 100]['steps'] * 0.067
                    best_time = lap_times.min()
                    avg_speed = track_length / lap_times.mean() if track_length else None
                    reliability = "Excellent" if completion_rate == 100 else f"{completion_rate:.0f}%"
                else:
                    best_time = None
                    avg_speed = None
                    reliability = "Failed"
                
                # Clean track name for display
                display_name = track_name
                if display_name.startswith(f"{run.model_name}_eval_"):
                    display_name = display_name[len(f"{run.model_name}_eval_"):]
                
                summary_data.append({
                    'Track': display_name[:15],  # Truncate long names
                    'Length': f"{track_length:.1f}m" if track_length else "N/A",
                    'Complex': f"{track_complexity:.3f}" if track_complexity else "N/A",
                    'Completion': f"{completion_rate:.0f}%",
                    'Best Time': f"{best_time:.1f}s" if best_time else "DNF",
                    'Avg Speed': f"{avg_speed:.2f}" if avg_speed else "N/A",
                    'Reliability': reliability
                })
            
            # Print formatted table
            if summary_data:
                # Header
                print(f"\n{'Track':<15} {'Length':<8} {'Complex':<9} {'Complete':<10} {'Best Time':<11} {'Avg Speed':<10} {'Reliability':<12}")
                print("-" * 80)
                
                # Rows
                for row in summary_data:
                    print(f"{row['Track']:<15} {row['Length']:<8} {row['Complex']:<9} {row['Completion']:<10} "
                          f"{row['Best Time']:<11} {row['Avg Speed']:<10} {row['Reliability']:<12}")
                
                print("\nNote: Avg Speed = track_length / avg_lap_time (m/s)")
                print("      Complex = curvature variance (higher = more complex track)")
    
    print("\n" + "=" * 70 + "\n")


def compare_runs(run_dirs):
    """Compare multiple training runs"""
    runs = [DeepRacerRun(d) for d in run_dirs]
    
    print("\n" + "=" * 70)
    print("COMPARING MULTIPLE RUNS")
    print("=" * 70)
    
    comparison_data = []
    for run in runs:
        df = run.training_df
        eval_df = run.eval_df
        
        if df is not None:
            episode_progress = df.groupby(['iteration', 'episode'])['progress'].max()
            completed = (episode_progress == 100).sum()
            total = len(episode_progress)
            
            data = {
                'run': run.name,
                'iterations': df['iteration'].nunique(),
                'completion_rate': completed / total * 100,
                'avg_progress': episode_progress.mean(),
            }
            
            if eval_df is not None:
                eval_episodes = eval_df.groupby('episode')['progress'].max()
                eval_completed = (eval_episodes == 100).sum()
                if eval_completed > 0:
                    eval_steps = eval_df[eval_df.groupby('episode')['progress'].transform('max') == 100].groupby('episode')['steps'].count()
                    best_lap = eval_steps.min() * 0.067
                    data['eval_completion'] = eval_completed / len(eval_episodes) * 100
                    data['best_lap_time'] = best_lap
            
            comparison_data.append(data)
    
    comp_df = pd.DataFrame(comparison_data)
    print("\n" + comp_df.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze AWS DeepRacer training and evaluation logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # text report only (always displays and saves to .txt)
  python deepracer_analyzer.py results/baseline
  
  # plots (saved to file, not displayed)
  python deepracer_analyzer.py results/baseline --plot
  
  # analyze model w/ multiple eval tracks
  python deepracer_analyzer.py results/baseline_train \\
    --eval-dir results/baseline_eval_oval \\
    --eval-dir results/baseline_eval_bowtie \\
    --plot
  
  # display plots
  python deepracer_analyzer.py results/baseline --plot --display
  
  # save outputs to specific directory
  python deepracer_analyzer.py results/baseline --plot --output-dir analysis_outputs/
  
  # compare multiple runs
  python deepracer_analyzer.py results/baseline results/improved --compare
        """
    )
    
    parser.add_argument('run_dirs', nargs='+', 
                       help='Directory/directories containing DeepRacer run data')
    parser.add_argument('--eval-dir', action='append', dest='eval_dirs',
                       help='Additional evaluation directory for the same model (can be used multiple times)')
    parser.add_argument('--auto-eval', action='store_true',
                       help='Automatically discover evaluation directories matching model name pattern')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate visualization plots')
    parser.add_argument('--output-dir', type=str, 
                       help='Directory to save plots and logs (default: source run directory)')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare multiple runs (requires 2+ run_dirs)')
    parser.add_argument('--display', action='store_true',
                       help='Display plots interactively (default: only save to file)')
    
    args = parser.parse_args()
    
    # Handle comparison mode
    if args.compare:
        if len(args.run_dirs) < 2:
            print("Error: --compare requires at least 2 run directories")
            return
        compare_runs(args.run_dirs)
        return
    
    # Single run analysis
    if len(args.run_dirs) > 1:
        print("Warning: Multiple directories provided but --compare not specified. Analyzing first directory only.")
    
    # Auto-discover evaluation directories if requested
    eval_dirs = args.eval_dirs or []
    if args.auto_eval:
        run_path = Path(args.run_dirs[0])
        
        # extract model name from training directory
        # supports patterns: baseline_train, baseline/baseline_train, baseline/train
        run_name = run_path.name
        if run_name.endswith('_train'):
            model_name = run_name[:-6]  # Remove '_train' suffix
        else:
            model_name = run_name
        
        # search parent directory for matching eval folders
        parent_dir = run_path.parent
        eval_pattern = f"{model_name}_eval_*"
        discovered_evals = list(parent_dir.glob(eval_pattern))
        
        if discovered_evals:
            print(f"Auto-discovered {len(discovered_evals)} evaluation folder(s) matching '{eval_pattern}':")
            for eval_dir in discovered_evals:
                print(f"  - {eval_dir.name}")
                eval_dirs.append(str(eval_dir))
        else:
            print(f"No evaluation folders found matching pattern '{eval_pattern}' in {parent_dir}")
    
    run = DeepRacerRun(args.run_dirs[0], eval_dirs=eval_dirs)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = run.run_dir
    
    # logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{run.model_name}_analysis_{timestamp}.txt"
    logger = Logger(log_file)
    sys.stdout = logger
    
    try:
        generate_report(run)
        
        # generate plots if requested
        if args.plot:
            # Training plots
            plot_training_progression(run, output_dir, display=args.display)
            
            # Evaluation plots
            all_evals = run.get_all_evaluations()
            
            if len(all_evals) > 0:
                print(f"\nGenerating evaluation plots...")
                
                # Choose plot type based on number of tracks
                if len(all_evals) == 1:
                    # Single track: detailed analysis
                    track_name, eval_df = list(all_evals.items())[0]
                    plot_evaluation_detailed(run, track_name, eval_df, 
                                           output_dir, display=args.display)
                else:
                    # Multiple tracks: comparison plot
                    plot_evaluation_comparison(run, all_evals, 
                                             output_dir, display=args.display)
                
                # Generate per-track failure maps
                print(f"\nGenerating per-track failure maps...")
                for track_name, eval_df in all_evals.items():
                    plot_evaluation_track_failures(run, track_name, eval_df,
                                                  output_dir, display=args.display)
                
                # Generate per-track speed profiles
                print(f"\nGenerating per-track speed profiles...")
                for track_name, eval_df in all_evals.items():
                    plot_speed_profile(run, track_name, eval_df,
                                     output_dir, display=args.display)
            else:
                print("\nNo evaluation data available - skipping evaluation plots")
        
        print(f"\nAnalysis complete. Log saved to {log_file}")
        
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == '__main__':
    main()
