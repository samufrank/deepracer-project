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
    ax.set_title('Track Layout with Failure Locations')
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
    
    # Create main analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Analysis: {run.name}', fontsize=16)
    
    # Progress over iterations
    iter_progress = df.groupby('iteration')['progress'].max()
    axes[0, 0].plot(iter_progress.index, iter_progress.values, 'o-')
    axes[0, 0].axhline(y=100, color='r', linestyle='--', alpha=0.3, label='Complete lap')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Max Progress (%)')
    axes[0, 0].set_title('Best Progress Per Iteration')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Completion rate over time
    episode_progress = df.groupby(['iteration', 'episode'])['progress'].max()
    complete_episodes = episode_progress.groupby('iteration').apply(
        lambda x: (x == 100).mean() * 100
    )
    axes[0, 1].plot(complete_episodes.index, complete_episodes.values, 'o-', color='green')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Episodes Completing Track (%)')
    axes[0, 1].set_title('Completion Rate Over Training')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 105])
    
    # failure heatmap by waypoint
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
    
    # speed distribution
    axes[1, 1].hist(df['throttle'], bins=30, color='blue', alpha=0.6, edgecolor='black')
    axes[1, 1].set_xlabel('Throttle (speed)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Speed Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{run.name}_training_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training analysis plot saved to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)
    
    # create separate track visualization
    track_output = Path(output_dir) / f"{run.name}_track_failures.png" if output_dir else None
    track_fig = plot_track_with_failures(df, track_output)
    if output_dir:
        print(f"Track failure map saved to {Path(output_dir) / f'{run.name}_track_failures.png'}")
    
    if display:
        plt.show()
    else:
        plt.close(track_fig)


def plot_evaluation_detailed(run, track_name, eval_df, output_dir=None, display=True):
    """Detailed visualization for single evaluation track"""
    if eval_df is None or len(eval_df) == 0:
        print(f"No evaluation data available for {track_name}")
        return
    
    print(f"Generating detailed evaluation plot for {track_name}...")
    
    # Create main analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Evaluation Analysis: {run.name} - {track_name}', fontsize=16)
    
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
        output_path = Path(output_dir) / f"{run.name}_eval_detailed.png"
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
    completion_rates = []
    best_lap_times = []
    avg_lap_times = []
    lap_time_data = []  # for box plot
    
    for track_name, eval_df in all_evals.items():
        episode_stats = eval_df.groupby('episode').agg({
            'progress': 'max',
            'steps': 'count'
        })
        
        track_names.append(track_name)
        
        # Completion rate
        n_complete = (episode_stats['progress'] == 100).sum()
        n_total = len(episode_stats)
        completion_rates.append(n_complete / n_total * 100)
        
        # Lap times
        completed = episode_stats[episode_stats['progress'] == 100]
        if len(completed) > 0:
            lap_times = completed['steps'] * 0.067
            best_lap_times.append(lap_times.min())
            avg_lap_times.append(lap_times.mean())
            lap_time_data.append(lap_times.values)
        else:
            best_lap_times.append(None)
            avg_lap_times.append(None)
            lap_time_data.append([])
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Multi-Track Evaluation Comparison: {run.name}', fontsize=16)
    
    x_pos = np.arange(len(track_names))
    
    # Panel 1: Completion rates
    bars1 = axes[0].bar(x_pos, completion_rates, color='green', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Track')
    axes[0].set_ylabel('Completion Rate (%)')
    axes[0].set_title('Track Completion Rates')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(track_names, rotation=45, ha='right')
    axes[0].set_ylim([0, 105])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, completion_rates):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Panel 2: Best lap times
    valid_best_times = [t for t in best_lap_times if t is not None]
    valid_avg_times = [t for t in avg_lap_times if t is not None]
    valid_track_indices = [i for i, t in enumerate(best_lap_times) if t is not None]
    
    if len(valid_best_times) > 0:
        x_valid = np.array(valid_track_indices)
        width = 0.35
        
        bars2 = axes[1].bar(x_valid - width/2, valid_best_times, width, 
                           label='Best', color='blue', alpha=0.7, edgecolor='black')
        bars3 = axes[1].bar(x_valid + width/2, valid_avg_times, width,
                           label='Average', color='orange', alpha=0.7, edgecolor='black')
        
        axes[1].set_xlabel('Track')
        axes[1].set_ylabel('Lap Time (s)')
        axes[1].set_title('Lap Times (Completed Tracks Only)')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(track_names, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars2, valid_best_times):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars3, valid_avg_times):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    else:
        axes[1].text(0.5, 0.5, 'No completed laps across all tracks',
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=12)
        axes[1].set_title('Lap Times (Completed Tracks Only)')
    
    # Panel 3: Lap time distributions (box plot)
    valid_lap_data = [data for data in lap_time_data if len(data) > 0]
    valid_labels = [track_names[i] for i, data in enumerate(lap_time_data) if len(data) > 0]
    
    if len(valid_lap_data) > 0:
        bp = axes[2].boxplot(valid_lap_data, tick_labels=valid_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        axes[2].set_xlabel('Track')
        axes[2].set_ylabel('Lap Time (s)')
        axes[2].set_title('Lap Time Consistency')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Adjust x-axis labels
        axes[2].set_xticklabels(valid_labels, rotation=45, ha='right')
    else:
        axes[2].text(0.5, 0.5, 'No completed laps for distribution analysis',
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=12)
        axes[2].set_title('Lap Time Consistency')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{run.name}_eval_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Multi-track comparison plot saved to {output_path}")
    
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
    ax.set_title(f'Evaluation Failure Map: {track_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if output_dir:
        # Clean track name for filename
        safe_track_name = track_name.replace(' ', '_').replace('/', '_')
        output_path = Path(output_dir) / f"{run.name}_eval_{safe_track_name}_failures.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Evaluation failure map saved to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)


def generate_report(run):
    """Generate text summary"""
    print("\n" + "=" * 70)
    print(f"DEEPRACER ANALYSIS REPORT: {run.name}")
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
        
        if episode_lengths.mean() > 300:
            print(f"  [Long episodes suggest slow driving or inefficient exploration]")
        
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
    log_file = output_dir / f"{run.name}_analysis_{timestamp}.txt"
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
            else:
                print("\nNo evaluation data available - skipping evaluation plots")
        
        print(f"\nAnalysis complete. Log saved to {log_file}")
        
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == '__main__':
    main()
