#!/usr/bin/env python3
"""
AI Engine Profiling Visualization Script
Aggregates profiling results from multiple systems and creates visualizations
showing how different AI engines scale with computational complexity.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import json
from matplotlib.ticker import FuncFormatter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_csv_files(csv_pattern):
    """Load all CSV files matching the pattern"""
    csv_files = list(Path('.').glob(csv_pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {csv_pattern}")
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records: {len(combined_df)}")
    print(f"Systems: {combined_df['system'].unique()}")
    print(f"Engines: {combined_df['engine'].unique()}")
    print(f"Devices: {combined_df['device'].unique()}")
    print(f"Operators: {combined_df['operator'].unique()}")
    
    return combined_df


def format_flops(x, pos):
    """Format FLOPs for axis labels"""
    if x >= 1e12:
        return f'{x/1e12:.1f}T'
    elif x >= 1e9:
        return f'{x/1e9:.1f}G'
    elif x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.1f}K'
    else:
        return f'{x:.0f}'


def calculate_throughput(df):
    """Calculate throughput in GFLOPS"""
    df['gflops'] = (df['flops'] / 1e9) / (df['mean_time_ms'] / 1000)
    return df


def plot_scaling_by_operator(df, output_dir):
    """Plot how each engine scales with FLOPs for each operator type"""
    df = calculate_throughput(df)
    
    operators = df['operator'].unique()
    
    for operator in operators:
        op_df = df[df['operator'] == operator].copy()
        
        if len(op_df) < 2:
            continue
        
        # Sort by FLOPs
        op_df = op_df.sort_values('flops')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Execution Time vs FLOPs
        ax1 = axes[0]
        for (engine, device), group in op_df.groupby(['engine', 'device']):
            label = f"{engine}-{device}"
            ax1.plot(group['flops'], group['mean_time_ms'], 
                    marker='o', label=label, linewidth=2, markersize=6, alpha=0.7)
        
        ax1.set_xlabel('FLOPs', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{operator.upper()} - Execution Time Scaling', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(FuncFormatter(format_flops))
        
        # Plot 2: Throughput (GFLOPS) vs FLOPs
        ax2 = axes[1]
        for (engine, device), group in op_df.groupby(['engine', 'device']):
            label = f"{engine}-{device}"
            ax2.plot(group['flops'], group['gflops'], 
                    marker='s', label=label, linewidth=2, markersize=6, alpha=0.7)
        
        ax2.set_xlabel('FLOPs', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Throughput (GFLOPS)', fontsize=12, fontweight='bold')
        ax2.set_title(f'{operator.upper()} - Throughput Scaling', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xscale('log')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(FuncFormatter(format_flops))
        
        plt.tight_layout()
        output_path = output_dir / f'scaling_{operator}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def plot_engine_comparison(df, output_dir):
    """Compare all engines across all operators"""
    df = calculate_throughput(df)
    
    # Create bins for FLOPs ranges
    df['flops_range'] = pd.cut(df['flops'], bins=10, labels=False)
    
    # Aggregate by engine, device, and flops range
    agg_df = df.groupby(['engine', 'device', 'flops_range', 'operator']).agg({
        'mean_time_ms': 'mean',
        'gflops': 'mean',
        'flops': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Overall execution time scaling
    ax1 = axes[0, 0]
    for (engine, device), group in agg_df.groupby(['engine', 'device']):
        group_sorted = group.sort_values('flops')
        label = f"{engine}-{device}"
        ax1.plot(group_sorted['flops'], group_sorted['mean_time_ms'], 
                marker='o', label=label, linewidth=2.5, markersize=5, alpha=0.8)
    
    ax1.set_xlabel('Average FLOPs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Performance Scaling - Execution Time', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='best', framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(FuncFormatter(format_flops))
    
    # Plot 2: Overall throughput scaling
    ax2 = axes[0, 1]
    for (engine, device), group in agg_df.groupby(['engine', 'device']):
        group_sorted = group.sort_values('flops')
        label = f"{engine}-{device}"
        ax2.plot(group_sorted['flops'], group_sorted['gflops'], 
                marker='s', label=label, linewidth=2.5, markersize=5, alpha=0.8)
    
    ax2.set_xlabel('Average FLOPs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Throughput (GFLOPS)', fontsize=12, fontweight='bold')
    ax2.set_title('Overall Performance Scaling - Throughput', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_xscale('log')
    ax2.legend(loc='best', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(FuncFormatter(format_flops))
    
    # Plot 3: Box plot of throughput by engine
    ax3 = axes[1, 0]
    df['engine_device'] = df['engine'] + '-' + df['device']
    engine_order = df.groupby('engine_device')['gflops'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, y='engine_device', x='gflops', order=engine_order, ax=ax3)
    ax3.set_xlabel('Throughput (GFLOPS)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Engine-Device', fontsize=12, fontweight='bold')
    ax3.set_title('Throughput Distribution by Engine', 
                 fontsize=14, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Efficiency ratio (higher is better)
    ax4 = axes[1, 1]
    
    # Calculate efficiency: GFLOPS per ms
    efficiency_df = df.groupby(['engine', 'device']).agg({
        'gflops': 'mean',
        'mean_time_ms': 'mean'
    }).reset_index()
    efficiency_df['efficiency'] = efficiency_df['gflops'] / efficiency_df['mean_time_ms']
    efficiency_df['engine_device'] = efficiency_df['engine'] + '-' + efficiency_df['device']
    efficiency_df = efficiency_df.sort_values('efficiency', ascending=True)
    
    bars = ax4.barh(efficiency_df['engine_device'], efficiency_df['efficiency'])
    
    # Color bars by engine
    colors = plt.cm.Set3(np.linspace(0, 1, len(efficiency_df)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax4.set_xlabel('Efficiency (GFLOPS/ms)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Engine-Device', fontsize=12, fontweight='bold')
    ax4.set_title('Computational Efficiency by Engine', 
                 fontsize=14, fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = output_dir / 'engine_comparison_overall.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_operator_heatmap(df, output_dir):
    """Create heatmap showing performance across operators and engines"""
    df = calculate_throughput(df)
    
    # Create pivot table
    df['engine_device'] = df['engine'] + '-' + df['device']
    
    # Average throughput by operator and engine
    pivot_df = df.pivot_table(
        values='gflops',
        index='operator',
        columns='engine_device',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Throughput (GFLOPS)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Average Throughput Heatmap by Operator and Engine', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Engine-Device', fontsize=12, fontweight='bold')
    ax.set_ylabel('Operator', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'operator_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_system_comparison(df, output_dir):
    """Compare performance across different systems"""
    if df['system'].nunique() < 2:
        print("Skipping system comparison (only one system in data)")
        return
    
    df = calculate_throughput(df)
    df['engine_device'] = df['engine'] + '-' + df['device']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Throughput by system
    ax1 = axes[0]
    system_throughput = df.groupby(['system', 'engine_device'])['gflops'].mean().reset_index()
    
    systems = system_throughput['system'].unique()
    engine_devices = system_throughput['engine_device'].unique()
    x = np.arange(len(engine_devices))
    width = 0.8 / len(systems)
    
    for i, system in enumerate(systems):
        system_data = system_throughput[system_throughput['system'] == system]
        values = [system_data[system_data['engine_device'] == ed]['gflops'].values[0] 
                 if len(system_data[system_data['engine_device'] == ed]) > 0 else 0 
                 for ed in engine_devices]
        ax1.bar(x + i * width, values, width, label=system, alpha=0.8)
    
    ax1.set_xlabel('Engine-Device', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Throughput (GFLOPS)', fontsize=12, fontweight='bold')
    ax1.set_title('Throughput Comparison Across Systems', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x + width * (len(systems) - 1) / 2)
    ax1.set_xticklabels(engine_devices, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Execution time by system
    ax2 = axes[1]
    system_time = df.groupby(['system', 'engine_device'])['mean_time_ms'].mean().reset_index()
    
    for i, system in enumerate(systems):
        system_data = system_time[system_time['system'] == system]
        values = [system_data[system_data['engine_device'] == ed]['mean_time_ms'].values[0] 
                 if len(system_data[system_data['engine_device'] == ed]) > 0 else 0 
                 for ed in engine_devices]
        ax2.bar(x + i * width, values, width, label=system, alpha=0.8)
    
    ax2.set_xlabel('Engine-Device', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Execution Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Execution Time Comparison Across Systems', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x + width * (len(systems) - 1) / 2)
    ax2.set_xticklabels(engine_devices, rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'system_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_stats(df, output_dir):
    """Generate summary statistics"""
    df = calculate_throughput(df)
    
    summary = {}
    
    # Overall statistics
    summary['overall'] = {
        'total_tests': len(df),
        'unique_operators': df['operator'].nunique(),
        'unique_engines': df['engine'].nunique(),
        'unique_systems': df['system'].nunique(),
        'avg_throughput_gflops': float(df['gflops'].mean()),
        'max_throughput_gflops': float(df['gflops'].max()),
        'min_throughput_gflops': float(df['gflops'].min()),
    }
    
    # Per-engine statistics
    df['engine_device'] = df['engine'] + '-' + df['device']
    engine_stats = df.groupby('engine_device').agg({
        'gflops': ['mean', 'std', 'min', 'max'],
        'mean_time_ms': ['mean', 'std', 'min', 'max']
    }).round(3)

    # Flatten column names and convert to dict with string keys
    engine_stats.columns = ['_'.join(col).strip() for col in engine_stats.columns.values]
    engine_stats_dict = engine_stats.reset_index().to_dict('records')
    summary['per_engine'] = {stat['engine_device']: {k: v for k, v in stat.items() if k != 'engine_device'}
                             for stat in engine_stats_dict}

    # Per-operator statistics
    operator_stats = df.groupby('operator').agg({
        'gflops': ['mean', 'std', 'min', 'max'],
        'mean_time_ms': ['mean', 'std', 'min', 'max']
    }).round(3)

    # Flatten column names and convert to dict with string keys
    operator_stats.columns = ['_'.join(col).strip() for col in operator_stats.columns.values]
    operator_stats_dict = operator_stats.reset_index().to_dict('records')
    summary['per_operator'] = {stat['operator']: {k: v for k, v in stat.items() if k != 'operator'}
                               for stat in operator_stats_dict}
    
    # Best performing configurations
    best_throughput = df.nlargest(10, 'gflops')[['engine', 'device', 'operator', 'gflops', 'flops']].to_dict('records')
    summary['best_throughput'] = best_throughput
    
    # Fastest execution
    fastest = df.nsmallest(10, 'mean_time_ms')[['engine', 'device', 'operator', 'mean_time_ms', 'flops']].to_dict('records')
    summary['fastest_execution'] = fastest
    
    # Save to JSON
    output_path = output_dir / 'summary_statistics.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_path}")
    
    # Save engine stats to CSV
    engine_stats.to_csv(output_dir / 'engine_statistics.csv')
    operator_stats.to_csv(output_dir / 'operator_statistics.csv')
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total tests: {summary['overall']['total_tests']}")
    print(f"Unique operators: {summary['overall']['unique_operators']}")
    print(f"Unique engines: {summary['overall']['unique_engines']}")
    print(f"Unique systems: {summary['overall']['unique_systems']}")
    print(f"Average throughput: {summary['overall']['avg_throughput_gflops']:.2f} GFLOPS")
    print(f"Max throughput: {summary['overall']['max_throughput_gflops']:.2f} GFLOPS")
    print("\nTop 5 configurations by throughput:")
    for i, config in enumerate(best_throughput[:5], 1):
        print(f"  {i}. {config['engine']}-{config['device']} {config['operator']}: {config['gflops']:.2f} GFLOPS")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize AI engine profiling results from multiple systems'
    )
    parser.add_argument('--input', '-i', default='*.csv',
                       help='CSV file pattern (default: *.csv)')
    parser.add_argument('--output-dir', '-o', default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--format', '-f', choices=['png', 'pdf', 'svg'], 
                       default='png', help='Output format')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("AI Engine Profiling Visualization Tool")
    print("="*80)
    
    # Load data
    df = load_csv_files(args.input)
    
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    
    # Generate all plots
    plot_scaling_by_operator(df, output_dir)
    plot_engine_comparison(df, output_dir)
    plot_operator_heatmap(df, output_dir)
    plot_system_comparison(df, output_dir)
    generate_summary_stats(df, output_dir)
    
    print("\n" + "="*80)
    print(f"All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()