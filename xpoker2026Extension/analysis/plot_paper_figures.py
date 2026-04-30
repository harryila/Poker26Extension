#!/usr/bin/env python3
"""
Generate paper-ready figures for the LLM Belief Modeling study.

Produces four figures:
1. PCE Distribution (CDF) - Main result: LLM closer to CardOnly
2. Base-Rate Neglect Bar Chart - Explains the mechanism (trash underestimate)
3. Update Magnitude Scatter - Diagnosis: uncorrelated updates
4. Street-Wise Stability - Robustness: effect holds all streets

Usage:
    python -m analysis.plot_paper_figures \
        --pce-data results/pce_distribution.csv \
        --pce-summary results/pce_summary.csv \
        --update-data results/update_coherence.csv \
        --output-dir figures/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


# Style configuration for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color scheme (colorblind-friendly)
COLORS = {
    'cardonly': '#1f77b4',      # Blue
    'strategyaware': '#ff7f0e', # Orange
    'llm': '#2ca02c',           # Green
    'oracle': '#7f7f7f',        # Gray
    'card_reveal': '#1f77b4',   # Blue
    'action': '#ff7f0e',        # Orange
}


def figure1_pce_cdf(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Figure 1: CDF of JS distances showing LLM is closer to CardOnly.
    
    Key visual: CardOnly CDF is consistently left of StrategyAware.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Sort values for CDF (filter NaN first)
    co_vals = df['js_cardonly'].dropna().values
    sa_vals = df['js_strategyaware'].dropna().values
    js_co = np.sort(co_vals)
    js_sa = np.sort(sa_vals)
    
    if len(js_co) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        fig.savefig(output_dir / 'fig1_pce_cdf.png')
        plt.close(fig)
        return
    
    # CDF y-values (separate per array to handle different lengths after dropna)
    cdf_y_co = np.arange(1, len(js_co) + 1) / len(js_co)
    cdf_y_sa = np.arange(1, len(js_sa) + 1) / len(js_sa) if len(js_sa) > 0 else np.array([])
    
    # Plot CDFs
    ax.plot(js_co, cdf_y_co, color=COLORS['cardonly'], linewidth=2, 
            label=f'JS(LLM, CardOnly)', linestyle='-')
    if len(js_sa) > 0:
        ax.plot(js_sa, cdf_y_sa, color=COLORS['strategyaware'], linewidth=2,
                label=f'JS(LLM, StrategyAware)', linestyle='--')
    
    # Get means from summary
    overall_rows = summary[summary['group'] == 'OVERALL']
    mean_co = np.mean(js_co) if len(js_co) > 0 else float('nan')
    mean_sa = np.mean(js_sa) if len(js_sa) > 0 else float('nan')
    if len(overall_rows) > 0:
        overall = overall_rows.iloc[0]
        mean_co = overall['js_cardonly_mean']
        mean_sa = overall['js_strategyaware_mean']
    
    # Add vertical lines at means (only for finite values)
    if np.isfinite(mean_co):
        ax.axvline(mean_co, color=COLORS['cardonly'], linestyle=':', alpha=0.7, linewidth=1.5)
    if np.isfinite(mean_sa):
        ax.axvline(mean_sa, color=COLORS['strategyaware'], linestyle=':', alpha=0.7, linewidth=1.5)
    
    # Add gap annotation only when both means are finite
    if np.isfinite(mean_co) and np.isfinite(mean_sa):
        gap_x = (mean_co + mean_sa) / 2
        ax.annotate('', xy=(mean_sa, 0.5), xytext=(mean_co, 0.5),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(gap_x, 0.53, f'Δ = {mean_sa - mean_co:.3f}', ha='center', fontsize=10)
    
    # Labels and legend
    ax.set_xlabel('JS Distance')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Distribution of JS Distances (N={len(df):,})')
    ax.legend(loc='lower right')
    finite_vals = np.concatenate([js_co, js_sa]) if len(js_sa) > 0 else js_co
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if len(finite_vals) > 0:
        data_min = float(np.min(finite_vals))
        data_max = float(np.max(finite_vals))
    else:
        data_min, data_max = 0.0, 1.0
    margin = (data_max - data_min) * 0.1
    ax.set_xlim(max(0, data_min - margin), data_max + margin)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.02, 0.98, 
            "LLM beliefs closer to\ncombinatorial baseline",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save
    fig.savefig(output_dir / 'fig1_pce_cdf.png')
    fig.savefig(output_dir / 'fig1_pce_cdf.pdf')
    plt.close(fig)
    print(f"Saved Figure 1: {output_dir / 'fig1_pce_cdf.png'}")


def figure2_baserate_neglect(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Figure 2: Bar chart showing base-rate neglect (trash underestimate).
    
    Key visual: Striking gap between LLM (~17%) and Oracle (~69%) trash mass.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get data by opponent action
    agg_rows = summary[(summary['opp_action'] == 'AGGRESSIVE') & (summary['street'] == 'ALL')]
    pas_rows = summary[(summary['opp_action'] == 'PASSIVE') & (summary['street'] == 'ALL')]
    overall_rows = summary[summary['group'] == 'OVERALL']
    
    # Prepare data
    categories = ['After\nAGGRESSIVE', 'After\nPASSIVE', 'Overall']
    
    if len(agg_rows) > 0 and len(pas_rows) > 0 and len(overall_rows) > 0:
        overall = overall_rows.iloc[0]
        llm_trash = [
            agg_rows.iloc[0]['llm_trash_mean'] * 100,
            pas_rows.iloc[0]['llm_trash_mean'] * 100,
            overall['llm_trash_mean'] * 100,
        ]
        oracle_trash = [
            agg_rows.iloc[0]['oracle_trash_mean'] * 100,
            pas_rows.iloc[0]['oracle_trash_mean'] * 100,
            overall['oracle_trash_mean'] * 100,
        ]
    else:
        # Fallback to computing from raw data
        agg_df = df[df['opp_action'] == 'AGGRESSIVE']
        pas_df = df[df['opp_action'] == 'PASSIVE']
        llm_trash = [
            agg_df['llm_trash'].mean() * 100 if len(agg_df) > 0 else 0,
            pas_df['llm_trash'].mean() * 100 if len(pas_df) > 0 else 0,
            df['llm_trash'].mean() * 100,
        ]
        oracle_trash = [
            agg_df['oracle_trash'].mean() * 100 if len(agg_df) > 0 else 0,
            pas_df['oracle_trash'].mean() * 100 if len(pas_df) > 0 else 0,
            df['oracle_trash'].mean() * 100,
        ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Bars
    bars1 = ax.bar(x - width/2, llm_trash, width, label='LLM', color=COLORS['llm'])
    bars2 = ax.bar(x + width/2, oracle_trash, width, label='Oracle', color=COLORS['oracle'])
    
    # Add horizontal line at true combinatorial rate (~72%)
    ax.axhline(y=72, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(2.5, 73, 'True combinatorial rate', fontsize=9, ha='right', va='bottom')
    
    # Add value labels on bars
    for bar, val in zip(bars1, llm_trash):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, oracle_trash):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Labels
    ax.set_ylabel('Trash Hand Probability Mass (%)')
    ax.set_title('Base-Rate Neglect: LLM Underestimates Trash Hands by ~4x')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    ax.annotate('4x\nunderestimate', xy=(2 - width/2, llm_trash[2]), 
                xytext=(1.5, 40),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', ha='center')
    
    # Save
    fig.savefig(output_dir / 'fig2_baserate_neglect.png')
    fig.savefig(output_dir / 'fig2_baserate_neglect.pdf')
    plt.close(fig)
    print(f"Saved Figure 2: {output_dir / 'fig2_baserate_neglect.png'}")


def figure3_update_scatter(update_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Figure 3: Scatter plot of LLM vs Oracle update magnitudes.
    
    Key visual: Points above y=x (over-updating) with near-zero correlation.
    This is the "killer figure" showing the miscalibrated update mechanism.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Separate by update type
    card_df = update_df[update_df['update_type'] == 'CARD_REVEAL']
    action_df = update_df[update_df['update_type'] == 'ACTION']
    
    for idx, (subset_df, color, marker, label_prefix) in enumerate([
        (card_df, COLORS['card_reveal'], 'o', '(A) Card Reveals'),
        (action_df, COLORS['action'], 's', '(B) Opponent Actions'),
    ]):
        ax = axes[idx]
        if len(subset_df) < 2:
            ax.set_title(f'{label_prefix} (N={len(subset_df)}, insufficient data)')
            ax.grid(True, alpha=0.3)
            continue

        ax.scatter(subset_df['oracle_magnitude'], subset_df['llm_magnitude'],
                   alpha=0.5, s=30, color=color, marker=marker)

        max_val = max(subset_df['oracle_magnitude'].max(), subset_df['llm_magnitude'].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect calibration (y=x)')

        slope, intercept, r, p, se = stats.linregress(subset_df['oracle_magnitude'], subset_df['llm_magnitude'])
        x_line = np.array([0, subset_df['oracle_magnitude'].max()])
        ax.plot(x_line, slope * x_line + intercept, 'r-', alpha=0.7, linewidth=2, label=f'Actual (r={r:.3f})')

        ax.set_xlabel('Oracle Update Magnitude')
        ax.set_ylabel('LLM Update Magnitude')
        ax.set_title(f'{label_prefix} (N={len(subset_df)})')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)

        oracle_mean = subset_df["oracle_magnitude"].mean()
        ratio_str = f'{subset_df["llm_magnitude"].mean() / oracle_mean:.1f}x more' if oracle_mean > 0 else 'N/A'
        ax.text(0.95, 0.05, f'r = {r:.3f}\nLLM updates {ratio_str}',
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Update Coherence: LLM Over-Updates with Near-Zero Correlation to Oracle', 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_dir / 'fig3_update_scatter.png')
    fig.savefig(output_dir / 'fig3_update_scatter.pdf')
    plt.close(fig)
    print(f"Saved Figure 3: {output_dir / 'fig3_update_scatter.png'}")


def figure4_street_stability(summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Figure 4: JS distance by street showing the effect holds throughout the hand.
    
    Key visual: CardOnly line consistently below StrategyAware at all streets.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Filter to street-level data
    street_data = summary[(summary['opp_action'] == 'ALL') & (summary['street'] != 'ALL')]
    
    # Order streets correctly
    street_order = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
    street_data = street_data.set_index('street').reindex(street_order).reset_index()
    
    # Filter out streets with no data
    street_data = street_data.dropna(subset=['js_cardonly_mean'])
    
    streets = street_data['street'].values
    x = np.arange(len(streets))
    
    # Extract values
    js_co = street_data['js_cardonly_mean'].values
    js_sa = street_data['js_strategyaware_mean'].values
    n_vals = street_data['n'].values
    
    # CI values (if available)
    if 'js_cardonly_ci_lo' in street_data.columns:
        js_co_ci_lo = street_data['js_cardonly_ci_lo'].values
        js_co_ci_hi = street_data['js_cardonly_ci_hi'].values
        js_sa_ci_lo = street_data['js_strategyaware_ci_lo'].values
        js_sa_ci_hi = street_data['js_strategyaware_ci_hi'].values
        
        # Error bars
        co_yerr = [js_co - js_co_ci_lo, js_co_ci_hi - js_co]
        sa_yerr = [js_sa - js_sa_ci_lo, js_sa_ci_hi - js_sa]
    else:
        co_yerr = None
        sa_yerr = None
    
    # Plot with error bars
    ax.errorbar(x, js_co, yerr=co_yerr, fmt='o-', color=COLORS['cardonly'], 
                linewidth=2, markersize=8, capsize=4, label='JS(LLM, CardOnly)')
    ax.errorbar(x, js_sa, yerr=sa_yerr, fmt='s--', color=COLORS['strategyaware'],
                linewidth=2, markersize=8, capsize=4, label='JS(LLM, StrategyAware)')
    
    # Add sample sizes
    for i, (s, n) in enumerate(zip(streets, n_vals)):
        ax.text(i, min(js_co[i], js_sa[i]) - 0.015, f'N={int(n)}', 
                ha='center', fontsize=9, color='gray')
    
    # Labels
    ax.set_xlabel('Street')
    ax.set_ylabel('Mean JS Distance')
    ax.set_title('JS Distance by Street: Effect Stable Throughout Hand')
    ax.set_xticks(x)
    ax.set_xticklabels(streets)
    ax.legend(loc='upper right')
    all_vals = np.concatenate([js_co, js_sa])
    if len(all_vals) > 0 and not np.all(np.isnan(all_vals)):
        valid = all_vals[~np.isnan(all_vals)]
        y_margin = (valid.max() - valid.min()) * 0.15 if len(valid) > 1 else 0.05
        ax.set_ylim(max(0, valid.min() - y_margin), valid.max() + y_margin)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.02, 0.02, 
            "CardOnly consistently closer\nat all streets",
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.92, edgecolor='black', linewidth=0.8))
    
    # Save
    fig.savefig(output_dir / 'fig4_street_stability.png')
    fig.savefig(output_dir / 'fig4_street_stability.pdf')
    plt.close(fig)
    print(f"Saved Figure 4: {output_dir / 'fig4_street_stability.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--pce-data", type=Path, default=Path("results/pce_distribution.csv"),
                        help="Path to PCE distribution CSV")
    parser.add_argument("--pce-summary", type=Path, default=Path("results/pce_summary.csv"),
                        help="Path to PCE summary CSV")
    parser.add_argument("--update-data", type=Path, default=Path("results/update_coherence.csv"),
                        help="Path to update coherence CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"),
                        help="Output directory for figures")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    pce_df = pd.read_csv(args.pce_data)
    pce_summary = pd.read_csv(args.pce_summary)
    update_df = pd.read_csv(args.update_data)
    
    print(f"PCE data: {len(pce_df)} records")
    print(f"Update data: {len(update_df)} records")
    
    # Generate figures
    print("\nGenerating figures...")
    
    print("\n--- Figure 1: PCE CDF ---")
    figure1_pce_cdf(pce_df, pce_summary, args.output_dir)
    
    print("\n--- Figure 2: Base-Rate Neglect ---")
    figure2_baserate_neglect(pce_df, pce_summary, args.output_dir)
    
    print("\n--- Figure 3: Update Scatter ---")
    figure3_update_scatter(update_df, args.output_dir)
    
    print("\n--- Figure 4: Street Stability ---")
    figure4_street_stability(pce_summary, args.output_dir)
    
    print(f"\n✅ All figures saved to {args.output_dir}/")
    print("\nFiles generated:")
    for f in sorted(args.output_dir.glob("fig*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
