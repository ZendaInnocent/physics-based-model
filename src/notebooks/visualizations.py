# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy>=1.26.0",
#     "pandas>=2.2.0",
#     "scipy>=1.13.0",
#     "SALib>=1.5.0",
#     "matplotlib>=3.9.0",
#     "seaborn>=0.13.0",
#     "pyarrow>=23.0.1",
# ]
# [tool.ruff.format]
# quote-style = "single"
# ///

import marimo

__generated_with = '0.20.4'
app = marimo.App(width='full', app_title='Visualizations: Paper Figures')


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Publication-Quality Visualizations

    This notebook generates all figures for the paper and supplementary materials.

    Figures:
    - **Figure 1**: Clogging probability distribution
    - **Figure 2**: Sensitivity analysis (Sobol' indices + SRC)
    - **Figure 3**: Risk factor contributions
    - **Figure 4**: Sediment class comparison
    """)
    return (mo,)


@app.cell
def imports():
    import matplotlib
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    from notebooks.utils.simulations import simulate_nozzle_clogging

    matplotlib.use('Agg')
    warnings.filterwarnings('ignore')
    sns.set_style('whitegrid')

    plt.rcParams.update(
        {
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.family': 'sans-serif',
        }
    )
    return (
        LinearRegression,
        Path,
        StandardScaler,
        np,
        pd,
        plt,
        simulate_nozzle_clogging,
        sns,
        stats,
    )
    return (
        LinearRegression,
        Path,
        StandardScaler,
        np,
        pd,
        plt,
        simulate_nozzle_clogging,
        sns,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Simulation Data
    """)


@app.cell
def load_data(simulate_nozzle_clogging):
    """Run simulation to generate data."""
    df = simulate_nozzle_clogging(total_samples=20_000, chunk_size=2_000, seed=42)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1: Clogging Probability Distribution

    Five-panel figure showing the distribution of clogging probability from Monte Carlo simulation.
    """)


@app.cell
def figure1_clogging_distribution(df, mo, np, pd, plt, sns):
    """Generate Figure 1: Clogging probability distribution."""
    fig = plt.figure(figsize=(16, 10))

    # Panel (a): Histogram with thresholds
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(
        df['clogging_probability'],
        bins=50,
        edgecolor='black',
        alpha=0.7,
        color='skyblue',
    )
    ax1.axvline(
        0.30, color='orange', linestyle='--', linewidth=2, label='Low/Moderate (0.30)'
    )
    ax1.axvline(
        0.50, color='red', linestyle='--', linewidth=2, label='Moderate/High (0.50)'
    )
    ax1.set_xlabel('Clogging Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Histogram with Risk Thresholds')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel (b): KDE by risk level
    ax2 = fig.add_subplot(2, 3, 2)
    risk_colors = {'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'}
    for risk in ['Low', 'Moderate', 'High']:
        subset = df[df['clogging_risk'] == risk]['clogging_probability']
        if len(subset) > 0:
            sns.kdeplot(
                subset,
                ax=ax2,
                color=risk_colors[risk],
                label=risk,
                fill=True,
                alpha=0.3,
            )
    ax2.set_xlabel('Clogging Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) KDE by Risk Level')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel (c): Mean by sediment class with 95% CI
    ax3 = fig.add_subplot(2, 3, 3)
    class_stats = df.groupby('particle_size_range')['clogging_probability'].agg(
        ['mean', 'std', 'count']
    )
    class_stats['ci95'] = 1.96 * class_stats['std'] / np.sqrt(class_stats['count'])
    classes = ['Fine', 'Medium', 'Coarse']
    means = [class_stats.loc[c, 'mean'] for c in classes]
    cis = [class_stats.loc[c, 'ci95'] for c in classes]
    ax3.bar(
        classes, means, yerr=cis, capsize=5, color=['#3498db', '#2ecc71', '#e74c3c']
    )
    ax3.set_ylabel('Mean Clogging Probability')
    ax3.set_title('(c) Mean by Sediment Class')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel (d): Box plot by sediment class
    ax4 = fig.add_subplot(2, 3, 4)
    sns.boxplot(
        data=df,
        x='particle_size_range',
        y='clogging_probability',
        ax=ax4,
        palette='Set2',
    )
    ax4.set_xlabel('Sediment Class')
    ax4.set_ylabel('Clogging Probability')
    ax4.set_title('(d) Distribution by Sediment Class')
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel (e): CDF
    ax5 = fig.add_subplot(2, 3, 5)
    sorted_probs = np.sort(df['clogging_probability'])
    cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax5.plot(sorted_probs, cdf, linewidth=2, color='steelblue')
    ax5.axvline(0.30, color='orange', linestyle='--', alpha=0.7)
    ax5.axvline(0.50, color='red', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Clogging Probability')
    ax5.set_ylabel('Cumulative Probability')
    ax5.set_title('(e) Cumulative Distribution Function')
    ax5.grid(True, alpha=0.3)

    # Panel (f): Risk category proportions
    ax6 = fig.add_subplot(2, 3, 6)
    risk_counts = df['clogging_risk'].value_counts()
    risk_order = ['Low', 'Moderate', 'High']
    risk_counts = risk_counts.reindex(risk_order)
    colors = [risk_colors[r] for r in risk_order]
    ax6.bar(risk_order, risk_counts.values, color=colors, edgecolor='black')
    for i, (risk, count) in enumerate(risk_counts.items()):
        pct = count / len(df) * 100
        ax6.text(i, count + 100, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    ax6.set_ylabel('Count')
    ax6.set_title('(f) Risk Category Distribution')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    mo.md(
        '**Figure S4:** Clogging probability distribution from 20,000 Monte Carlo samples.'
    )
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2: Sensitivity Analysis

    Sobol' sensitivity indices and standardized regression coefficients.
    """)


@app.cell
def figure2_sensitivity(df, mo, np, pd, plt, sns):
    """Generate Figure 2: Sensitivity analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Standardized Regression Coefficients
    features = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
    ]
    X = df[features].values
    y = df['clogging_probability'].values

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)

    src = model.coef_
    feature_labels = [
        'TSS',
        'Pressure',
        'Nozzle Diameter',
        'Duration',
        'Particle Diameter',
    ]

    ax = axes[0]
    colors = ['red' if s > 0 else 'blue' for s in src]
    ax.barh(feature_labels, src, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Standardized Regression Coefficient')
    ax.set_title('(a) Standardized Regression Coefficients')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    mo.md(
        '**Figure S5:** Sensitivity analysis of clogging probability to input parameters.'
    )
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3: Sediment Class Comparison

    Detailed comparison of clogging behaviour across sediment size classes.
    """)


@app.cell
def figure3_sediment_comparison(df, mo, np, pd, plt, sns):
    """Generate Figure 3: Sediment class comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Violin plot by class
    ax = axes[0, 0]
    sns.violinplot(
        data=df,
        x='particle_size_range',
        y='clogging_probability',
        ax=ax,
        palette='Set2',
    )
    ax.set_xlabel('Sediment Class')
    ax.set_ylabel('Clogging Probability')
    ax.set_title('(a) Distribution by Sediment Class')
    ax.grid(True, alpha=0.3, axis='y')

    # Particle diameter vs probability scatter
    ax = axes[0, 1]
    sample = df.sample(n=2000, random_state=42)
    for cls in ['Fine', 'Medium', 'Coarse']:
        subset = sample[sample['particle_size_range'] == cls]
        ax.scatter(
            subset['particle_diameter_um'],
            subset['clogging_probability'],
            alpha=0.3,
            label=cls,
            s=10,
        )
    ax.set_xscale('log')
    ax.set_xlabel('Particle Diameter (um)')
    ax.set_ylabel('Clogging Probability')
    ax.set_title('(b) Particle Size vs Clogging Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Risk distribution by class
    ax = axes[1, 0]
    risk_by_class = (
        df.groupby(['particle_size_range', 'clogging_risk'])
        .size()
        .unstack(fill_value=0)
    )
    risk_by_class = risk_by_class.div(risk_by_class.sum(axis=1), axis=0)
    risk_by_class[['Low', 'Moderate', 'High']].plot(
        kind='bar', ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c']
    )
    ax.set_xlabel('Sediment Class')
    ax.set_ylabel('Proportion')
    ax.set_title('(c) Risk Distribution by Class')
    ax.legend(title='Risk Level')
    ax.grid(True, alpha=0.3, axis='y')

    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_table = df.groupby('particle_size_range')['clogging_probability'].agg(
        ['mean', 'std', 'median', 'count']
    )
    stats_table.columns = ['Mean', 'Std', 'Median', 'Count']
    stats_table = stats_table.round(3)
    table = ax.table(
        cellText=stats_table.values,
        colLabels=stats_table.columns,
        rowLabels=stats_table.index,
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title('(d) Summary Statistics')

    plt.tight_layout()
    mo.md('**Figure S6:** Sediment class comparison and clogging behaviour analysis.')
    return (fig,)


if __name__ == '__main__':
    app.run()
