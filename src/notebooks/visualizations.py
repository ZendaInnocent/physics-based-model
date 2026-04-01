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

__generated_with = "0.21.1"
app = marimo.App(width="full", app_title="Visualizations: Paper Figures")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizations

    Figures:
    - **Figure 1**: Clogging probability distribution
    - **Figure 2**: Sensitivity analysis (Sobol' indices + SRC)
    - **Figure 3**: Risk factor contributions
    - **Figure 4**: Sediment class comparison
    """)
    return


@app.cell
def imports():
    import warnings

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    from nozzle_clogging.simulations import simulate_nozzle_clogging

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")
    sns.set_style("whitegrid")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
        }
    )
    return (
        LinearRegression,
        StandardScaler,
        np,
        plt,
        simulate_nozzle_clogging,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Simulation Data
    """)
    return


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
    return


@app.cell
def figure1_clogging_distribution(df, mo, np, plt, sns):
    """Generate Figure 1: Clogging probability distribution."""

    _fig1 = plt.figure(figsize=(16, 10))

    # Panel (a): Histogram with thresholds
    _ax1 = _fig1.add_subplot(2, 3, 1)
    _ax1.hist(
        df["clogging_probability"],
        bins=50,
        edgecolor="black",
        alpha=0.7,
        color="skyblue",
    )
    _ax1.axvline(
        0.30,
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Low/Moderate (0.30)",
    )
    _ax1.axvline(
        0.50,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Moderate/High (0.50)",
    )
    _ax1.set_xlabel("Clogging Probability")
    _ax1.set_ylabel("Frequency")
    _ax1.set_title("(a) Histogram with Risk Thresholds")
    _ax1.legend(fontsize=8)
    _ax1.grid(True, alpha=0.3)

    # Panel (b): KDE by risk level
    _ax2 = _fig1.add_subplot(2, 3, 2)
    _risk_colors = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}
    for risk in ["Low", "Moderate", "High"]:
        _subset1 = df[df["clogging_risk"] == risk]["clogging_probability"]
        if len(_subset1) > 0:
            sns.kdeplot(
                _subset1,
                ax=_ax2,
                color=_risk_colors[risk],
                label=risk,
                fill=True,
                alpha=0.3,
            )
    _ax2.set_xlabel("Clogging Probability")
    _ax2.set_ylabel("Density")
    _ax2.set_title("(b) KDE by Risk Level")
    _ax2.legend(fontsize=8)
    _ax2.grid(True, alpha=0.3)

    # Panel (c): Mean by sediment class with 95% CI
    _ax3 = _fig1.add_subplot(2, 3, 3)
    _class_stats = df.groupby("particle_size_range")["clogging_probability"].agg(
        ["mean", "std", "count"]
    )
    _class_stats["ci95"] = 1.96 * _class_stats["std"] / np.sqrt(_class_stats["count"])
    _classes = ["Fine", "Medium", "Coarse"]
    _means = [_class_stats.loc[c, "mean"] for c in _classes]
    _cis = [_class_stats.loc[c, "ci95"] for c in _classes]
    _ax3.bar(
        _classes,
        _means,
        yerr=_cis,
        capsize=5,
        color=["#3498db", "#2ecc71", "#e74c3c"],
    )
    _ax3.set_ylabel("Mean Clogging Probability")
    _ax3.set_title("(c) Mean by Sediment Class")
    _ax3.grid(True, alpha=0.3, axis="y")

    # Panel (d): Box plot by sediment class
    _ax4 = _fig1.add_subplot(2, 3, 4)
    sns.boxplot(
        data=df,
        x="particle_size_range",
        y="clogging_probability",
        ax=_ax4,
        palette="Set2",
    )
    _ax4.set_xlabel("Sediment Class")
    _ax4.set_ylabel("Clogging Probability")
    _ax4.set_title("(d) Distribution by Sediment Class")
    _ax4.grid(True, alpha=0.3, axis="y")

    # Panel (e): CDF
    _ax5 = _fig1.add_subplot(2, 3, 5)
    _sorted_probs = np.sort(df["clogging_probability"])
    _cdf = np.arange(1, len(_sorted_probs) + 1) / len(_sorted_probs)
    _ax5.plot(_sorted_probs, _cdf, linewidth=2, color="steelblue")
    _ax5.axvline(0.30, color="orange", linestyle="--", alpha=0.7)
    _ax5.axvline(0.50, color="red", linestyle="--", alpha=0.7)
    _ax5.set_xlabel("Clogging Probability")
    _ax5.set_ylabel("Cumulative Probability")
    _ax5.set_title("(e) Cumulative Distribution Function")
    _ax5.grid(True, alpha=0.3)

    # Panel (f): Risk category proportions
    _ax6 = _fig1.add_subplot(2, 3, 6)
    _risk_counts = df["clogging_risk"].value_counts()
    _risk_order = ["Low", "Moderate", "High"]
    _risk_counts = _risk_counts.reindex(_risk_order)
    _colors1 = [_risk_colors[r] for r in _risk_order]
    _ax6.bar(_risk_order, _risk_counts.values, color=_colors1, edgecolor="black")
    for i, (risk, count) in enumerate(_risk_counts.items()):
        pct = count / len(df) * 100
        _ax6.text(i, count + 100, f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    _ax6.set_ylabel("Count")
    _ax6.set_title("(f) Risk Category Distribution")
    _ax6.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    mo.vstack(
        [
            mo.md("Clogging probability distribution from 20,000 Monte Carlo samples."),
            _fig1,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2: Sensitivity Analysis

    Sobol' sensitivity indices and standardized regression coefficients.
    """)
    return


@app.cell
def figure2_sensitivity(LinearRegression, StandardScaler, df, mo, plt):
    """Generate Figure 2: Sensitivity analysis."""

    _fig, _axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Standardized Regression Coefficients
    _features = [
        "TSS_mg_L",
        "pressure_kPa",
        "nozzle_diameter_mm",
        "duration_hrs",
        "particle_diameter_um",
    ]
    _X = df[_features].values
    _y = df["clogging_probability"].values

    _scaler = StandardScaler()
    _X_scaled = _scaler.fit_transform(_X)
    _model = LinearRegression()
    _model.fit(_X_scaled, _y)

    _src = _model.coef_
    _feature_labels = [
        "TSS",
        "Pressure",
        "Nozzle Diameter",
        "Duration",
        "Particle Diameter",
    ]

    _ax2a = _axes2[0]
    _colors2 = ["red" if s > 0 else "blue" for s in _src]
    _ax2a.barh(_feature_labels, _src, color=_colors2, alpha=0.7)
    _ax2a.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    _ax2a.set_xlabel("Standardized Regression Coefficient")
    _ax2a.set_title("(a) Standardized Regression Coefficients")
    _ax2a.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    mo.vstack(
        [
            mo.md("Sensitivity analysis of clogging probability to input parameters."),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3: Sediment Class Comparison

    Detailed comparison of clogging behaviour across sediment size classes.
    """)
    return


@app.cell
def figure3_sediment_comparison(df, mo, plt, sns):
    """Generate Figure 3: Sediment class comparison."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Violin plot by class
    ax = axes[0, 0]
    sns.violinplot(
        data=df,
        x="particle_size_range",
        y="clogging_probability",
        ax=ax,
        palette="Set2",
    )
    ax.set_xlabel("Sediment Class")
    ax.set_ylabel("Clogging Probability")
    ax.set_title("(a) Distribution by Sediment Class")
    ax.grid(True, alpha=0.3, axis="y")

    # Particle diameter vs probability scatter
    ax = axes[0, 1]
    sample = df.sample(n=2000, random_state=42)
    for cls in ["Fine", "Medium", "Coarse"]:
        subset = sample[sample["particle_size_range"] == cls]
        ax.scatter(
            subset["particle_diameter_um"],
            subset["clogging_probability"],
            alpha=0.3,
            label=cls,
            s=10,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Particle Diameter (um)")
    ax.set_ylabel("Clogging Probability")
    ax.set_title("(b) Particle Size vs Clogging Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Risk distribution by class
    ax = axes[1, 0]
    risk_by_class = (
        df.groupby(["particle_size_range", "clogging_risk"])
        .size()
        .unstack(fill_value=0)
    )
    risk_by_class = risk_by_class.div(risk_by_class.sum(axis=1), axis=0)
    risk_by_class[["Low", "Moderate", "High"]].plot(
        kind="bar", ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"]
    )
    ax.set_xlabel("Sediment Class")
    ax.set_ylabel("Proportion")
    ax.set_title("(c) Risk Distribution by Class")
    ax.legend(title="Risk Level")
    ax.grid(True, alpha=0.3, axis="y")

    # Summary statistics
    ax = axes[1, 1]
    ax.axis("off")
    stats_table = df.groupby("particle_size_range")["clogging_probability"].agg(
        ["mean", "std", "median", "count"]
    )
    stats_table.columns = ["Mean", "Std", "Median", "Count"]
    stats_table = stats_table.round(3)
    table = ax.table(
        cellText=stats_table.values,
        colLabels=stats_table.columns,
        rowLabels=stats_table.index,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title("(d) Summary Statistics")

    plt.tight_layout()
    mo.vstack(
        [mo.md("Sediment class comparison and clogging behaviour analysis."), fig]
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
