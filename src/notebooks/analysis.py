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
app = marimo.App(
    width="full",
    app_title="Analysis: Sensitivity and Convergence",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Supplementary Analysis: Sensitivity and Convergence

    This notebook provides supplementary analysis supporting the main paper:

    1. **Monte Carlo Convergence** - Variance stabilisation analysis
    2. **Sobol' Sensitivity Analysis** - First-order and total-effect indices
    3. **Mono- vs Polydisperse Comparison** - Bias quantification
    """)
    return (mo,)


@app.cell
def imports():
    import warnings

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    from nozzle_clogging.config import PARTICLE_SIZE_RANGES
    from nozzle_clogging.simulations import (
        compute_clogging,
        compute_physics,
        generate_lhs_samples,
        simulate_nozzle_clogging,
    )

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")
    sns.set_style("whitegrid")
    return (
        PARTICLE_SIZE_RANGES,
        compute_clogging,
        compute_physics,
        generate_lhs_samples,
        np,
        pd,
        plt,
        simulate_nozzle_clogging,
        sobol_analyze,
        sobol_sample,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Monte Carlo Convergence Analysis

    This section examines how the estimated statistics stabilise as the number of samples increases.
    """)
    return


@app.cell
def run_convergence_analysis(pd, simulate_nozzle_clogging):
    """Run convergence analysis at increasing sample sizes."""
    sample_sizes = [1000, 2000, 5000, 10000, 15000, 20000]
    _conv_results = []

    for n in sample_sizes:
        df = simulate_nozzle_clogging(total_samples=n, chunk_size=1000, seed=42)
        _conv_results.append(
            {
                "n_samples": n,
                "mean_pc": df["clogging_probability"].mean(),
                "std_pc": df["clogging_probability"].std(),
                "median_pc": df["clogging_probability"].median(),
                "p95_pc": df["clogging_probability"].quantile(0.95),
                "low_pct": (df["clogging_risk"] == "Low").mean(),
                "moderate_pct": (df["clogging_risk"] == "Moderate").mean(),
                "high_pct": (df["clogging_risk"] == "High").mean(),
            }
        )

    convergence_df = pd.DataFrame(_conv_results)
    return (convergence_df,)


@app.cell
def plot_convergence(convergence_df, mo, plt):
    """Plot convergence of key statistics."""

    _fig, _axes = plt.subplots(2, 2, figsize=(12, 8))

    def _plot_panel(_a, _x, _y, _title, _xlabel, _ylabel, _hline=None, _label=None):
        _a.plot(_x, _y, "o-", linewidth=2)
        if _hline is not None:
            _a.axhline(_hline, color="red", linestyle="--", alpha=0.5, label=_label)
        _a.set_xlabel(_xlabel)
        _a.set_ylabel(_ylabel)
        _a.set_title(_title)
        _a.legend()
        _a.grid(True, alpha=0.3)

    _plot_panel(
        _axes[0, 0],
        convergence_df["n_samples"],
        convergence_df["mean_pc"],
        "Mean Convergence",
        "Number of Samples",
        "Mean Clogging Probability",
        convergence_df.iloc[-1]["mean_pc"],
        f"Final: {convergence_df.iloc[-1]['mean_pc']:.3f}",
    )
    _plot_panel(
        _axes[0, 1],
        convergence_df["n_samples"],
        convergence_df["std_pc"],
        "Std Dev Convergence",
        "Number of Samples",
        "Standard Deviation",
        convergence_df.iloc[-1]["std_pc"],
        f"Final: {convergence_df.iloc[-1]['std_pc']:.3f}",
    )

    _a = _axes[1, 0]
    _a.plot(
        convergence_df["n_samples"],
        convergence_df["low_pct"],
        "o-",
        label="Low",
        linewidth=2,
    )
    _a.plot(
        convergence_df["n_samples"],
        convergence_df["moderate_pct"],
        "s-",
        label="Moderate",
        linewidth=2,
    )
    _a.plot(
        convergence_df["n_samples"],
        convergence_df["high_pct"],
        "^-",
        label="High",
        linewidth=2,
    )
    _a.set_xlabel("Number of Samples")
    _a.set_ylabel("Proportion")
    _a.set_title("Risk Category Convergence")
    _a.legend()
    _a.grid(True, alpha=0.3)

    _plot_panel(
        _axes[1, 1],
        convergence_df["n_samples"],
        convergence_df["p95_pc"],
        "Tail Estimate Convergence",
        "Number of Samples",
        "95th Percentile",
        convergence_df.iloc[-1]["p95_pc"],
        f"Final: {convergence_df.iloc[-1]['p95_pc']:.3f}",
    )

    plt.tight_layout()
    mo.vstack(
        [
            mo.md("Convergence of key statistics with increasing sample size."),
            _fig,
        ]
    )
    return


@app.cell
def display_convergence_table(convergence_df, mo):
    """Display convergence results as a table."""
    display_df = convergence_df.copy()
    for _c in ["low_pct", "moderate_pct", "high_pct"]:
        display_df[_c] = (display_df[_c] * 100).round(1).astype(str) + "%"
    for _c in ["mean_pc", "std_pc", "median_pc", "p95_pc"]:
        display_df[_c] = display_df[_c].round(3)
    mo.ui.table(display_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sobol' Sensitivity Analysis

    Global sensitivity analysis using Sobol' indices to decompose output variance into contributions from individual parameters and their interactions.
    """)
    return


@app.cell
def run_sobol_analysis(
    compute_clogging,
    compute_physics,
    np,
    pd,
    sobol_analyze,
    sobol_sample,
):
    """Run Sobol' sensitivity analysis."""
    problem = {
        "num_vars": 5,
        "names": ["TSS", "Pressure", "NozzleDiameter", "Duration", "ParticleDiameter"],
        "bounds": [
            [10, 500],
            [100, 400],
            [1.5, 6.0],
            [1, 12],
            [10, 300],
        ],
    }

    param_values = sobol_sample.sample(problem, 1024, calc_second_order=True)

    _sobol_results = []
    chunk_size = 256
    for i in range(0, len(param_values), chunk_size):
        chunk = param_values[i : i + chunk_size]
        df_chunk = pd.DataFrame(
            chunk,
            columns=[
                "TSS_mg_L",
                "pressure_kPa",
                "nozzle_diameter_mm",
                "duration_hrs",
                "particle_diameter_um",
            ],
        )
        df_chunk["velocity_m_s"] = 0.85 * np.sqrt(
            (2 * df_chunk["pressure_kPa"].values * 1000) / 1000
        )
        df_chunk = compute_physics(df_chunk)
        df_chunk = compute_clogging(df_chunk)
        _sobol_results.append(df_chunk["clogging_probability"].values)

    Y = np.concatenate(_sobol_results)

    Si = sobol_analyze.analyze(
        problem, Y, calc_second_order=True, print_to_console=False
    )
    return Si, problem


@app.cell
def plot_sobol_indices(Si, mo, np, plt, problem):
    """Plot Sobol' sensitivity indices."""

    names = problem["names"]
    S1 = Si["S1"]
    ST = Si["ST"]
    S1_conf = Si["S1_conf"]
    ST_conf = Si["ST_conf"]

    _fig, _ax = plt.subplots(figsize=(10, 6))

    _x = np.arange(len(names))
    _w = 0.35

    _ax.bar(
        _x - _w / 2,
        S1,
        _w,
        label="First-order (S1)",
        color="steelblue",
        yerr=S1_conf,
        capsize=5,
    )
    _ax.bar(
        _x + _w / 2,
        ST,
        _w,
        label="Total-effect (ST)",
        color="coral",
        yerr=ST_conf,
        capsize=5,
    )

    _ax.set_ylabel("Sensitivity Index")
    _ax.set_title("Sobol' Sensitivity Indices for Clogging Probability")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(names, rotation=15)
    _ax.legend()
    _ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    mo.vstack(
        [
            mo.md(
                "Sobol' sensitivity indices. First-order (S1) measures direct effect; "
                "total-effect (ST) includes interactions. The gap between S1 and ST indicates interaction effects."
            ),
            _fig,
        ]
    )
    return


@app.cell
def display_sobol_table(Si, mo, np, pd, problem):
    """Display Sobol' indices as a table."""
    sobol_df = pd.DataFrame(
        {
            "Parameter": problem["names"],
            "S1 (First-order)": np.round(Si["S1"], 4),
            "S1_conf": np.round(Si["S1_conf"], 4),
            "ST (Total-effect)": np.round(Si["ST"], 4),
            "ST_conf": np.round(Si["ST_conf"], 4),
        }
    )
    mo.ui.table(sobol_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mono- vs Polydisperse Comparison

    Quantifies the bias introduced by using mono-disperse (median) particle sizes instead of full lognormal distributions.
    """)
    return


@app.cell
def run_polydisperse_comparison(
    PARTICLE_SIZE_RANGES,
    compute_clogging,
    compute_physics,
    generate_lhs_samples,
    np,
    pd,
    simulate_nozzle_clogging,
):
    """Compare mono-disperse and polydisperse simulations."""
    N_SAMPLES = 20_000
    SEED = 42

    _df_poly = simulate_nozzle_clogging(
        total_samples=N_SAMPLES, chunk_size=2000, seed=SEED
    )

    median_sizes = {
        name: np.sqrt(ps_min * ps_max)
        for name, (ps_min, ps_max) in PARTICLE_SIZE_RANGES.items()
    }

    _results_mono = []
    for class_name, median_dp in median_sizes.items():
        (TSS, pressure, nozzle_diam, duration, _, _) = generate_lhs_samples(
            N_SAMPLES // 3, SEED
        )
        _df_m = pd.DataFrame(
            {
                "TSS_mg_L": TSS,
                "pressure_kPa": pressure,
                "velocity_m_s": 0.85 * np.sqrt((2 * pressure * 1000) / 1000),
                "nozzle_diameter_mm": nozzle_diam,
                "duration_hrs": duration,
                "particle_diameter_um": np.full(len(TSS), median_dp),
                "particle_size_range": np.full(len(TSS), class_name),
            }
        )
        _df_m = compute_physics(_df_m)
        _df_m = compute_clogging(_df_m)
        _results_mono.append(_df_m)

    _df_mono_all = pd.concat(_results_mono, ignore_index=True)

    _bias = (
        _df_mono_all["clogging_probability"].mean()
        - _df_poly["clogging_probability"].mean()
    )
    _high_risk_mono = (_df_mono_all["clogging_risk"] == "High").mean()
    _high_risk_poly = (_df_poly["clogging_risk"] == "High").mean()

    comparison = {
        "df_poly": _df_poly,
        "df_mono": _df_mono_all,
        "bias": _bias,
        "high_risk_mono": _high_risk_mono,
        "high_risk_poly": _high_risk_poly,
    }
    return (comparison,)


@app.cell
def plot_polydisperse_comparison(comparison, mo, np, plt):
    """Plot mono vs polydisperse comparison."""

    _df_p = comparison["df_poly"]
    _df_m = comparison["df_mono"]

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))

    _a = _axes[0]
    _a.hist(
        _df_p["clogging_probability"],
        bins=50,
        alpha=0.5,
        label="Polydisperse",
        density=True,
    )
    _a.hist(
        _df_m["clogging_probability"],
        bins=50,
        alpha=0.5,
        label="Mono-disperse",
        density=True,
    )
    _a.set_xlabel("Clogging Probability")
    _a.set_ylabel("Density")
    _a.set_title("Probability Distribution Comparison")
    _a.legend()
    _a.grid(True, alpha=0.3)

    _a = _axes[1]
    _classes = ["Fine", "Medium", "Coarse"]
    _poly_means = [
        _df_p[_df_p["particle_size_range"] == c]["clogging_probability"].mean()
        for c in _classes
    ]
    _mono_means = [
        _df_m[_df_m["particle_size_range"] == c]["clogging_probability"].mean()
        for c in _classes
    ]
    _x = np.arange(len(_classes))
    _w = 0.35
    _a.bar(_x - _w / 2, _poly_means, _w, label="Polydisperse", color="steelblue")
    _a.bar(_x + _w / 2, _mono_means, _w, label="Mono-disperse", color="coral")
    _a.set_xticks(_x)
    _a.set_xticklabels(_classes)
    _a.set_ylabel("Mean Clogging Probability")
    _a.set_title("Mean Probability by Particle Class")
    _a.legend()
    _a.grid(True, alpha=0.3, axis="y")

    _a = _axes[2]
    _risk_cats = ["Low", "Moderate", "High"]
    _poly_risk = [(_df_p["clogging_risk"] == r).mean() for r in _risk_cats]
    _mono_risk = [(_df_m["clogging_risk"] == r).mean() for r in _risk_cats]
    _x2 = np.arange(len(_risk_cats))
    _a.bar(_x2 - _w / 2, _poly_risk, _w, label="Polydisperse", color="steelblue")
    _a.bar(_x2 + _w / 2, _mono_risk, _w, label="Mono-disperse", color="coral")
    _a.set_xticks(_x2)
    _a.set_xticklabels(_risk_cats)
    _a.set_ylabel("Proportion")
    _a.set_title("Risk Category Distribution")
    _a.legend()
    _a.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"**Mono- vs polydisperse comparison. "
                f"Bias in mean probability: {comparison['bias']:.4f}. "
                f"High risk: mono={comparison['high_risk_mono']:.1%}, poly={comparison['high_risk_poly']:.1%}."
            ),
            _fig,
        ]
    )
    return


@app.cell
def display_comparison_summary(comparison, mo, pd):
    """Display comparison summary statistics."""
    summary = pd.DataFrame(
        {
            "Metric": [
                "Mean Probability",
                "Std Deviation",
                "High Risk %",
                "Moderate Risk %",
                "Low Risk %",
            ],
            "Polydisperse": [
                comparison["df_poly"]["clogging_probability"].mean(),
                comparison["df_poly"]["clogging_probability"].std(),
                (comparison["df_poly"]["clogging_risk"] == "High").mean(),
                (comparison["df_poly"]["clogging_risk"] == "Moderate").mean(),
                (comparison["df_poly"]["clogging_risk"] == "Low").mean(),
            ],
            "Mono-disperse": [
                comparison["df_mono"]["clogging_probability"].mean(),
                comparison["df_mono"]["clogging_probability"].std(),
                (comparison["df_mono"]["clogging_risk"] == "High").mean(),
                (comparison["df_mono"]["clogging_risk"] == "Moderate").mean(),
                (comparison["df_mono"]["clogging_risk"] == "Low").mean(),
            ],
        }
    )
    for _col in ["Polydisperse", "Mono-disperse"]:
        summary[_col] = summary[_col].apply(
            lambda x: f"{x:.3f}" if x < 1 else f"{x:.1%}"
        )
    mo.ui.table(summary)
    return


if __name__ == "__main__":
    app.run()
