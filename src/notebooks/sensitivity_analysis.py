import marimo

__generated_with = '0.21.1'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns

    from nozzle_clogging.sensitivity import (
        calculate_correlation_sensitivity,
        calculate_morris_indices,
        calculate_sobol_indices,
    )
    from nozzle_clogging.simulation import run_simulation
    from nozzle_clogging.visualization import prepare_sensitivity_tornado_data

    return (
        calculate_correlation_sensitivity,
        calculate_morris_indices,
        calculate_sobol_indices,
        mo,
        plt,
        prepare_sensitivity_tornado_data,
        run_simulation,
        sns,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Sensitivity Analysis

    Three methods to rank input parameter importance:
    - **Correlation** (Pearson, Spearman) - fast, linear/monotonic
    - **Sobol indices** - variance-based, captures interactions
    - **Morris screening** - efficient ranking with few samples
    """)
    return


# ============================================================
# Correlation Analysis
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""## 1. Correlation Analysis""")
    return


@app.cell
def _(mo):
    n_samples_corr = mo.ui.slider(
        start=5_000, stop=50_000, step=5_000, value=10_000, label='Sample size'
    )
    run_corr = mo.ui.run_button(label='Run Correlation')
    mo.vstack([n_samples_corr, run_corr])
    return n_samples_corr, run_corr


@app.cell
def _(
    calculate_correlation_sensitivity,
    mo,
    n_samples_corr,
    run_corr,
    run_simulation,
):
    mo.stop(not run_corr.value, mo.md('Click **Run Correlation** above.'))

    _df = run_simulation(total_samples=n_samples_corr.value, seed=42)
    correlation_results = calculate_correlation_sensitivity(_df)
    correlation_results
    return (correlation_results,)


@app.cell
def _(correlation_results, plt):
    _data = correlation_results[
        correlation_results['output'] == 'clogging_probability'
    ].sort_values('spearman_r', key=abs, ascending=True)

    fig_tornado, axes_tornado = plt.subplots(1, 2, figsize=(14, 5))

    axes_tornado[0].barh(_data['input'], _data['spearman_r'], color='#4C72B0')
    axes_tornado[0].set_xlabel('Spearman r')
    axes_tornado[0].set_title('Spearman Correlation with P(clog)')
    axes_tornado[0].axvline(0, color='black', linewidth=0.5)

    axes_tornado[1].barh(_data['input'], _data['pearson_r'], color='#C44E52')
    axes_tornado[1].set_xlabel('Pearson r')
    axes_tornado[1].set_title('Pearson Correlation with P(clog)')
    axes_tornado[1].axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    fig_tornado
    return


# ============================================================
# Sobol Indices
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""## 2. Sobol Indices""")
    return


@app.cell
def _(mo):
    n_samples_sobol = mo.ui.slider(
        start=1_000, stop=10_000, step=1_000, value=5_000, label='Base sample size'
    )
    run_sobol = mo.ui.run_button(label='Run Sobol')
    mo.vstack([n_samples_sobol, run_sobol])
    return n_samples_sobol, run_sobol


@app.cell
def _(calculate_sobol_indices, mo, n_samples_sobol, run_sobol):
    mo.stop(not run_sobol.value, mo.md('Click **Run Sobol** above.'))

    sobol_results = calculate_sobol_indices(
        n_samples=n_samples_sobol.value, seed=42, output_col='clogging_probability'
    )
    sobol_results
    return (sobol_results,)


@app.cell
def _(plt, sobol_results):
    _x = range(len(sobol_results))
    _w = 0.35

    fig_sobol, ax_sobol = plt.subplots(figsize=(10, 5))
    ax_sobol.bar(
        [i - _w / 2 for i in _x],
        sobol_results['S1'],
        _w,
        label='S1 (First-order)',
        color='#4C72B0',
    )
    ax_sobol.bar(
        [i + _w / 2 for i in _x],
        sobol_results['ST'],
        _w,
        label='ST (Total-order)',
        color='#55A868',
    )
    ax_sobol.errorbar(
        [i - _w / 2 for i in _x],
        sobol_results['S1'],
        yerr=sobol_results['S1_conf'],
        fmt='none',
        color='black',
        capsize=3,
    )
    ax_sobol.errorbar(
        [i + _w / 2 for i in _x],
        sobol_results['ST'],
        yerr=sobol_results['ST_conf'],
        fmt='none',
        color='black',
        capsize=3,
    )
    ax_sobol.set_xticks(list(_x))
    ax_sobol.set_xticklabels(sobol_results['input'], rotation=45, ha='right')
    ax_sobol.set_ylabel('Sobol Index')
    ax_sobol.set_title('Sobol Sensitivity Indices (with 95% CI)')
    ax_sobol.legend()
    plt.tight_layout()
    fig_sobol
    return


# ============================================================
# Morris Screening
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""## 3. Morris Screening""")
    return


@app.cell
def _(mo):
    n_samples_morris = mo.ui.slider(
        start=100, stop=2_000, step=100, value=500, label='Trajectories'
    )
    run_morris = mo.ui.run_button(label='Run Morris')
    mo.vstack([n_samples_morris, run_morris])
    return n_samples_morris, run_morris


@app.cell
def _(calculate_morris_indices, mo, n_samples_morris, run_morris):
    mo.stop(not run_morris.value, mo.md('Click **Run Morris** above.'))

    morris_results = calculate_morris_indices(
        n_samples=n_samples_morris.value,
        n_levels=4,
        seed=42,
        output_col='clogging_probability',
    )
    morris_results
    return (morris_results,)


@app.cell
def _(morris_results, plt):
    fig_morris, ax_morris = plt.subplots(figsize=(8, 6))
    ax_morris.scatter(
        morris_results['mu_star'],
        morris_results['sigma'],
        s=100,
        color='#4C72B0',
    )
    for _idx, _row in morris_results.iterrows():
        ax_morris.annotate(
            _row['input'],
            (_row['mu_star'], _row['sigma']),
            textcoords='offset points',
            xytext=(5, 5),
            fontsize=9,
        )
    ax_morris.set_xlabel('mu* (mean |elementary effect|)')
    ax_morris.set_ylabel('sigma (std of elementary effects)')
    ax_morris.set_title('Morris Screening: mu* vs sigma')
    ax_morris.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax_morris.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig_morris
    return


# ============================================================
# Interpretation
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    **Tornado plot:** Parameters sorted by absolute correlation.
    Longest bars = most influential. Spearman captures monotonic
    relationships; Pearson captures linear ones. A large gap between
    Spearman and Pearson indicates nonlinear relationships.

    **Sobol indices:** S1 = fraction of variance from each parameter
    alone. ST = including interactions. If ST >> S1, the parameter
    interacts with others. S1 values should sum to ~1 if the model
    is additive.

    **Morris plot:** mu* = importance (higher = more influential).
    sigma = nonlinearity/interaction. Points far from the origin with
    high sigma are important nonlinear parameters.

    **What to report:** "Sensitivity analysis (Sobol, Morris, Spearman
    correlation) consistently identified dp/Dn ratio and particle
    diameter as the dominant factors, with S1=XX and mu*=XX
    respectively."
    """)
    return


if __name__ == '__main__':
    app.run()
