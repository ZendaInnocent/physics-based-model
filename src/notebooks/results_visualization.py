import marimo

__generated_with = '0.21.1'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        # Results Visualization

        This notebook generates all figures and tables for the paper:
        1. Model schematic
        2. Convergence plot
        3. Sensitivity tornado plot
        4. CDF plot
        5. Risk distribution
        """
    )
    return


@app.cell
def _(mo):
    n_samples = mo.ui.number(
        value=50_000, label='Sample size', start=1000, stop=100_000
    )
    run_simulation_btn = mo.ui.run_button(label='Run Simulation')
    mo.md(f'{n_samples}\n{run_simulation_btn}')
    return n_samples, run_simulation_btn


@app.cell
def _(mo, n_samples, run_simulation_btn):
    if run_simulation_btn.value:
        from nozzle_clogging.simulation import run_simulation

        df = run_simulation(
            total_samples=n_samples.value,
            chunk_size=2_000,
            seed=42,
        )
        mo.md(f'Simulation complete with {n_samples.value} samples')
    else:
        df = None
        mo.md("Click 'Run Simulation' to start")
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""## Summary Statistics""")
    return


@app.cell
def _(df, mo):
    if df is not None:
        from nozzle_clogging.visualization import generate_results_table

        summary_stats = generate_results_table(df)
        mo.ui.table(summary_stats, label='Summary Statistics')
    else:
        summary_stats = None
        mo.md('')
    return (summary_stats,)


@app.cell
def _(mo):
    mo.md(r"""## Risk Distribution""")
    return


@app.cell
def _(df, mo):
    if df is not None:
        from nozzle_clogging.visualization import generate_risk_proportions_table

        risk_table = generate_risk_proportions_table(df)
        mo.ui.table(risk_table, label='Risk Proportions with 95% CI')
    else:
        risk_table = None
        mo.md('')
    return (risk_table,)


@app.cell
def _(df, mo):
    if df is not None:
        from nozzle_clogging.visualization import prepare_risk_distribution_data

        risk_data = prepare_risk_distribution_data(df)
        mo.md('Risk distribution data prepared')
    else:
        risk_data = None
        mo.md('')
    return (risk_data,)


@app.cell
def _(mo, risk_data):
    if risk_data:
        mo.md('**Risk Distribution:**')
        for cat, count, prop in zip(
            risk_data['categories'],
            risk_data['counts'],
            risk_data['proportions'],
        ):
            mo.md(f'- {cat}: {count} ({prop:.4f})')
    else:
        mo.md('')
    return


@app.cell
def _(mo):
    mo.md(r"""## CDF Plot Data""")
    return


@app.cell
def _(df, mo):
    if df is not None:
        from nozzle_clogging.visualization import prepare_cdf_data

        cdf_data = prepare_cdf_data(df, output='clogging_probability', n_points=100)
        mo.md('CDF data prepared')
    else:
        cdf_data = None
        mo.md('')
    return (cdf_data,)


@app.cell
def _(cdf_data, mo):
    if cdf_data:
        mo.md(f'**CDF points:** {len(cdf_data["values"])}')
        vals = cdf_data['values']
        mo.md(f'**Value range:** [{vals[0]:.4f}, {vals[-1]:.4f}]')
    else:
        mo.md('')
    return


@app.cell
def _(mo):
    mo.md(r"""## Sensitivity Analysis Results""")
    return


@app.cell
def _(df, mo):
    if df is not None:
        from nozzle_clogging.sensitivity import calculate_correlation_sensitivity

        correlation_results = calculate_correlation_sensitivity(df)
        mo.md('Correlation analysis complete')
    else:
        correlation_results = None
        mo.md('')
    return (correlation_results,)


@app.cell
def _(correlation_results, mo):
    if correlation_results is not None:
        mo.ui.table(correlation_results, label='Correlation Sensitivity')
    else:
        mo.md('')
    return


@app.cell
def _(correlation_results, mo):
    if correlation_results is not None:
        from nozzle_clogging.visualization import prepare_sensitivity_tornado_data

        tornado_data = prepare_sensitivity_tornado_data(
            correlation_results,
            output='clogging_probability',
            method='spearman',
        )
        mo.md('**Tornado Plot Data (Spearman):**')
    else:
        tornado_data = None
        mo.md('')
    return (tornado_data,)


@app.cell
def _(mo, tornado_data):
    if tornado_data:
        for inp, corr, p_val in zip(
            tornado_data['inputs'],
            tornado_data['correlations'],
            tornado_data['p_values'],
        ):
            sig = (
                '***'
                if p_val < 0.001
                else '**'
                if p_val < 0.01
                else '*'
                if p_val < 0.05
                else ''
            )
            mo.md(f'- {inp}: {corr:.4f} (p={p_val:.4f}) {sig}')
    else:
        mo.md('')
    return


@app.cell
def _(mo):
    mo.md(
        r"""## Paper Generation

The paper generation module has been removed.
The manuscript is maintained as static Markdown at `paper-writing/paper.md`
with pandoc for conversion to LaTeX/DOCX. Use `paper-writing/generate_figures.py`
to regenerate all figures from simulation results."""
    )
    return
