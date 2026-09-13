import marimo

__generated_with = '0.21.1'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from nozzle_clogging import config
    from nozzle_clogging.convergence import run_multiple_seeds
    from nozzle_clogging.simulation import run_simulation

    return config, mo, pd, plt, run_multiple_seeds, run_simulation


@app.cell
def _(mo):
    mo.md(r"""
    # Seed Comparison Analysis

    Assesses stability of Monte Carlo estimates across different random seeds.
    If results vary significantly by seed, the sample size is insufficient.
    """)
    return


@app.cell
def _(mo):
    n_seeds_slider = mo.ui.slider(
        start=3, stop=20, step=1, value=10, label='Number of seeds'
    )
    n_samples_slider = mo.ui.slider(
        start=10_000, stop=50_000, step=10_000, value=20_000, label='Samples per seed'
    )
    mo.vstack([n_seeds_slider, n_samples_slider])
    return n_samples_slider, n_seeds_slider


@app.cell
def _(config, n_samples_slider, n_seeds_slider, run_multiple_seeds):
    multi_df = run_multiple_seeds(
        sample_size=n_samples_slider.value,
        n_runs=n_seeds_slider.value,
        base_seed=config.RANDOM_SEED,
    )
    return (multi_df,)


@app.cell
def _(mo, multi_df):
    mo.ui.table(multi_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Per-Seed Mean Comparison
    """)
    return


@app.cell
def _(multi_df, plt):
    _outputs = ['clogging_probability', 'X', 'volume_fraction']
    _titles = ['Clogging Probability', 'X', 'Volume Fraction']

    fig_seed, axes_seed = plt.subplots(1, 3, figsize=(15, 4))
    for _i, (_output, _title) in enumerate(zip(_outputs, _titles)):
        _data = multi_df[multi_df['output'] == _output]
        axes_seed[_i].bar(
            range(len(_data)),
            _data['mean'],
            yerr=(_data['mean'] - _data['ci_lower'], _data['ci_upper'] - _data['mean']),
            capsize=4,
            color='#4C72B0',
            alpha=0.7,
            edgecolor='black',
        )
        axes_seed[_i].axhline(
            _data['mean'].mean(), color='red', linestyle='--', label='Grand mean'
        )
        axes_seed[_i].set_xlabel('Seed (run)')
        axes_seed[_i].set_ylabel('Mean +/- 95% CI')
        axes_seed[_i].set_title(_title)
        axes_seed[_i].legend(fontsize=8)

    fig_seed.suptitle('Per-Seed Mean Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_seed
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Between-Run vs Within-Run Variance
    """)
    return


@app.cell
def _(mo, multi_df):
    _rows = []
    for _output in ['clogging_probability', 'X', 'volume_fraction']:
        _data = multi_df[multi_df['output'] == _output]
        _grand_mean = _data['mean'].mean()
        _between_std = _data['mean'].std(ddof=1)
        _between_cv = _between_std / _grand_mean if _grand_mean != 0 else 0
        _within_std = _data['std'].mean()
        _ratio = _between_std / _within_std if _within_std > 0 else 0

        _rows.append(
            f'| {_output} | {_grand_mean:.6f} | {_between_std:.6f} | '
            f'{_between_cv:.2%} | {_within_std:.6f} | {_ratio:.4f} | '
            f'{"Yes" if _between_cv < 0.02 else "Check"} |'
        )

    mo.md(
        '| Output | Grand Mean | Between Std | CV | Within Std | Ratio | Stable |\n'
        '|--------|-----------|------------|-----|-----------|-------|--------|\n'
        + '\n'.join(_rows)
        + '\n\n**Ratio** = between-run std / within-run std. < 0.1 = stable.\n'
        '**CV** < 2% confirms sufficient sample size.'
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Risk Proportion Stability
    """)
    return


@app.cell
def _(config, multi_df, pd, run_simulation):
    _risk_rows = []
    for _seed_offset in range(multi_df['run'].nunique()):
        _seed = config.RANDOM_SEED + _seed_offset
        _df = run_simulation(total_samples=10_000, seed=_seed)
        _counts = _df['clogging_risk'].value_counts()
        _n = len(_df)
        for _level in config.RISK_LEVELS:
            _risk_rows.append(
                {
                    'seed': _seed,
                    'risk': _level,
                    'proportion': _counts.get(_level, 0) / _n,
                }
            )
    risk_seed_df = pd.DataFrame(_risk_rows)
    return (risk_seed_df,)


@app.cell
def _(config, plt, risk_seed_df):
    _colors = {'Low': '#55A868', 'Moderate': '#F2C94C', 'High': '#C44E52'}

    fig_risk, axes_risk = plt.subplots(1, 3, figsize=(15, 4))
    for _i, _level in enumerate(config.RISK_LEVELS):
        _data = risk_seed_df[risk_seed_df['risk'] == _level]
        axes_risk[_i].bar(
            range(len(_data)),
            _data['proportion'],
            color=_colors[_level],
            edgecolor='black',
        )
        axes_risk[_i].axhline(_data['proportion'].mean(), color='red', linestyle='--')
        axes_risk[_i].set_ylim(0, 1)
        axes_risk[_i].set_xlabel('Seed')
        axes_risk[_i].set_ylabel('Proportion')
        axes_risk[_i].set_title(f'{_level} Risk')

        _cv = _data['proportion'].std() / _data['proportion'].mean()
        axes_risk[_i].text(
            0.95,
            0.95,
            f'CV={_cv:.1%}',
            transform=axes_risk[_i].transAxes,
            ha='right',
            va='top',
            fontsize=9,
        )

    fig_risk.suptitle('Risk Proportion Stability Across Seeds', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_risk
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    **Per-seed means:** Error bars (95% CI) should overlap across all seeds.
    If they don't, the sample size is too small or the distribution is
    heavy-tailed.

    **Between/within ratio:** Ratio < 0.1 means between-run variance is
    negligible compared to within-run variance. The MC estimate is stable.

    **Risk proportions:** CV (coefficient of variation) across seeds should
    be < 5% for each risk category. Higher CV means sample size is
    insufficient for reliable proportion estimates.

    **Paper sentence:** "N=XX,000 was selected based on convergence analysis.
    Stability was verified across N independent seeds with between-run
    CV < 2%."
    """)
    return


if __name__ == '__main__':
    app.run()
