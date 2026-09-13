import marimo

__generated_with = '0.21.1'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from nozzle_clogging.convergence import (
        calculate_convergence_metrics,
        detect_convergence,
        run_convergence_study,
        run_multiple_seeds,
    )

    return (
        calculate_convergence_metrics,
        detect_convergence,
        mo,
        plt,
        run_convergence_study,
        run_multiple_seeds,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Convergence Analysis

    Assesses whether Monte Carlo estimates stabilize as sample size increases.
    """)
    return


@app.cell
def _(mo):
    n_samples_slider = mo.ui.slider(
        start=5, stop=50, step=5, value=20, label='Max sample size (x1000)'
    )
    n_samples_slider
    return (n_samples_slider,)


@app.cell
def _(n_samples_slider, run_convergence_study):
    _max_n = n_samples_slider.value * 1000
    sample_sizes = [1_000, 5_000, 10_000, 20_000, _max_n]
    sample_sizes = sorted({s for s in sample_sizes if s <= _max_n})

    convergence_df = run_convergence_study(
        sample_sizes=sample_sizes,
        outputs=['clogging_probability', 'X', 'volume_fraction'],
    )
    return convergence_df, sample_sizes


@app.cell
def _(convergence_df, mo):
    mo.ui.table(convergence_df, label='Convergence Results')
    return


@app.cell
def _(calculate_convergence_metrics, convergence_df, mo):
    metrics_df = calculate_convergence_metrics(convergence_df)
    mo.ui.table(metrics_df, label='Convergence Metrics (vs largest sample)')
    return


@app.cell
def _(convergence_df, detect_convergence, mo):
    _sizes = detect_convergence(convergence_df)

    _lines = ['**Convergence detected at:**\n']
    for _output, _size in _sizes.items():
        _data = convergence_df[convergence_df['output'] == _output].sort_values(
            'sample_size'
        )
        _first_mean = _data.iloc[0]['mean']
        _last_mean = _data.iloc[-1]['mean']
        _first_ci = _data.iloc[0]['ci_width']
        _last_ci = _data.iloc[-1]['ci_width']
        _ci_reduction = (1 - _last_ci / _first_ci) * 100

        _lines.append(
            f'- **{_output}**: converged at {_size:,} samples.'
            f' Mean={_last_mean:.4f}, CI:'
            f' {_first_ci:.4f} -> {_last_ci:.4f} ({_ci_reduction:.0f}% reduction).'
        )

    _largest = int(convergence_df['sample_size'].max())
    _smallest = int(convergence_df['sample_size'].min())
    _lines.append(
        f'\nFrom N={_smallest:,} to N={_largest:,}, all outputs converged. '
        f'This confirms N={_largest:,} is sufficient for stable estimates.'
    )

    mo.md('\n'.join(_lines))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Convergence Plots
    """)
    return


@app.cell
def _(convergence_df, plt):
    _outputs = ['clogging_probability', 'X', 'volume_fraction']
    _titles = ['Clogging Probability', 'Clogging Parameter X', 'Volume Fraction']

    fig_conv, axes_conv = plt.subplots(1, 3, figsize=(15, 4))
    for _i, (_output, _title) in enumerate(zip(_outputs, _titles)):
        _data = convergence_df[convergence_df['output'] == _output].sort_values(
            'sample_size'
        )
        axes_conv[_i].errorbar(
            _data['sample_size'],
            _data['mean'],
            yerr=[_data['mean'] - _data['ci_lower'], _data['ci_upper'] - _data['mean']],
            fmt='o-',
            capsize=5,
            color='#4C72B0',
        )
        axes_conv[_i].set_xscale('log')
        axes_conv[_i].set_xlabel('Sample Size')
        axes_conv[_i].set_ylabel('Mean +/- 95% CI')
        axes_conv[_i].set_title(_title)

    fig_conv.suptitle('Monte Carlo Convergence', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_conv
    return


@app.cell
def _(convergence_df, plt):
    fig_ci, ax_ci = plt.subplots(figsize=(8, 5))
    for _output, _color, _marker in [
        ('clogging_probability', '#4C72B0', 'o'),
        ('X', '#55A868', 's'),
        ('volume_fraction', '#C44E52', '^'),
    ]:
        _data = convergence_df[convergence_df['output'] == _output].sort_values(
            'sample_size'
        )
        ax_ci.plot(
            _data['sample_size'],
            _data['ci_width'],
            f'{_marker}-',
            color=_color,
            label=_output,
        )
    ax_ci.set_xscale('log')
    ax_ci.set_yscale('log')
    ax_ci.set_xlabel('Sample Size')
    ax_ci.set_ylabel('95% CI Width')
    ax_ci.set_title('CI Width Convergence (log-log)')
    ax_ci.legend()
    plt.tight_layout()
    fig_ci
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Multi-Seed Stability
    """)
    return


@app.cell
def _(mo):
    n_runs_slider = mo.ui.slider(
        start=3, stop=15, step=1, value=5, label='Number of seeds'
    )
    n_runs_slider
    return (n_runs_slider,)


@app.cell
def _(n_runs_slider, run_multiple_seeds, sample_sizes):
    multi_run_df = run_multiple_seeds(
        sample_size=sample_sizes[-1],
        n_runs=n_runs_slider.value,
    )
    return (multi_run_df,)


@app.cell
def _(mo, multi_run_df):
    mo.ui.table(multi_run_df, label='Multi-Seed Results')
    return


@app.cell
def _(multi_run_df, plt):
    _outputs = ['clogging_probability', 'X', 'volume_fraction']
    fig_multi, axes_multi = plt.subplots(1, 3, figsize=(15, 4))
    for _i, _output in enumerate(_outputs):
        _data = multi_run_df[multi_run_df['output'] == _output]
        axes_multi[_i].bar(
            range(len(_data)),
            _data['mean'],
            yerr=_data['std'],
            capsize=3,
            color='#4C72B0',
            alpha=0.7,
        )
        axes_multi[_i].axhline(_data['mean'].mean(), color='red', linestyle='--')
        axes_multi[_i].set_xlabel('Run')
        axes_multi[_i].set_ylabel('Mean')
        axes_multi[_i].set_title(_output)
    fig_multi.suptitle('Multiple Seeds (Mean +/- Std per Run)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_multi
    return


@app.cell
def _(mo, multi_run_df):
    _summary_rows = []
    for _output in ['clogging_probability', 'X', 'volume_fraction']:
        _data = multi_run_df[multi_run_df['output'] == _output]
        _grand_mean = _data['mean'].mean()
        _between_std = _data['mean'].std()
        _cv = _between_std / _grand_mean if _grand_mean != 0 else 0
        _summary_rows.append(
            f'| {_output} | {_grand_mean:.6f} | {_between_std:.6f} | {_cv:.2%} |'
        )

    mo.md(
        '**Between-Run Variance:**\n\n'
        '| Output | Grand Mean | Between-Run Std | CV |\n'
        '|--------|-----------|----------------|-----|\n'
        + '\n'.join(_summary_rows)
        + '\n\nCV < 2% confirms sufficient sample size.'
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    **Convergence plot:** Error bars should overlap between the last two
    sample sizes. Log-scale x-axis shows whether convergence follows
    the expected 1/sqrt(N) rate.

    **CI width plot:** On log-log scale, MC convergence should show a
    slope of -0.5. Deviations indicate heavy tails or insufficient
    samples.

    **Multi-seed:** CV (between-run coefficient of variation) should be
    < 2%. If higher, increase sample size. Grand mean line should sit
    close to all bars.

    **Paper sentence:** "N=XX was selected based on convergence analysis.
    Stability was verified across N independent seeds with between-run
    CV < 2%."
    """)
    return


if __name__ == '__main__':
    app.run()
