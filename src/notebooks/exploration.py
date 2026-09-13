import marimo

__generated_with = '0.21.1'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from nozzle_clogging import config
    from nozzle_clogging.simulation import run_simulation

    return config, mo, np, pd, plt, run_simulation, stats


@app.cell
def _(mo):
    mo.md(r"""
    # Data Exploration: Nozzle Clogging Simulation

    This notebook explores the Monte Carlo simulation dataset for
    sediment-induced clogging in sprinkler irrigation systems.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Data Generation
    """)
    return


@app.cell
def _(mo):
    n_samples_slider = mo.ui.slider(
        start=5_000, stop=50_000, step=5_000, value=20_000, label='Total samples'
    )
    n_samples_slider
    return (n_samples_slider,)


@app.cell
def _(n_samples_slider, run_simulation):
    df = run_simulation(total_samples=n_samples_slider.value)
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    Dataset generated: **{len(df):,}** samples
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Raw Data Peek
    """)
    return


@app.cell
def _(df, mo):
    mo.vstack([mo.md('**First 10 rows:**'), df.head(10)])
    return


@app.cell
def _(df, mo):
    mo.vstack([mo.md('**Last 5 rows:**'), df.tail(10)])
    return


@app.cell
def _(df):
    # Extract magnitude-only view for easier inspection
    mag_cols = [c for c in df.columns if hasattr(df[c], 'pint')]
    df_mag = df.copy()
    for _col in mag_cols:
        df_mag[_col] = df[_col].pint.magnitude
    df_mag
    return (df_mag,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Dataset Dimensions
    """)
    return


@app.cell
def _(df, mo, pd):
    dim_summary = pd.DataFrame(
        {
            'Metric': ['Rows', 'Columns', 'Total cells', 'Memory (MB)'],
            'Value': [
                f'{df.shape[0]:,}',
                df.shape[1],
                f'{df.shape[0] * df.shape[1]:,}',
                f'{df.memory_usage(deep=True).sum() / 1e6:.2f}',
            ],
        }
    )
    mo.ui.table(dim_summary)
    return


@app.cell
def _(df, mo):
    mo.md(f"""
    **Shape:** {df.shape[0]:,} rows x {df.shape[1]} columns
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Data Types
    """)
    return


@app.cell
def _(df, mo, pd):
    dtype_df = pd.DataFrame(
        {
            'Column': df.columns,
            'Dtype': [str(df[c].dtype) for c in df.columns],
            'Non-null': [df[c].notna().sum() for c in df.columns],
            'Null': [df[c].isna().sum() for c in df.columns],
        }
    )
    mo.ui.table(dtype_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Class Distribution
    """)
    return


@app.cell
def _(df, mo):
    # Particle size range distribution
    ps_dist = df['particle_size_range'].value_counts().reset_index()
    ps_dist.columns = ['Particle Size Range', 'Count']
    ps_dist['Proportion'] = ps_dist['Count'] / len(df)
    ps_dist['Proportion'] = ps_dist['Proportion'].map('{:.2%}'.format)

    mo.vstack([mo.md('### Particle Size Range Distribution'), mo.ui.table(ps_dist)])
    return


@app.cell
def _(config, df, mo, pd):
    # Clogging risk distribution
    _risk_counts = df['clogging_risk'].value_counts()
    risk_rows = []
    for level in config.RISK_LEVELS:
        count = _risk_counts.get(level, 0)
        risk_rows.append(
            {
                'Risk Level': level,
                'Count': count,
                'Proportion': f'{count / len(df):.2%}',
            }
        )
    risk_df = pd.DataFrame(risk_rows)

    mo.vstack([mo.md('### Clogging Risk Distribution'), mo.ui.table(risk_df)])
    return


@app.cell
def _(config, df, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Particle size range bar chart
    ps_order = list(config.PARTICLE_SIZE_RANGES.keys())
    ps_counts = df['particle_size_range'].value_counts().reindex(ps_order)
    axes[0].bar(ps_order, ps_counts.values, color=['#4C72B0', '#55A868', '#C44E52'])
    axes[0].set_title('Particle Size Range Distribution')
    axes[0].set_ylabel('Count')
    for _i, _v in enumerate(ps_counts.values):
        axes[0].text(_i, _v + 100, f'{_v:,}', ha='center', fontsize=9)

    # Clogging risk bar chart
    risk_order = config.RISK_LEVELS
    risk_counts = df['clogging_risk'].value_counts().reindex(risk_order)
    colors = ['#55A868', '#F2C94C', '#C44E52']
    axes[1].bar(risk_order, risk_counts.values, color=colors)
    axes[1].set_title('Clogging Risk Distribution')
    axes[1].set_ylabel('Count')
    for _i, _v in enumerate(risk_counts.values):
        axes[1].text(_i, _v + 100, f'{_v:,}', ha='center', fontsize=9)

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Observations:**

    - Are particle size classes evenly distributed? If not, the LHS sampling
      strategy may have a bias toward certain classes.
    - How skewed is the risk distribution? A dominant "Low" category could
      indicate the model is conservative, while a dominant "High" would
      suggest most parameter combinations lead to clogging.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Descriptive Statistics
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.1 Input Parameters
    """)
    return


@app.cell
def _(df_mag, mo):
    input_cols = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
    ]
    input_stats = df_mag[input_cols].describe().T
    input_stats['range'] = input_stats['max'] - input_stats['min']
    input_stats['iqr'] = input_stats['75%'] - input_stats['25%']
    input_stats = input_stats.round(4)
    input_stats.index.name = 'Parameter'

    mo.vstack(
        [
            mo.md('**Summary statistics for input parameters:**'),
            mo.ui.table(input_stats.reset_index()),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.2 Computed Physics Parameters
    """)
    return


@app.cell
def _(df_mag, mo):
    physics_cols = [
        'stokes_number',
        'stokes_factor',
        'dp_dn_ratio',
        'dp_dn_factor',
        'velocity_shear_factor',
        'settling_velocity',
        'settling_velocity_factor',
    ]
    physics_stats = df_mag[physics_cols].describe().T
    physics_stats['range'] = physics_stats['max'] - physics_stats['min']
    physics_stats['iqr'] = physics_stats['75%'] - physics_stats['25%']
    physics_stats = physics_stats.round(6)
    physics_stats.index.name = 'Parameter'

    mo.vstack(
        [
            mo.md('**Summary statistics for computed physics parameters:**'),
            mo.ui.table(physics_stats.reset_index()),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.3 Output Parameters
    """)
    return


@app.cell
def _(df_mag, mo):
    output_cols = [
        'volume_fraction',
        'X_base',
        'physical_factor',
        'X',
        'clogging_probability',
    ]
    output_stats = df_mag[output_cols].describe().T
    output_stats['range'] = output_stats['max'] - output_stats['min']
    output_stats['iqr'] = output_stats['75%'] - output_stats['25%']
    output_stats = output_stats.round(6)
    output_stats.index.name = 'Parameter'

    mo.vstack(
        [
            mo.md('**Summary statistics for output parameters:**'),
            mo.ui.table(output_stats.reset_index()),
        ]
    )
    return


@app.cell
def _(df_mag, plt):
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    axes2_flat = axes2.flatten()

    plot_cols = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'particle_diameter_um',
        'clogging_probability',
        'X',
    ]
    titles = [
        'TSS (mg/L)',
        'Pressure (kPa)',
        'Nozzle Diameter (mm)',
        'Particle Diameter (um)',
        'Clogging Probability',
        'Clogging Parameter X',
    ]

    for _i, (_col, _title) in enumerate(zip(plot_cols, titles)):
        axes2_flat[_i].hist(df_mag[_col], bins=50, edgecolor='black', alpha=0.7)
        axes2_flat[_i].set_title(_title)
        axes2_flat[_i].set_ylabel('Frequency')

    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Observations:**

    - **TSS**: Distribution is perfectly uniform (skewness ~0). LHS works
      as designed. TSS and `volume_fraction` are perfectly correlated (r=1.0)
      since volume_fraction = TSS / rho_sediment.
    - **Pressure**: Uniform distribution, no gaps. Velocities range from
      ~12 to ~24 m/s - all well above the 8 m/s shear threshold, which
      means the `velocity_shear_factor` is always low (0.25-0.40).
    - **Nozzle diameter**: Negative correlation with clogging (-0.41).
      Smaller nozzles clog more - physically intuitive since dp/Dn ratio
      increases. This is one of the strongest predictors.
    - **Duration**: Weak positive correlation (+0.25) with clogging.
      Longer runs accumulate more sediment, but the effect is modest
      compared to particle size and nozzle geometry.
    - **Particle diameter**: Highly right-skewed (skewness=1.1) due to
      lognormal sampling within each class. The combined distribution
      shows three overlapping lognormal peaks. This is the **strongest
      single predictor** of clogging (Spearman r=+0.78).
    """)
    return


@app.cell
def _(df_mag, mo, np, pd, stats):
    ci_cols = ['clogging_probability', 'X', 'volume_fraction']
    ci_rows = []
    for _col in ci_cols:
        _vals = df_mag[_col].dropna().values
        _n = len(_vals)
        _mean = np.mean(_vals)
        _se = stats.sem(_vals)
        _ci_lo, _ci_hi = stats.t.interval(0.95, df=_n - 1, loc=_mean, scale=_se)
        _pi_lo, _pi_hi = np.percentile(_vals, [2.5, 97.5])
        ci_rows.append(
            {
                'Parameter': _col,
                'Mean': f'{_mean:.6f}',
                '95% CI Lower': f'{_ci_lo:.6f}',
                '95% CI Upper': f'{_ci_hi:.6f}',
                '95% PI Lower (2.5%)': f'{_pi_lo:.6f}',
                '95% PI Upper (97.5%)': f'{_pi_hi:.6f}',
                'N': f'{_n:,}',
            }
        )
    ci_df = pd.DataFrame(ci_rows)
    mo.vstack(
        [
            mo.md(r"""
    ## 6.4 Confidence and Prediction Intervals

    95% confidence intervals (CI) for the mean and 95% prediction intervals (PI)
    covering the central 97.5% range of the output distributions.
    """),
            mo.ui.table(ci_df),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Correlations
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.1 Pearson Correlation (Linear)
    """)
    return


@app.cell
def _(df_mag, mo):
    corr_cols = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
        'stokes_number',
        'dp_dn_ratio',
        'settling_velocity',
        'volume_fraction',
        'X',
        'clogging_probability',
    ]
    pearson_corr = df_mag[corr_cols].corr(method='pearson').round(3)

    mo.vstack(
        [
            mo.md('**Pearson correlation matrix (all numeric columns):**'),
            pearson_corr,
        ]
    )
    return (pearson_corr,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.2 Spearman Correlation (Monotonic)
    """)
    return


@app.cell
def _(df_mag, mo):
    corr_cols2 = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
        'stokes_number',
        'dp_dn_ratio',
        'settling_velocity',
        'volume_fraction',
        'X',
        'clogging_probability',
    ]
    spearman_corr = df_mag[corr_cols2].corr(method='spearman').round(3)

    mo.vstack([mo.md('**Spearman correlation matrix:**'), spearman_corr])
    return


@app.cell
def _(pearson_corr, plt):
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    im = ax3.imshow(pearson_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax3.set_xticks(range(len(pearson_corr.columns)))
    ax3.set_yticks(range(len(pearson_corr.columns)))
    ax3.set_xticklabels(pearson_corr.columns, rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels(pearson_corr.columns, fontsize=8)

    # Annotate cells
    for _i in range(len(pearson_corr)):
        for _j in range(len(pearson_corr)):
            val = pearson_corr.values[_i, _j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax3.text(
                _j, _i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color
            )

    plt.colorbar(im, ax=ax3, shrink=0.8)
    ax3.set_title('Pearson Correlation Matrix')
    plt.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Findings:**

    - **LHS sampling verified**: All input-input correlations are near zero
      (max |r| < 0.02). The Latin Hypercube design is working correctly.
    - **TSS -> volume_fraction**: Perfect 1.0 correlation confirmed.
      Direct proportionality via density division.
    - **`dp_dn_ratio` is the strongest linear predictor** (Pearson +0.74,
      Spearman +0.89). The particle-to-nozzle ratio drives clogging.
    - **`stokes_number` has higher Spearman than Pearson** (+0.85 vs +0.58),
      indicating a strong monotonic but nonlinear relationship.
    - **Pressure is surprisingly weak** (Pearson +0.07). Higher pressure
      increases velocity, which increases Stokes number (more deposition)
      but also increases shear (self-cleaning). These effects nearly cancel.
    - **WARNING: `physical_factor` is constant at 0.1** (clipped floor).
      The product of dp_dn_factor * stokes_factor * velocity_shear_factor
      * settling_velocity_factor is always below 0.1, so it gets clipped
      up. This means X = X_base * 0.1 always - the physical modifiers
      provide **zero discriminating power**.
    - **`settling_velocity_factor` is always 0.2** (floor value). Flow
      velocities (12-24 m/s) are so high that settling never matters.
      This factor should be reconsidered for the operating range.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Distribution Skewness
    """)
    return


@app.cell
def _(df_mag, mo, pd, stats):
    skew_cols = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
        'stokes_number',
        'stokes_factor',
        'dp_dn_ratio',
        'dp_dn_factor',
        'velocity_shear_factor',
        'settling_velocity',
        'settling_velocity_factor',
        'volume_fraction',
        'X_base',
        'physical_factor',
        'X',
        'clogging_probability',
    ]

    skew_rows = []
    for _col in skew_cols:
        _vals = df_mag[_col].dropna()
        skewness = _vals.skew()
        kurtosis = _vals.kurtosis()
        _range = _vals.max() - _vals.min()
        if _range < 1e-12:
            sw_p_str = 'N/A (constant)'
            normal_str = 'N/A'
        else:
            _sw_stat, sw_p = stats.shapiro(
                _vals.sample(min(5000, len(_vals)), random_state=42)
            )
            sw_p_str = f'{sw_p:.2e}'
            normal_str = 'Yes' if sw_p > 0.05 else 'No'
        skew_rows.append(
            {
                'Column': _col,
                'Skewness': round(skewness, 4),
                'Kurtosis': round(kurtosis, 4),
                'Shapiro-Wilk p': sw_p_str,
                'Normal (p>0.05)': normal_str,
            }
        )

    skew_df = pd.DataFrame(skew_rows)
    mo.vstack(
        [
            mo.md(
                '**Skewness, kurtosis, and normality test for all numeric columns:**'
            ),
            mo.ui.table(skew_df),
        ]
    )
    return


@app.cell
def _(df_mag, np, plt):
    qq_cols = ['clogging_probability', 'X', 'stokes_number', 'dp_dn_ratio']
    fig4, axes4 = plt.subplots(2, 2, figsize=(10, 8))
    axes4_flat = axes4.flatten()

    for _i, _col in enumerate(qq_cols):
        _vals = df_mag[_col].dropna().values
        sorted_vals = np.sort(_vals)
        n = len(sorted_vals)
        theoretical = np.arange(1, n + 1) / (n + 1)
        # Simple Q-Q: plot sorted data vs uniform quantiles
        axes4_flat[_i].scatter(theoretical[::50], sorted_vals[::50], s=5, alpha=0.5)
        axes4_flat[_i].set_title(f'{_col}')
        axes4_flat[_i].set_xlabel('Theoretical quantile')
        axes4_flat[_i].set_ylabel('Observed value')

    plt.tight_layout()
    fig4
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Skewness findings:**

    | Skewness | Interpretation |
    |----------|----------------|
    | <-1 | Highly left-skewed |
    | -1 to -0.5 | Moderately left-skewed |
    | -0.5 to 0.5 | Approximately symmetric |
    | 0.5 to 1 | Moderately right-skewed |
    | >1 | Highly right-skewed |

    **Key answers:**

    - **`clogging_probability` is moderately right-skewed** (~0.84).
      Most simulations predict low risk, with a long tail toward
      high probability. This matches the 65%/30%/5% risk split.
    - **`X_base` is extremely right-skewed** (skewness ~13). The raw
      clogging parameter before logistic transform has extreme outliers.
      The logistic transform (expit) compresses this into the bounded
      0-1 range, reducing skewness to ~0.84 for `clogging_probability`.
    - **`X` is right-skewed** (~2.7) after clipping to [0, 50] and
      multiplying by the constant physical_factor (0.1). Less extreme
      than X_base due to the clipping.
    - **`particle_diameter_um` is right-skewed** (1.1) as expected from
      lognormal sampling. The three lognormal components (Fine/Medium/
      Coarse) create a multimodal combined distribution.
    - **Input parameters are all approximately symmetric** (skewness ~0).
      LHS sampling is working correctly for all continuous inputs.
    - **`stokes_factor` is heavily left-skewed** because 93% of values
      are exactly 1.0 (Stokes number exceeds critical threshold for most
      conditions). This factor has little discriminating power.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Observations and Ideas

    ### Input Parameter Observations
    - All continuous inputs (TSS, pressure, nozzle diameter, duration) are
      perfectly uniform with skewness ~0. LHS sampling is working correctly.
    - Input-input correlations are all < |0.02|, confirming orthogonal
      coverage of the parameter space.
    - `particle_diameter_um` is the exception: right-skewed (1.1) with
      multimodal structure from three lognormal components. This is
      intentional by design.

    ### Physics Parameter Observations
    - **`stokes_factor` has almost no variance**: 92.9% of values are
      exactly 1.0. For the pressure/nozzle range studied, Stokes numbers
      almost always exceed the critical threshold (0.1). This factor
      contributes essentially nothing to discriminating between conditions.
    - **`velocity_shear_factor` is always near its floor** (0.25-0.40).
      All velocities (12-24 m/s) far exceed the 8 m/s shear threshold,
      so the self-cleaning effect is always "active" but only varies
      modestly.
    - **`settling_velocity_factor` is a constant 0.2** (floor value).
      At the high flow velocities in this parameter space, settling is
      always negligible relative to flow. This factor has zero
      discriminating power.
    - **`physical_factor` is constant at 0.1** (clipped floor). The
      product of all four modifiers is always below 0.1, so it gets
      clipped up to the minimum. This means `X = X_base * 0.1` for
      every single sample. The physical modifiers provide no variation.

    ### Output Distribution Observations
    - `clogging_probability` is moderately right-skewed (0.84) with a
      mean of 0.35. Most simulations predict Low risk (65%), but a
      substantial 30% predict High risk. The Moderate band is thin (5%),
      suggesting the logistic curve threshold sits in a steep region.
    - `X_base` is extremely right-skewed (13.4) with heavy outliers
      (max ~13,552). The raw clogging parameter varies over 7 orders
      of magnitude. After logistic transform, `clogging_probability`
      is bounded to [0, 1] with reduced skewness.
    - Since `physical_factor` is constant, ALL output variation comes
      from `X_base` alone. The model is effectively:
      `P_clog = logistic(X_base * 0.1 - 3.0)`.

    ### Correlation Observations
    - `dp_dn_ratio` (particle-to-nozzle diameter) is the strongest
      predictor (Spearman r=+0.89). Smaller nozzles with larger
      particles = much higher clogging risk.
    - `particle_diameter_um` is the second strongest (Spearman r=+0.78).
      Particle size drives clogging through both the dp/Dn ratio and
      Stokes number pathways.
    - `nozzle_diameter_mm` has the strongest negative correlation
      (-0.42). Larger nozzles reduce clogging - consistent with the
      dp/Dn ratio mechanism.
    - `pressure_kPa` is surprisingly weak (+0.07). Higher pressure
      increases velocity (more Stokes deposition) but also increases
      shear (more self-cleaning). These competing effects nearly
      cancel out.
    - `TSS_mg_L` has a modest positive correlation (+0.27). Higher
      sediment concentration increases volume fraction, but the
      effect is secondary to particle size and nozzle geometry.

    ### Skewness Observations
    - `X_base` needs log transformation if used in linear models
      (skewness 13.4).
    - `particle_diameter_um` could benefit from log transform
      (skewness 1.1) for symmetric modeling.
    - `clogging_probability` (skewness 0.84) is acceptable for most
      methods but may need consideration for regression approaches.
    - All physics factors that are clipped to floors/ceilings need
      model revision - they contribute no information.

    ### Critical Issue: Physical Modifiers Are Dead
    The most important finding: **three of four physical modifiers are
    saturated at their floor values** for the studied parameter range.
    The model reduces to a single-parameter function of `X_base`. This
    means:
    1. Sensitivity analysis will show only `dp_dn_ratio` and particle
       size matter (everything else is washed out by clipping).
    2. The physics model's nuanced effects (Stokes inertia, shear
       cleaning, settling) are not operating in the studied regime.
    3. The risk thresholds (0.30/0.50) were likely tuned for a regime
       where physical factors varied. With constant 0.1, the effective
       threshold on X_base is shifted.

    ### Next Steps
    - **Investigate physical_factor clipping**: What parameter ranges
      would let each modifier vary? Run targeted simulations at lower
      pressures or smaller particles.
    - **Convergence study**: Verify 20k samples is sufficient given
      the simplified model structure.
    - **Sensitivity analysis**: Expect dp_dn_ratio and particle size
      to dominate. Confirm this.
    - **Consider model revision**: If physical modifiers are always
      clipped, the model may need recalibration for the irrigation
      pressure range (100-400 kPa).
    """)
    return


if __name__ == '__main__':
    app.run()
