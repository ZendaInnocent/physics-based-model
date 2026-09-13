import marimo

__generated_with = '0.21.1'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from nozzle_clogging import config
    from nozzle_clogging.simulation import run_simulation

    return config, mo, pd, plt, run_simulation, sns


@app.cell
def _(mo):
    mo.md(r"""
    # Data Understanding with Visualization

    Univariate and multivariate plots for exploring the Monte Carlo
    simulation dataset.
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

    mag_cols = [c for c in df.columns if hasattr(df[c], 'pint')]
    df_mag = df.copy()
    for _col in mag_cols:
        df_mag[_col] = df[_col].pint.magnitude
    return df, df_mag


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Univariate Plots
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.1 Histograms
    """)
    return


@app.cell
def _(df_mag, plt):
    input_cols = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
    ]
    input_labels = [
        'TSS (mg/L)',
        'Pressure (kPa)',
        'Nozzle Diameter (mm)',
        'Duration (hrs)',
        'Particle Diameter (um)',
    ]

    fig_h1, axes_h1 = plt.subplots(2, 3, figsize=(15, 8))
    axes_h1_flat = axes_h1.flatten()

    for _i, (_col, _label) in enumerate(zip(input_cols, input_labels)):
        axes_h1_flat[_i].hist(
            df_mag[_col], bins=50, edgecolor='black', alpha=0.7, color='#4C72B0'
        )
        axes_h1_flat[_i].set_title(_label)
        axes_h1_flat[_i].set_ylabel('Frequency')
        axes_h1_flat[_i].axvline(
            df_mag[_col].mean(),
            color='red',
            linestyle='--',
            label=f'mean={df_mag[_col].mean():.1f}',
        )
        axes_h1_flat[_i].legend(fontsize=8)

    axes_h1_flat[5].set_visible(False)
    fig_h1.suptitle('Input Parameter Distributions', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_h1
    return


@app.cell
def _(df_mag, plt):
    physics_cols = [
        'stokes_number',
        'dp_dn_ratio',
        'dp_dn_factor',
        'velocity_shear_factor',
        'settling_velocity',
    ]
    physics_labels = [
        'Stokes Number',
        'dp/Dn Ratio',
        'dp/Dn Factor',
        'Velocity Shear Factor',
        'Settling Velocity (m/s)',
    ]

    fig_h2, axes_h2 = plt.subplots(2, 3, figsize=(15, 8))
    axes_h2_flat = axes_h2.flatten()

    for _i, (_col, _label) in enumerate(zip(physics_cols, physics_labels)):
        axes_h2_flat[_i].hist(
            df_mag[_col], bins=50, edgecolor='black', alpha=0.7, color='#55A868'
        )
        axes_h2_flat[_i].set_title(_label)
        axes_h2_flat[_i].set_ylabel('Frequency')
        axes_h2_flat[_i].axvline(
            df_mag[_col].mean(),
            color='red',
            linestyle='--',
            label=f'mean={df_mag[_col].mean():.4f}',
        )
        axes_h2_flat[_i].legend(fontsize=8)

    axes_h2_flat[5].set_visible(False)
    fig_h2.suptitle('Physics Parameter Distributions', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_h2
    return


@app.cell
def _(df_mag, plt):
    output_cols = ['volume_fraction', 'X_base', 'X', 'clogging_probability']
    output_labels = [
        'Volume Fraction',
        'X_base',
        'X (Clogging Parameter)',
        'Clogging Probability',
    ]

    fig_h3, axes_h3 = plt.subplots(2, 2, figsize=(12, 8))
    axes_h3_flat = axes_h3.flatten()

    for _i, (_col, _label) in enumerate(zip(output_cols, output_labels)):
        axes_h3_flat[_i].hist(
            df_mag[_col], bins=50, edgecolor='black', alpha=0.7, color='#C44E52'
        )
        axes_h3_flat[_i].set_title(_label)
        axes_h3_flat[_i].set_ylabel('Frequency')
        axes_h3_flat[_i].axvline(
            df_mag[_col].mean(),
            color='blue',
            linestyle='--',
            label=f'mean={df_mag[_col].mean():.4f}',
        )
        axes_h3_flat[_i].legend(fontsize=8)

    fig_h3.suptitle('Output Parameter Distributions', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_h3
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:**

    - Inputs (TSS, pressure, nozzle, duration) are uniformly
      distributed by design (LHS).
    - `particle_diameter_um` is right-skewed with visible lognormal peaks
      from three size classes.
    - `X_base` is extremely right-skewed with a long tail;
      `clogging_probability` shows two modes - a bulk of low values
      and a cluster near 1.0 (High risk).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.2 Density Plots (KDE)
    """)
    return


@app.cell
def _(df_mag, plt, sns):
    kde_cols = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
    ]
    kde_labels = [
        'TSS (mg/L)',
        'Pressure (kPa)',
        'Nozzle Diameter (mm)',
        'Duration (hrs)',
    ]

    fig_kde1, axes_kde1 = plt.subplots(2, 2, figsize=(12, 8))
    axes_kde1_flat = axes_kde1.flatten()

    for _i, (_col, _label) in enumerate(zip(kde_cols, kde_labels)):
        sns.kdeplot(data=df_mag, x=_col, ax=axes_kde1_flat[_i], fill=True, alpha=0.5)
        axes_kde1_flat[_i].set_title(_label)
        axes_kde1_flat[_i].set_ylabel('Density')

    fig_kde1.suptitle('Input Parameter Density Plots', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_kde1
    return


@app.cell
def _(df_mag, plt, sns):
    kde_out_cols = ['X_base', 'X', 'clogging_probability']
    kde_out_labels = ['X_base', 'X (Clogging Parameter)', 'Clogging Probability']

    fig_kde2, axes_kde2 = plt.subplots(1, 3, figsize=(15, 4))

    for _i, (_col, _label) in enumerate(zip(kde_out_cols, kde_out_labels)):
        sns.kdeplot(
            data=df_mag, x=_col, ax=axes_kde2[_i], fill=True, alpha=0.5, color='#C44E52'
        )
        axes_kde2[_i].set_title(_label)
        axes_kde2[_i].set_ylabel('Density')

    fig_kde2.suptitle('Output Parameter Density Plots', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_kde2
    return


@app.cell
def _(df_mag, plt, sns):
    fig_kde3, ax_kde3 = plt.subplots(figsize=(10, 5))

    for _range in ['Fine', 'Medium', 'Coarse']:
        subset = df_mag[df_mag['particle_size_range'] == _range]['particle_diameter_um']
        sns.kdeplot(subset, ax=ax_kde3, label=_range, fill=True, alpha=0.3)

    ax_kde3.set_title('Particle Diameter by Size Range')
    ax_kde3.set_xlabel('Particle Diameter (um)')
    ax_kde3.set_ylabel('Density')
    ax_kde3.legend()
    plt.tight_layout()
    fig_kde3
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:**

    - Input KDEs confirm flat (uniform) distributions.
    - The three particle size classes show distinct lognormal peaks
      (Fine ~22 um, Medium ~87 um, Coarse ~212 um medians) with overlap.
    - `clogging_probability` density has a sharp peak near 0 and a
      secondary bump near 1.0 - the logistic function creates a
      bimodal output from the continuous `X` input.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.3 Box and Whisker Plots
    """)
    return


@app.cell
def _(df_mag, plt):
    box_input_cols = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
    ]

    fig_box1, axes_box1 = plt.subplots(1, 4, figsize=(14, 5))

    for _i, _col in enumerate(box_input_cols):
        _bp = axes_box1[_i].boxplot(df_mag[_col].values, patch_artist=True, widths=0.6)
        _bp['boxes'][0].set_facecolor('#4C72B0')
        _bp['boxes'][0].set_alpha(0.6)
        axes_box1[_i].set_title(_col.replace('_', '\n'), fontsize=9)
        axes_box1[_i].set_ylabel('Value')

    fig_box1.suptitle('Input Parameter Box Plots', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_box1
    return


@app.cell
def _(df_mag, plt):
    box_phys_cols = [
        'stokes_number',
        'dp_dn_ratio',
        'dp_dn_factor',
        'settling_velocity',
    ]

    fig_box2, axes_box2 = plt.subplots(1, 4, figsize=(14, 5))

    for _i, _col in enumerate(box_phys_cols):
        _bp = axes_box2[_i].boxplot(df_mag[_col].values, patch_artist=True, widths=0.6)
        _bp['boxes'][0].set_facecolor('#55A868')
        _bp['boxes'][0].set_alpha(0.6)
        axes_box2[_i].set_title(_col.replace('_', '\n'), fontsize=9)
        axes_box2[_i].set_ylabel('Value')

    fig_box2.suptitle('Physics Parameter Box Plots', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_box2
    return


@app.cell
def _(df_mag, plt):
    box_out_cols = ['volume_fraction', 'X_base', 'X', 'clogging_probability']

    fig_box3, axes_box3 = plt.subplots(1, 4, figsize=(14, 5))

    for _i, _col in enumerate(box_out_cols):
        _bp = axes_box3[_i].boxplot(df_mag[_col].values, patch_artist=True, widths=0.6)
        _bp['boxes'][0].set_facecolor('#C44E52')
        _bp['boxes'][0].set_alpha(0.6)
        axes_box3[_i].set_title(_col.replace('_', '\n'), fontsize=9)
        axes_box3[_i].set_ylabel('Value')

    fig_box3.suptitle('Output Parameter Box Plots', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_box3
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:**
    - Input box plots show symmetric distributions with no outliers (uniform by design).
    - `stokes_number` and `X_base` have extreme outliers extending far
      above the IQR - these are genuine physical extremes (large particles
      in small nozzles at high TSS).
    - `dp_dn_factor` shows two clusters: values below 0.6 (easy passage
      regime) and at 1.0 (obstruction regime).
    - `clogging_probability` median is low (~0.15) but the upper whisker reaches 1.0.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.4 Categorical Distributions
    """)
    return


@app.cell
def _(config, df, plt):
    fig_bar, axes_bar = plt.subplots(1, 2, figsize=(12, 4))

    ps_order = list(config.PARTICLE_SIZE_RANGES.keys())
    ps_counts = df['particle_size_range'].value_counts().reindex(ps_order)
    bars1 = axes_bar[0].bar(
        ps_order,
        ps_counts.values,
        color=['#4C72B0', '#55A868', '#C44E52'],
        edgecolor='black',
    )
    axes_bar[0].set_title('Particle Size Range')
    axes_bar[0].set_ylabel('Count')
    for _bar, _v in zip(bars1, ps_counts.values):
        axes_bar[0].text(
            _bar.get_x() + _bar.get_width() / 2,
            _v + 100,
            f'{_v:,}',
            ha='center',
            fontsize=9,
        )

    risk_order = config.RISK_LEVELS
    risk_counts = df['clogging_risk'].value_counts().reindex(risk_order)
    colors = ['#55A868', '#F2C94C', '#C44E52']
    bars2 = axes_bar[1].bar(
        risk_order, risk_counts.values, color=colors, edgecolor='black'
    )
    axes_bar[1].set_title('Clogging Risk Level')
    axes_bar[1].set_ylabel('Count')
    for _bar, _v in zip(bars2, risk_counts.values):
        axes_bar[1].text(
            _bar.get_x() + _bar.get_width() / 2,
            _v + 100,
            f'{_v:,}',
            ha='center',
            fontsize=9,
        )

    plt.tight_layout()
    fig_bar
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:**
    - Particle size classes are perfectly balanced (~33% each) confirming
      LHS categorical sampling.
    - Risk distribution is skewed: ~65% Low, ~30% High, ~5% Moderate.
      The thin Moderate band indicates the logistic curve transitions
      sharply between Low and High risk.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Multivariate Plots
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.1 Correlation Matrix Heatmaps
    """)
    return


@app.cell
def _(df_mag, plt, sns):
    heatmap_cols = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
        'stokes_number',
        'dp_dn_ratio',
        'settling_velocity',
        'X',
        'clogging_probability',
    ]
    heatmap_labels = [
        'TSS',
        'Pressure',
        'Nozzle Dia.',
        'Duration',
        'Particle Dia.',
        'Stokes',
        'dp/Dn',
        'Settling V.',
        'X',
        'P(clog)',
    ]

    fig_corr, axes_corr = plt.subplots(1, 2, figsize=(20, 8))

    pearson_mat = df_mag[heatmap_cols].corr(method='pearson')
    pearson_mat.index = heatmap_labels
    pearson_mat.columns = heatmap_labels
    sns.heatmap(
        pearson_mat,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        ax=axes_corr[0],
        square=True,
        linewidths=0.5,
    )
    axes_corr[0].set_title('Pearson Correlation (Linear)', fontsize=12)

    spearman_mat = df_mag[heatmap_cols].corr(method='spearman')
    spearman_mat.index = heatmap_labels
    spearman_mat.columns = heatmap_labels
    sns.heatmap(
        spearman_mat,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        ax=axes_corr[1],
        square=True,
        linewidths=0.5,
    )
    axes_corr[1].set_title('Spearman Correlation (Monotonic)', fontsize=12)

    plt.tight_layout()
    fig_corr
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:** Pearson and Spearman side by side reveal
    nonlinear relationships. Key differences:
    - `stokes_number` vs P(clog): Spearman (+0.85) >> Pearson (+0.58)
      meaning the relationship is strongly monotonic but nonlinear.
    - Input-input correlations are all < |0.02| confirming LHS works.
    - `dp_dn_ratio` is the strongest predictor in both metrics.
    - `pressure` shows weak correlation in both - its opposing effects
      (more Stokes deposition vs more shear cleaning) cancel.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.2 Scatter Plot Matrix (Lower Triangle)
    """)
    return


@app.cell
def _(df_mag, sns):
    pair_cols = [
        'TSS_mg_L',
        'nozzle_diameter_mm',
        'particle_diameter_um',
        'duration_hrs',
        'clogging_probability',
    ]
    pair_df = df_mag[pair_cols].copy()
    pair_df.columns = ['TSS', 'Nozzle Dia.', 'Particle Dia.', 'Duration', 'P(clog)']

    g = sns.pairplot(
        pair_df.sample(3000, random_state=42),
        diag_kind='kde',
        plot_kws={'alpha': 0.15, 's': 8},
        diag_kws={'fill': True, 'alpha': 0.5},
        corner=True,
    )
    g.figure.suptitle('Inputs vs Clogging Probability (Lower Triangle)', y=1.01)
    g
    return


@app.cell
def _(df_mag, sns):
    pair_cols2 = [
        'dp_dn_ratio',
        'stokes_number',
        'settling_velocity',
        'X',
        'clogging_probability',
    ]
    pair_df2 = df_mag[pair_cols2].copy()
    pair_df2.columns = ['dp/Dn', 'Stokes', 'Settling V.', 'X', 'P(clog)']

    g2 = sns.pairplot(
        pair_df2.sample(3000, random_state=42),
        diag_kind='kde',
        plot_kws={'alpha': 0.15, 's': 8, 'color': '#55A868'},
        diag_kws={'fill': True, 'alpha': 0.5, 'color': '#55A868'},
        corner=True,
    )
    g2.figure.suptitle('Physics Parameters vs P(clog) (Lower Triangle)', y=1.01)
    g2
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:** The lower-triangle scatter matrix shows:
    - **dp/Dn vs P(clog)**: Clear positive trend - small nozzles with
      large particles produce High risk. The relationship fans out
      (heteroscedastic).
    - **Particle Dia. vs dp/Dn**: Near-linear positive relationship
      as expected (ratio scales with numerator).
    - **X vs P(clog)**: Tight S-shaped curve showing the logistic
      transform in action.
    - **Stokes vs P(clog)**: Most points cluster at high Stokes
      (inertial regime) with P(clog) spread across all values,
      confirming Stokes alone doesn't determine risk.
    - **Settling V. vs everything**: Weak relationships - settling
      is not a discriminating factor in this regime.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.3 Input Parameters vs Clogging Probability
    """)
    return


@app.cell
def _(df_mag, plt):
    scatter_inputs = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
    ]
    scatter_labels = [
        'TSS (mg/L)',
        'Pressure (kPa)',
        'Nozzle Diameter (mm)',
        'Duration (hrs)',
        'Particle Diameter (um)',
    ]

    fig_sc, axes_sc = plt.subplots(2, 3, figsize=(15, 9))
    axes_sc_flat = axes_sc.flatten()

    for _i, (_col, _label) in enumerate(zip(scatter_inputs, scatter_labels)):
        axes_sc_flat[_i].scatter(
            df_mag[_col],
            df_mag['clogging_probability'],
            alpha=0.1,
            s=5,
            color='#4C72B0',
        )
        axes_sc_flat[_i].set_xlabel(_label)
        axes_sc_flat[_i].set_ylabel('P(clogging)')

        _spearman = df_mag[_col].corr(df_mag['clogging_probability'], method='spearman')
        axes_sc_flat[_i].set_title(
            f'{_label}\n(Spearman r={_spearman:.3f})', fontsize=10
        )

    axes_sc_flat[5].set_visible(False)
    fig_sc.suptitle('Input Parameters vs Clogging Probability', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_sc
    return


@app.cell
def _(df_mag, plt):
    scatter_phys = [
        'dp_dn_ratio',
        'stokes_number',
        'settling_velocity',
        'X',
        'volume_fraction',
    ]
    scatter_phys_labels = [
        'dp/Dn Ratio',
        'Stokes Number',
        'Settling Velocity (m/s)',
        'X',
        'Volume Fraction',
    ]

    fig_sc2, axes_sc2 = plt.subplots(2, 3, figsize=(15, 9))
    axes_sc2_flat = axes_sc2.flatten()

    for _i, (_col, _label) in enumerate(zip(scatter_phys, scatter_phys_labels)):
        axes_sc2_flat[_i].scatter(
            df_mag[_col],
            df_mag['clogging_probability'],
            alpha=0.1,
            s=5,
            color='#55A868',
        )
        axes_sc2_flat[_i].set_xlabel(_label)
        axes_sc2_flat[_i].set_ylabel('P(clogging)')

        _spearman = df_mag[_col].corr(df_mag['clogging_probability'], method='spearman')
        axes_sc2_flat[_i].set_title(
            f'{_label}\n(Spearman r={_spearman:.3f})', fontsize=10
        )

    axes_sc2_flat[5].set_visible(False)
    fig_sc2.suptitle('Physics Parameters vs Clogging Probability', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_sc2
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:** Scatter plots reveal the shape of each
    input-output relationship:
    - **Nozzle diameter**: Inverse relationship - smaller nozzles
      produce higher P(clog). The upper band (P~1) thickens as
      diameter decreases.
    - **Particle diameter**: Strong positive, with P(clog)=1 for
      particles > ~300 um regardless of other conditions.
    - **TSS and Duration**: Weak positive trends - they modulate
      risk but don't dominate.
    - **Pressure**: Nearly flat cloud - competing effects cancel.
    - **dp/Dn ratio**: Strongest predictor with a sharp transition
      zone around dp/Dn ~ 0.05-0.15.
    - **X**: S-shaped cluster matching the logistic curve.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.4 Clogging Probability by Categories
    """)
    return


@app.cell
def _(config, df_mag, pd, plt):
    fig_gbox, axes_gbox = plt.subplots(1, 3, figsize=(15, 5))

    _ps_order = list(config.PARTICLE_SIZE_RANGES.keys())
    ps_data = [
        df_mag[df_mag['particle_size_range'] == r]['clogging_probability'].values
        for r in _ps_order
    ]
    _bp1 = axes_gbox[0].boxplot(
        ps_data, labels=_ps_order, patch_artist=True, widths=0.6
    )
    for _patch, _color in zip(_bp1['boxes'], ['#4C72B0', '#55A868', '#C44E52']):
        _patch.set_facecolor(_color)
        _patch.set_alpha(0.6)
    axes_gbox[0].set_title('By Particle Size Range')
    axes_gbox[0].set_ylabel('P(clogging)')

    _risk_order = config.RISK_LEVELS
    risk_data = [
        df_mag[df_mag['clogging_risk'] == r]['clogging_probability'].values
        for r in _risk_order
    ]
    _bp2 = axes_gbox[1].boxplot(
        risk_data, labels=_risk_order, patch_artist=True, widths=0.6
    )
    for _patch, _color in zip(_bp2['boxes'], ['#55A868', '#F2C94C', '#C44E52']):
        _patch.set_facecolor(_color)
        _patch.set_alpha(0.6)
    axes_gbox[1].set_title('By Risk Level')
    axes_gbox[1].set_ylabel('P(clogging)')

    nozzle_bins = pd.cut(df_mag['nozzle_diameter_mm'], bins=4)
    nozzle_groups = (
        df_mag.groupby(nozzle_bins, observed=False)['clogging_probability']
        .apply(list)
        .values
    )
    nozzle_labels = [str(interval) for interval in nozzle_bins.cat.categories]
    _bp3 = axes_gbox[2].boxplot(
        nozzle_groups, labels=nozzle_labels, patch_artist=True, widths=0.6
    )
    for _patch in _bp3['boxes']:
        _patch.set_facecolor('#93786B')
        _patch.set_alpha(0.6)
    axes_gbox[2].set_title('By Nozzle Diameter Bin')
    axes_gbox[2].set_ylabel('P(clogging)')
    axes_gbox[2].tick_params(axis='x', rotation=30)

    fig_gbox.suptitle('Clogging Probability by Categories', fontsize=14, y=1.01)
    plt.tight_layout()
    fig_gbox
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:**
    - **By particle size**: Coarse particles produce much higher
      median P(clog) and wider spread. Fine particles are almost
      always Low risk.
    - **By risk level**: Clean separation between categories
      confirms thresholds (0.30/0.50) are well-placed.
    - **By nozzle diameter**: Smaller nozzle bins (1.5-2.6 mm)
      show higher median and upper quartile. The 4.5-6.0 mm bin
      is almost entirely Low risk.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.5 Violin Plots
    """)
    return


@app.cell
def _(config, df_mag, plt, sns):
    fig_violin, axes_violin = plt.subplots(1, 2, figsize=(12, 5))

    _risk_order = config.RISK_LEVELS
    risk_palette = {'Low': '#55A868', 'Moderate': '#F2C94C', 'High': '#C44E52'}

    sns.violinplot(
        data=df_mag,
        x='clogging_risk',
        y='clogging_probability',
        order=_risk_order,
        palette=risk_palette,
        ax=axes_violin[0],
        inner='quart',
    )
    axes_violin[0].set_title('P(clogging) by Risk Level')
    axes_violin[0].set_xlabel('Risk Level')
    axes_violin[0].set_ylabel('Clogging Probability')

    _ps_order = list(config.PARTICLE_SIZE_RANGES.keys())
    ps_palette = {'Fine': '#4C72B0', 'Medium': '#55A868', 'Coarse': '#C44E52'}
    sns.violinplot(
        data=df_mag,
        x='particle_size_range',
        y='X',
        order=_ps_order,
        palette=ps_palette,
        ax=axes_violin[1],
        inner='quart',
    )
    axes_violin[1].set_title('Clogging Parameter X by Particle Size')
    axes_violin[1].set_xlabel('Particle Size Range')
    axes_violin[1].set_ylabel('X')

    plt.tight_layout()
    fig_violin
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:** Violins show the full density shape per group:
    - **P(clog) by risk**: Low is tightly concentrated near 0, High
      has a bimodal shape (peak near 0.8 and 1.0), Moderate is a
      thin band - the logistic curve skips quickly through this zone.
    - **X by particle size**: Coarse particles produce the widest X
      distribution with a heavy right tail. Fine particles cluster
      near 0. The physical mechanism is that larger particles have
      higher dp/Dn ratios, dominating the clogging parameter.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.6 Joint Density Plots
    """)
    return


@app.cell
def _(df_mag, sns):
    g_joint1 = sns.jointplot(
        data=df_mag.sample(10000, random_state=42),
        x='particle_diameter_um',
        y='clogging_probability',
        kind='kde',
        fill=True,
        alpha=0.5,
        cmap='Blues',
        levels=10,
    )
    g_joint1.figure.suptitle('Particle Diameter vs P(clogging)', y=1.02)
    g_joint1
    return


@app.cell
def _(df_mag, sns):
    g_joint2 = sns.jointplot(
        data=df_mag.sample(10000, random_state=42),
        x='dp_dn_ratio',
        y='clogging_probability',
        kind='kde',
        fill=True,
        alpha=0.5,
        cmap='Greens',
        levels=10,
    )
    g_joint2.figure.suptitle('dp/Dn Ratio vs P(clogging)', y=1.02)
    g_joint2
    return


@app.cell
def _(df_mag, sns):
    g_joint3 = sns.jointplot(
        data=df_mag.sample(10000, random_state=42),
        x='nozzle_diameter_mm',
        y='clogging_probability',
        kind='hex',
        cmap='YlOrRd',
        mincnt=1,
    )
    g_joint3.figure.suptitle('Nozzle Diameter vs P(clogging)', y=1.02)
    g_joint3
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:** Joint density plots show where data concentrates:
    - **Particle Dia. vs P(clog)**: Two density lobes - a dense blob
      at small particles/low risk and a spread at large particles/
      high risk. The transition is gradual.
    - **dp/Dn vs P(clog)**: The highest density is at low dp/Dn with
      P(clog)~0. Density thins as dp/Dn increases and P(clog) rises.
      A clear threshold effect around dp/Dn~0.10.
    - **Nozzle Dia. vs P(clog)** (hexbin): Dense cells at large
      nozzles/low risk. The High risk band (P>0.5) is concentrated
      at nozzle < 3 mm.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.7 Parallel Coordinates Plot
    """)
    return


@app.cell
def _(df_mag, plt):
    par_cols = [
        'TSS_mg_L',
        'nozzle_diameter_mm',
        'particle_diameter_um',
        'duration_hrs',
        'clogging_probability',
    ]
    par_df = df_mag[par_cols + ['clogging_risk']].copy()

    for _col in par_cols:
        _rng = par_df[_col].max() - par_df[_col].min()
        if _rng > 0:
            par_df[_col] = (par_df[_col] - par_df[_col].min()) / _rng

    par_sample = par_df.sample(2000, random_state=42)

    color_map = {'Low': '#55A868', 'Moderate': '#F2C94C', 'High': '#C44E52'}
    par_colors = par_sample['clogging_risk'].map(color_map)

    fig_par, ax_par = plt.subplots(figsize=(14, 6))

    for _idx, _row in par_sample.iterrows():
        vals = _row[par_cols].values.astype(float)
        ax_par.plot(
            range(len(par_cols)),
            vals,
            color=par_colors[_idx],
            alpha=0.08,
            linewidth=0.5,
        )

    ax_par.set_xticks(range(len(par_cols)))
    ax_par.set_xticklabels(
        ['TSS', 'Nozzle Dia.', 'Particle Dia.', 'Duration', 'P(clog)'], fontsize=9
    )
    ax_par.set_ylabel('Normalized Value (0-1)')
    ax_par.set_title('Parallel Coordinates: Normalized Inputs by Risk Level')

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=c, lw=2, label=_lbl) for _lbl, c in color_map.items()
    ]
    ax_par.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    fig_par
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Understanding:** Parallel coordinates trace each simulation as
    a line across normalized axes, colored by risk level:
    - **High risk (red)** lines tend to have high particle diameter,
      small nozzle diameter, and consequently high P(clog).
    - **Low risk (green)** lines cluster at low particle diameter,
      large nozzle diameter, and low P(clog).
    - **TSS and Duration** show similar patterns across risk levels,
      confirming they are secondary factors.
    - The crossing pattern at the nozzle diameter axis is striking:
      High risk lines cross from low to high when moving from nozzle
      diameter to particle diameter, showing the ratio effect.
    """)
    return


if __name__ == '__main__':
    app.run()
