"""Generate the six manuscript figures.

Produces ``paper/figures/fig_01_schematic.png`` .. ``fig_06_modifier_diagnostics.png``,
matching the figure captions and reporting tables in ``paper/manuscript/manuscript.md``.

Run:  ``uv run python paper/analysis/generate_figures.py``  (from repo root)
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pint

from nozzle_clogging import config
from nozzle_clogging.convergence import run_convergence_study
from nozzle_clogging.physics import calculate_velocity_shear_factor
from nozzle_clogging.sensitivity import (
    calculate_morris_indices,
    calculate_sobol_indices,
)
from nozzle_clogging.simulation import run_simulation
from nozzle_clogging.units import ureg

SEED = config.RANDOM_SEED
OUT_DIR = Path(__file__).resolve().parents[1] / 'figures'
CLASS_ORDER = ['Fine', 'Medium', 'Coarse']

COLORS = {
    'Fine': 'tab:blue',
    'Medium': 'tab:orange',
    'Coarse': 'tab:green',
}
INPUT_SHORT = {
    'TSS_mg_L': 'TSS',
    'pressure_kPa': 'P',
    'nozzle_diameter_mm': 'D$_{n}$',
    'duration_hrs': 't',
    'particle_diameter_um': 'd$_{p}$',
}


def _magnitude(df, col):
    return df[col].pint.magnitude


def _save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _panel_label(ax, letter):
    ax.text(
        0.02,
        0.98,
        f'({letter})',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=11,
        fontweight='bold',
    )


def fig_01_schematic() -> None:
    """Five-stage uncertainty-propagation chain (no data)."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)

    stages = [
        ('1 · Inputs (LHS)\nTSS, P, D$_{n}$, t, d$_{p}$', 'Traveling\n20,000 samples'),
        (
            '2 · Derived hydraulics\n$V = C_d\\sqrt{2P/\\rho_w}$\n'
            '$\\varphi = \\mathrm{TSS}\\cdot10^{-3}/\\rho_s$\n$t_{res} = D_n / V$',
            'Velocity, volume\nfraction, residence',
        ),
        (
            '3 · Physical modifiers\nStokes · size ratio · shear · settling',
            'Four bounded\nfactors $\\phi \\in [0.5,\\ 10]$',
        ),
        (
            '4 · Clogging potential\n$X = X_{base}\\, \\phi$\nclipped to $[0,\\ 50]$',
            'Dimensionless\nindex',
        ),
        (
            '5 · Probability\n$P_c = \\dfrac{1}{1+e^{-\\gamma(X-x_0)}}$',
            'Logistic mapping\n$P_c \\in [0,\\ 1]$',
        ),
        ('6 · Risk tiers\nLow · Moderate · High', '$P_c < 0.30$ · 0.30–0.50 · > 0.50'),
    ]

    W, G, y = 1.45, 0.22, 2.0
    x0 = 0.5
    xs = [x0 + i * (W + G) for i in range(len(stages))]
    for x, (title, sub) in zip(xs, stages):
        ax.add_patch(
            plt.Rectangle(
                (x, y - 0.85), W, 1.7, fc='#eef2f7', ec='#34495e', lw=1.4, zorder=2
            )
        )
        ax.text(
            x + W / 2,
            y + 0.42,
            title,
            ha='center',
            va='center',
            fontsize=8.4,
            fontweight='bold',
        )
        ax.text(
            x + W / 2,
            y - 0.55,
            sub,
            ha='center',
            va='center',
            fontsize=7.2,
            color='#5d6d7e',
        )
        if x != xs[-1]:
            ax.annotate(
                '',
                xy=(x + W + G * 0.55, y),
                xytext=(x + W + G * 0.45, y),
                arrowprops=dict(arrowstyle='-|>', color='#34495e', lw=1.6),
            )
    ax.text(
        5,
        3.95,
        'Uncertainty propagation: five stage chain',
        ha='center',
        fontsize=12,
        fontweight='bold',
    )
    _save(fig, 'fig_01_schematic.png')


def fig_02_convergence(conv) -> None:
    """Running mean (95% CI band) and running std of P_c vs sample size."""
    c = conv[conv['output'] == 'clogging_probability'].sort_values('sample_size')
    n = c['sample_size'].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(n, c['mean'].to_numpy(), '-o', color='tab:blue', lw=1.8, ms=4)
    ax1.fill_between(
        n,
        c['ci_lower'].to_numpy(),
        c['ci_upper'].to_numpy(),
        color='tab:blue',
        alpha=0.15,
    )
    ax1.set_xlabel('Monte Carlo sample size')
    ax1.set_ylabel('Mean $P_c$')
    ax1.set_ylim(0.52, 0.56)
    ax1.grid(alpha=0.3)
    _panel_label(ax1, 'a')

    ax2.plot(n, c['std'].to_numpy(), '-s', color='tab:orange', lw=1.8, ms=4)
    ax2.set_xlabel('Monte Carlo sample size')
    ax2.set_ylabel('Std. dev. of $P_c$')
    ax2.set_ylim(0.30, 0.32)
    ax2.grid(alpha=0.3)
    _panel_label(ax2, 'b')

    fig.suptitle('Convergence of clogging probability', fontsize=12)
    fig.tight_layout()
    _save(fig, 'fig_02_convergence.png')


def _bootstrap_mean_ci(values, rng, n_boot=2000):
    means = np.array(
        [
            np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_boot)
        ]
    )
    return np.percentile(means, [2.5, 97.5])


def fig_03_distribution(df) -> None:
    """Histogram with thresholds, CDF, box plots by class, class means + CI."""
    pc = _magnitude(df, 'clogging_probability')
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.hist(pc, bins=60, color='tab:blue', alpha=0.75)
    for thr in (config.RISK_LOW_THRESHOLD, config.RISK_MODERATE_THRESHOLD):
        ax1.axvline(thr, color='tab:red', ls='--', lw=1.2)
    ax1.annotate(
        'Low/Moderate/High thresholds at 0.30 and 0.50',
        xy=(0.30, ax1.get_ylim()[1] * 0.85),
        fontsize=7.5,
        color='tab:red',
        xytext=(0.02, ax1.get_ylim()[1] * 0.75),
        arrowprops=dict(arrowstyle='-', color='tab:red', lw=0.8),
    )
    ax1.set_xlabel('Clogging probability $P_c$')
    ax1.set_ylabel('Count (20,000 LHS samples)')
    ax1.grid(alpha=0.3)
    _panel_label(ax1, 'a')

    pc_sorted = np.sort(pc)
    ax2.plot(pc_sorted, np.linspace(0, 1, len(pc_sorted)), color='tab:green', lw=1.6)
    ax2.set_xlabel('$P_c$')
    ax2.set_ylabel('Cumulative probability')
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.3)
    _panel_label(ax2, 'b')

    data = [
        _magnitude(df[df['particle_size_range'] == c], 'clogging_probability')
        for c in CLASS_ORDER
    ]
    bp = ax3.boxplot(data, tick_labels=CLASS_ORDER, patch_artist=True)
    for patch, c in zip(bp['boxes'], CLASS_ORDER):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.6)
    ax3.set_ylabel('$P_c$')
    ax3.grid(alpha=0.3, axis='y')
    _panel_label(ax3, 'c')

    class_means, class_cis = [], []
    for c in CLASS_ORDER:
        vals = _magnitude(df[df['particle_size_range'] == c], 'clogging_probability')
        class_means.append(np.mean(vals))
        class_cis.append(_bootstrap_mean_ci(vals, rng))
    class_means = np.array(class_means)
    class_cis = np.array(class_cis)
    xpos = np.arange(3)
    ax4.bar(xpos, class_means, color=[COLORS[c] for c in CLASS_ORDER], alpha=0.8)
    ax4.errorbar(
        xpos,
        class_means,
        yerr=[class_means - class_cis[:, 0], class_cis[:, 1] - class_means],
        fmt='none',
        ecolor='k',
        capsize=4,
    )
    ax4.set_xticks(xpos, CLASS_ORDER)
    ax4.set_ylabel('Mean $P_c$')
    ax4.set_ylim(0, 1.0)
    ax4.grid(alpha=0.3, axis='y')
    _panel_label(ax4, 'd')

    fig.suptitle(
        'Distribution of simulated clogging probability (N = 20,000)', fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, 'fig_03_distribution.png')


def _src_coefficients(df):
    cols = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
    ]
    X = np.column_stack([_magnitude(df, c) for c in cols])
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    y = _magnitude(df, 'clogging_probability')
    beta, *_ = np.linalg.lstsq(X, y - y.mean(), rcond=None)
    pred = X @ beta + y.mean()
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return list(zip(cols, beta)), r2


def fig_04_sensitivity(df, sob, mor) -> None:
    """SRC (a), Sobol S1/ST with CIs (b), Morris mu* vs sigma (c)."""
    src, r2 = _src_coefficients(df)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.8))

    cols, betas = zip(*src)
    labels = [INPUT_SHORT[c] for c in cols]
    ax1.bar(range(5), betas, color=['tab:red' if b < 0 else 'tab:blue' for b in betas])
    ax1.set_xticks(range(5), labels)
    ax1.axhline(0, color='k', lw=0.7)
    ax1.set_ylabel('Standardised regression coefficient')
    ax1.set_title(f'SRC ($R^2$ = {r2:.2f})', fontsize=10)
    ax1.grid(alpha=0.3, axis='y')
    _panel_label(ax1, 'a')

    inputs = sob['input'].to_numpy()
    x = np.arange(5)
    w = 0.38
    ax2.bar(
        x - w / 2,
        sob['S1'].to_numpy(),
        w,
        yerr=[
            sob['S1'].to_numpy() - sob['S1_lo'].to_numpy(),
            sob['S1_hi'].to_numpy() - sob['S1'].to_numpy(),
        ],
        label='$S_1$',
        color='tab:blue',
        alpha=0.8,
        capsize=3,
        error_kw={'lw': 0.8},
    )
    ax2.bar(
        x + w / 2,
        sob['ST'].to_numpy(),
        w,
        yerr=[
            sob['ST'].to_numpy() - sob['ST_lo'].to_numpy(),
            sob['ST_hi'].to_numpy() - sob['ST'].to_numpy(),
        ],
        label='$S_T$',
        color='tab:orange',
        alpha=0.8,
        capsize=3,
        error_kw={'lw': 0.8},
    )
    ax2.set_xticks(x, [INPUT_SHORT[c] for c in inputs])
    ax2.set_ylabel('Sobol index')
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(alpha=0.3, axis='y')
    _panel_label(ax2, 'b')

    mu = mor['mu_star'].to_numpy()
    sg = mor['sigma'].to_numpy()
    ax3.scatter(mu, sg, s=45, color='tab:green', zorder=3)
    for x_l, y_l, name in zip(mu, sg, mor['input'].to_numpy()):
        ax3.annotate(
            INPUT_SHORT[name],
            (x_l, y_l),
            textcoords='offset points',
            xytext=(6, 5),
            fontsize=10,
        )
    ax3.set_xlabel('$\\mu^*$ (mean absolute effect)')
    ax3.set_ylabel('$\\sigma$ (interaction / non-linearity)')
    ax3.grid(alpha=0.3)
    _panel_label(ax3, 'c')

    fig.suptitle(
        'Global sensitivity analysis by three complementary methods', fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, 'fig_04_sensitivity.png')


def _binned_mean(x, y, n_bins=12, log=False):
    if log:
        edges = np.logspace(np.log10(max(x.min(), 1e-9)), np.log10(x.max()), n_bins + 1)
    else:
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.array(
        [np.mean(y[idx == i]) if np.any(idx == i) else np.nan for i in range(n_bins)]
    )
    return centers, means


def fig_05_sediment_class(df) -> None:
    """Violins (a), risk proportions (b), marginal effects (c-f)."""
    pc = _magnitude(df, 'clogging_probability')

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

    vals = [
        _magnitude(df[df['particle_size_range'] == c], 'clogging_probability')
        for c in CLASS_ORDER
    ]
    vp = ax1.violinplot(vals, positions=[0, 1, 2], showmeans=True, showextrema=True)
    for body, c in zip(vp['bodies'], CLASS_ORDER):
        body.set_facecolor(COLORS[c])
        body.set_alpha(0.6)
    ax1.set_xticks([0, 1, 2], CLASS_ORDER)
    ax1.set_ylabel('$P_c$')
    ax1.grid(alpha=0.3, axis='y')
    _panel_label(ax1, 'a')

    risk_order = ['Low', 'Moderate', 'High']
    props = np.zeros((3, 3))
    for i, c in enumerate(CLASS_ORDER):
        for j, r in enumerate(risk_order):
            sub = df[(df['particle_size_range'] == c) & (df['clogging_risk'] == r)]
            props[i, j] = len(sub)
    props = props / props.sum(1, keepdims=True) * 100
    bottom = np.zeros(3)
    risk_colors = {'Low': '#8bc34a', 'Moderate': '#ffc107', 'High': '#e53935'}
    for j, r in enumerate(risk_order):
        ax2.bar(
            [0, 1, 2],
            props[:, j],
            bottom=bottom,
            label=r,
            color=risk_colors[r],
            width=0.6,
        )
        bottom += props[:, j]
    ax2.set_xticks([0, 1, 2], CLASS_ORDER)
    ax2.set_ylabel('Share of class (%)')
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
    _panel_label(ax2, 'b')

    dp = _magnitude(df, 'particle_diameter_um')
    cx, my = _binned_mean(dp, pc, log=True)
    ax3.plot(cx, my, '-o', ms=3, color='tab:blue')
    ax3.set_xscale('log')
    ax3.set_xlabel('Particle diameter $d_p$ ($\\mu$m)')
    ax3.set_ylabel('Mean $P_c$')
    ax3.grid(alpha=0.3)
    _panel_label(ax3, 'c')

    panels = [
        (ax4, 'nozzle_diameter_mm', 'Nozzle diameter $D_n$ (mm)'),
        (ax5, 'TSS_mg_L', 'TSS (mg L$^{-1}$)'),
        (ax6, 'duration_hrs', 'Operating duration $t$ (h)'),
    ]
    for ax, col, xlab in panels:
        xv = _magnitude(df, col)
        cx, my = _binned_mean(xv, pc)
        ax.plot(cx, my, '-o', ms=3, color='tab:orange')
        ax.set_xlabel(xlab)
        ax.set_ylabel('Mean $P_c$')
        ax.grid(alpha=0.3)
        _panel_label(
            ax,
            'd' if col == 'nozzle_diameter_mm' else ('e' if col == 'TSS_mg_L' else 'f'),
        )

    fig.suptitle(f'Sediment class behaviour (N = {len(df):,})', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, 'fig_05_sediment_class.png')


def fig_06_modifier_diagnostics(df) -> None:
    """Modifier means by class (a), factor correlation with Pc (b),
    phi_V(v) with envelope (c), Pc vs dp/Dn with bridging threshold (d)."""
    pc = _magnitude(df, 'clogging_probability')
    factors = [
        'stokes_factor',
        'dp_dn_factor',
        'velocity_shear_factor',
        'settling_velocity_factor',
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    (ax1, ax2), (ax3, ax4) = axes

    x = np.arange(len(factors))
    w = 0.26
    for i, c in enumerate(CLASS_ORDER):
        sub = df[df['particle_size_range'] == c]
        means = [_magnitude(sub, f).mean() for f in factors]
        ax1.bar(x + (i - 1) * w, means, w, label=c, color=COLORS[c], alpha=0.85)
    ax1.set_xticks(x, [f.replace('_', ' ').replace('factor', '') for f in factors])
    ax1.set_ylabel('Mean modifier factor')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis='y')
    _panel_label(ax1, 'a')

    corrs = []
    for f in factors:
        v = _magnitude(df, f)
        corrs.append(np.corrcoef(v, pc)[0, 1])
    ax2.bar(x, corrs, color=['tab:red' if r < 0 else 'tab:blue' for r in corrs])
    ax2.axhline(0, color='k', lw=0.7)
    ax2.set_xticks(x, [f.replace('_', ' ').replace('factor', '') for f in factors])
    ax2.set_ylabel('Pearson r with $P_c$')
    ax2.grid(alpha=0.3, axis='y')
    _panel_label(ax2, 'b')

    v = np.linspace(0, 30, 300)
    phiV = calculate_velocity_shear_factor(
        cast(pint.Quantity, ureg.Quantity(v, 'm/s'))
    ).magnitude
    ax3.plot(
        v,
        phiV,
        lw=1.8,
        color='tab:purple',
        label='$\\phi_V(v)$   ($V_{th}$ = 8 m s$^{-1}$)',
    )
    ax3.axvline(8, color='k', ls='--', lw=1.1, alpha=0.7)
    ax3.axvspan(
        12,
        24,
        color='tab:orange',
        alpha=0.12,
        label='operating envelope\n12–24 m s$^{-1}$',
    )
    ax3.set_xlabel('Orifice velocity (m s$^{-1}$)')
    ax3.set_ylabel('Velocity-shear factor $\\phi_V$')
    ax3.text(8, 0.28, '$V_{th}$', ha='center', fontsize=8, color='k')
    ax3.text(
        14,
        0.34,
        '$\\phi_V \\approx 0.94 \\rightarrow 0.84$',
        fontsize=7.5,
        color='tab:orange',
    )
    ax3.legend(fontsize=8, loc='lower left')
    ax3.grid(alpha=0.3)
    _panel_label(ax3, 'c')

    ratio = _magnitude(df, 'dp_dn_ratio')
    ax4.scatter(ratio, pc, s=2, alpha=0.12, color='tab:blue', rasterized=True)
    ax4.axvline(config.DP_DN_OBSTRUCTION_THRESHOLD, color='tab:red', ls='--', lw=1.2)
    ax4.text(
        config.DP_DN_OBSTRUCTION_THRESHOLD,
        0.98,
        'bridging onset $d_p/D_n$ = 0.05',
        fontsize=7.5,
        color='tab:red',
        transform=ax4.get_xaxis_transform(),
    )
    ax4.set_xlabel('Size ratio $d_p / D_n$')
    ax4.set_ylabel('$P_c$')
    ax4.grid(alpha=0.3)
    _panel_label(ax4, 'd')

    fig.suptitle('Physical modifier diagnostics', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, 'fig_06_modifier_diagnostics.png')


def _verify(df):
    pc = _magnitude(df, 'clogging_probability')
    print(
        f'mean {pc.mean():.3f}  std {pc.std(ddof=1):.3f}  '
        f'median {np.median(pc):.3f}  p5 {np.percentile(pc, 5):.3f}  '
        f'p95 {np.percentile(pc, 95):.3f}  CV {pc.std(ddof=1) / pc.mean():.2f}'
    )
    for c in CLASS_ORDER:
        sub = df[df['particle_size_range'] == c]
        print(
            f'{c}: N={len(sub)} meanX={_magnitude(sub, "X").mean():.2f} '
            f'meanPc={_magnitude(sub, "clogging_probability").mean():.3f}'
        )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    fig_01_schematic()
    conv = run_convergence_study(
        sample_sizes=[2_500, 5_000, 10_000, 15_000, 20_000], chunk_size=2_000
    )
    fig_02_convergence(conv)
    conv_disp = conv[conv['output'] == 'clogging_probability'].sort_values(
        'sample_size'
    )
    print(
        'convergence:',
        list(
            zip(
                conv_disp['sample_size'],
                conv_disp['mean'].round(3),
                conv_disp['std'].round(3),
            )
        ),
    )

    df = run_simulation(total_samples=20_000, chunk_size=2_000)
    _verify(df)
    fig_03_distribution(df)

    sob = calculate_sobol_indices(n_samples=10_000)
    print('sobol:')
    print(sob.round(3).to_string(index=False))
    mor = calculate_morris_indices(n_samples=1_000, n_levels=4)
    print(
        'morris (mu*, sigma):',
        [
            (r['input'], round(r['mu_star'], 3), round(r['sigma'], 3))
            for _, r in mor.iterrows()
        ],
    )
    fig_04_sensitivity(df, sob, mor)

    fig_05_sediment_class(df)
    fig_06_modifier_diagnostics(df)
    print(f'figures written to {OUT_DIR}')


if __name__ == '__main__':
    main()
