"""Small helpers for the Lorehaven notebooks.

The notebooks run every step themselves. This file only holds the pieces that would otherwise
be copied for each metric or analysis.

Part 1 (01_price_cut.ipynb): region-season cells, the WLS regression, coefficients, the
historical band, and the event-study / placebo plots.
Part 2 (02_promotion.ipynb): the LTV regression, regression-adjusted means by discount depth,
and one plot.
Part 3 (03_modeling.ipynb): the joining logit (design matrix, fit, predictions), calibration
tables, the intercept shift for unseen seasons, and the average joining curve by price.

Periods are integers t = 0 (winter 2017) ... 11 (fall 2019).
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import optimize, stats
from scipy.special import expit

SEASONS = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
SEASON_ABBR = ['Win', 'Spr', 'Sum', 'Fall']
FIRST_YEAR = 2017
#  t runs from 0 (winter 2017) to 11 (fall 2019).
REF_T = 8          # winter 2019: last season before the cut, event-study reference
CUT_T = 9          # spring 2019: first season at the $12 base price
CONTROL = 'panem'  # the only region that kept $15

METRICS = {
    'revenue_pc': {'label': 'Revenue per customer', 'unit': '$ per customer per season'},
    'log_revenue_pc': {'label': 'Log revenue per customer', 'unit': 'log points'},
    'member_pct': {'label': 'Membership percentage', 'unit': 'percentage points'},
}

# Chart palette (same as the EDA notebooks): blue = cut regions, orange = panem
TREATED_C, CONTROL_C, THIRD_C = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BAND_C = '#e4e3dc'

def period_label(t, short=False):
    # t -> 'Win 2019'. short=True gives a two-line tick label for winters and the season name elsewhere
    season, year = SEASON_ABBR[t % 4], FIRST_YEAR + t // 4
    if short:
        return f'{season}\n{year}' if t % 4 == 0 else season
    return f'{season} {year}'

def load_panel(path='data/customer_panel.csv', extra_cols=()):
    # Read the columns the analysis needs (plus `extra_cols`) and add the period index t
    cols = ['region_name', 'year', 'season', 'membership_price', 'sign_up_flag', *extra_cols]
    panel = pd.read_csv(path, usecols=cols)
    panel['t'] = (panel['year'] - FIRST_YEAR) * 4 + panel['season'].str.lower().map(SEASONS)
    return panel

# ==================================================================================================
# PART 1 - PRICE CUT DiD
# ==================================================================================================
def build_cells(panel):
    # One row per region and season. Metrics are computed after aggregation
    cells = (panel.assign(revenue=panel['membership_price'] * panel['sign_up_flag'])
                  .groupby(['region_name', 't'])
                  .agg(n=('sign_up_flag', 'size'), members=('sign_up_flag', 'sum'), revenue=('revenue', 'sum'))
                  .reset_index())
    # metrics
    cells['revenue_pc'] = cells['revenue'] / cells['n']
    cells['log_revenue_pc'] = np.log(cells['revenue_pc'])
    cells['member_pct'] = 100 * cells['members'] / cells['n']
    # features
    cells['treated'] = cells['region_name'].ne(CONTROL).astype(int)
    cells['post'] = cells['t'].ge(CUT_T).astype(int)
    cells['treated_post'] = cells['treated'] * cells['post']
    cells['season_of_year'] = cells['t'] % 4
    cells['westeros_winter'] = (cells['region_name'].eq('westeros') & cells['season_of_year'].eq(0)).astype(int)
    return cells


# ---------------------------------------------------------------- estimation
def fit_wls(data, y, terms, westeros_adj=True, extra=()):
    # WLS of y on region FE + season FE + terms (+ Westeros-winter dummy), weighted by customer count.
    rhs = ['C(region_name)', 'C(t)', *terms, *extra]
    if westeros_adj:
        rhs.append('westeros_winter')
    return smf.wls(f"{y} ~ {' + '.join(rhs)}", data=data, weights=data['n']).fit(cov_type='HC1')


def coef_table(fit, terms, level=0.95):
    # Estimate, SE, CI and p-value for the given terms of a fitted model.
    ci = fit.conf_int(alpha=1 - level).loc[terms]
    return pd.DataFrame({'estimate': fit.params[terms], 'se': fit.bse[terms],
                         'ci_low': ci[0], 'ci_high': ci[1], 'p_value': fit.pvalues[terms]})


def event_points(fit, periods, ref=REF_T):
    """Event-study points from a fit with ev_<k> terms, plus the reference season fixed at 0."""
    tab = coef_table(fit, [f'ev_{k}' for k in periods if k != ref])
    tab.loc[f'ev_{ref}'] = [0.0, 0.0, 0.0, 0.0, np.nan]
    tab['t'] = [int(name.split('_')[1]) for name in tab.index]
    tab = tab.sort_values('t').reset_index(drop=True)
    tab.insert(0, 'season', tab['t'].map(period_label))
    tab['post'] = tab['t'].ge(CUT_T)
    tab['is_ref'] = tab['t'].eq(ref)
    return tab


def historical_band(points, level=0.95):
    """Band from the season-to-season spread of the pre-cut points (the reference season's 0 included).

    half_width     : one new season, center ± t * sd * sqrt(1 + 1/n_pre)
    half_width_avg : post-cut average minus pre-cut average, ± t * sd * sqrt(1/n_post + 1/n_pre)
    """
    pre = points.loc[points['t'].lt(CUT_T), 'estimate']
    post = points.loc[points['t'].ge(CUT_T), 'estimate']
    n_pre, n_post = len(pre), len(post)
    sd = pre.std(ddof=1)
    t_crit = stats.t.ppf(0.5 + level / 2, df=n_pre - 1)
    return pd.Series({
        'n_pre': n_pre, 'n_post': n_post, 'sd': sd, 't_crit': t_crit,
        'center': pre.mean(),
        'half_width': t_crit * sd * np.sqrt(1 + 1 / n_pre),
        'post_avg_gap': post.mean() - pre.mean(),
        'half_width_avg': t_crit * sd * np.sqrt(1 / n_post + 1 / n_pre),
    })


# ---------------------------------------------------------------- plots
def apply_style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.8,
        'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'lines.linewidth': 2, 'legend.frameon': False, 'font.size': 9,
        'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    })


def mark_cut(ax):
    """Dashed line between the last $15 season and the first $12 season."""
    ax.axvline(CUT_T - 0.5, color=INK2, lw=1, ls='--', zorder=1)
    ax.text(CUT_T - 0.4, 0.98, 'price cut\n\\$15 → \\$12', transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontsize=8, color=INK2)


def _period_axis(ax, periods):
    periods = list(periods)
    ax.set_xticks(periods, [period_label(t, short=True) for t in periods])
    ax.set_xlim(min(periods) - 0.5, max(periods) + 0.5)


def plot_levels(cells, y, ax=None):
    """Customer-weighted average of the five cut regions vs panem, by season."""
    ax = ax or plt.subplots(figsize=(8, 3.4))[1]
    g = cells.assign(wy=cells[y] * cells['n']).groupby(['treated', 't'])[['wy', 'n']].sum()
    levels = (g['wy'] / g['n']).unstack('treated')
    ax.plot(levels.index, levels[1], marker='o', ms=5, color=TREATED_C, label='cut regions (5, customer-weighted)')
    ax.plot(levels.index, levels[0], marker='o', ms=5, color=CONTROL_C, label='panem (kept \\$15)')
    mark_cut(ax)
    _period_axis(ax, levels.index)
    ax.set_ylabel(METRICS[y]['unit'])
    ax.set_title(f"{METRICS[y]['label']}: levels")
    ax.legend(loc='upper left')
    return ax


def plot_event_study(points, band, y, compare=None, ax=None):
    """Gap vs panem relative to the reference season, with the historical band and model CIs.

    compare: optional second set of points (e.g. without the Westeros adjustment), drawn hollow.
    """
    ax = ax or plt.subplots(figsize=(8, 4.2))[1]
    x, est, ref = points['t'].to_numpy(), points['estimate'].to_numpy(), points['is_ref'].to_numpy()

    ax.axhspan(band['center'] - band['half_width'], band['center'] + band['half_width'], color=BAND_C, lw=0, zorder=0)
    ax.axhline(band['center'], color=MUTED, lw=1, ls=':', zorder=1)
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    yerr = np.vstack([est - points['ci_low'], points['ci_high'] - est])[:, ~ref]
    ax.errorbar(x[~ref], est[~ref], yerr=yerr, fmt='none', ecolor=TREATED_C, elinewidth=1.2, alpha=0.45, zorder=2)
    ax.plot(x, est, color=TREATED_C, lw=1.5, zorder=2)
    ax.scatter(x[~ref], est[~ref], s=42, color=TREATED_C, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.scatter(x[ref], est[ref], s=42, facecolor=SURFACE, edgecolor=TREATED_C, linewidth=1.5, zorder=3)
    if compare is not None:
        ax.scatter(compare['t'] + 0.18, compare['estimate'], s=20, facecolor='none', edgecolor=INK2, linewidth=1, zorder=3)

    mark_cut(ax)
    _period_axis(ax, x)
    ax.set_ylabel(METRICS[y]['unit'])
    ax.set_title(f"{METRICS[y]['label']}: cut regions minus panem, relative to {period_label(REF_T)}")
    handles = [
        Line2D([], [], color=TREATED_C, marker='o', ms=6, mec=SURFACE, label='gap vs panem (relative to reference)'),
        Line2D([], [], color=TREATED_C, marker='o', ms=6, mfc=SURFACE, ls='none', label='reference season (= 0)'),
        Line2D([], [], color=TREATED_C, alpha=0.45, lw=1.2, label='model-based 95% CI (diagnostic)'),
        Patch(color=BAND_C, label='historical-gap band (95%, pre-cut)'),
        Line2D([], [], color=MUTED, lw=1, ls=':', label='pre-cut average gap'),
    ]
    if compare is not None:
        handles.append(Line2D([], [], color=INK2, marker='o', ms=5, mfc='none', ls='none',
                              label='without Westeros adjustment'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    return ax


def plot_placebos(placebo, did, y, ax=None):
    """Placebo DiDs next to the real DiD. Filled = with the Westeros adjustment, hollow = without.

    placebo : rows with cut, westeros_adj, estimate, ci_low, ci_high
    did     : the real DiD, indexed by westeros_adj (True/False), same columns
    """
    ax = ax or plt.subplots(figsize=(8, 2.6))[1]
    real = did.reset_index().assign(cut=f'real cut: {period_label(CUT_T)}')
    tab = pd.concat([placebo, real], ignore_index=True)
    names = list(dict.fromkeys(tab['cut']))
    ypos = tab['cut'].map({name: len(names) - 1 - i for i, name in enumerate(names)})

    adj = tab['westeros_adj'].astype(bool)
    for is_real, color, label in [(False, MUTED, 'placebo DiD'), (True, TREATED_C, 'real DiD')]:
        m = adj & tab['cut'].str.startswith('real').eq(is_real)
        est, lo, hi = (tab.loc[m, c].to_numpy(dtype=float) for c in ('estimate', 'ci_low', 'ci_high'))
        ax.errorbar(est, ypos[m].to_numpy(), xerr=np.vstack([est - lo, hi - est]), fmt='o', color=color, ms=7,
                    elinewidth=2, capsize=0, label=f'{label} (model-based 95% CI)')
    ax.scatter(tab.loc[~adj, 'estimate'], ypos[~adj] - 0.22, s=22, facecolor='none', edgecolor=INK2,
               linewidth=1, label='without Westeros adjustment')
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(names)), names[::-1])
    ax.set_ylim(-0.7, len(names) - 0.4)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID)
    ax.set_xlabel(METRICS[y]['unit'])
    ax.set_title(f"{METRICS[y]['label']}: placebo cut dates vs the real cut")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i].startswith('without'))   # hollow markers last
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=3)
    return ax


# ==================================================================================================
# PART 2: PROMOTION LTV
# ==================================================================================================
def fit_ltv(data, y, fixed_effects=True):
    # OLS of LTV on discount depth (categorical, $1 off is the base), SEs clustered by customer
    rhs = 'C(depth)' + (' + C(region_name) + C(t)' if fixed_effects else '')
    return smf.ols(f'{y} ~ {rhs}', data=data).fit(cov_type='cluster', cov_kwds={'groups': data['customer_id']})


def depth_means(fit, depths=(1, 2, 3, 4, 5), level=0.95):
    """Average predicted LTV at each depth, with a delta-method CI.

    For each depth d: set every offer to d, predict, and average, so region and offer season keep
    their observed mix. The model is additive, so this equals the average design row with the
    depth dummies switched to d: x'b. Its variance is x' V x, with V the clustered covariance.
    On a depth-only fit this returns the raw means.
    """
    X = pd.DataFrame(fit.model.exog, columns=fit.model.exog_names)
    depth_cols = [c for c in X.columns if c.startswith('C(depth)')]
    b, cov = fit.params.to_numpy(), fit.cov_params().to_numpy()
    z = stats.norm.ppf(0.5 + level / 2)
    rows = {}
    for d in depths:
        x = X.mean()
        x[depth_cols] = 0.0                      # depths[0] is the base level: all depth dummies 0
        if f'C(depth)[T.{d}]' in x:
            x[f'C(depth)[T.{d}]'] = 1.0
        est, se = x.to_numpy() @ b, np.sqrt(x.to_numpy() @ cov @ x.to_numpy())
        rows[d] = {'estimate': est, 'se': se, 'ci_low': est - z * se, 'ci_high': est + z * se}
    return pd.DataFrame(rows).T.rename_axis('depth')


def plot_depth_series(series, title, ylabel='$ per offered customer', colors=None, ax=None):
    """One line per entry of `series` ({label: table indexed by depth with estimate, ci_low, ci_high}).

    Points are slightly offset so overlapping 95% CIs stay readable.
    """
    ax = ax or plt.subplots(figsize=(7.5, 3.8))[1]
    colors = colors or [MUTED, TREATED_C, THIRD_C]
    offsets = np.linspace(-0.08, 0.08, len(series)) if len(series) > 1 else [0.0]
    for (label, tab), color, dx in zip(series.items(), colors, offsets):
        depth = tab.index.to_numpy(dtype=float)
        est, lo, hi = (tab[c].to_numpy(dtype=float) for c in ('estimate', 'ci_low', 'ci_high'))
        ax.errorbar(depth + dx, est, yerr=np.vstack([est - lo, hi - est]), fmt='o-', color=color, ms=6, lw=1.5,
                    elinewidth=2, capsize=0, label=label)
    depths = list(next(iter(series.values())).index)
    ax.set_xticks(depths, [f'\\${int(d)} off' for d in depths])
    ax.set_xlim(min(depths) - 0.4, max(depths) + 0.4)
    ax.axhline(0, color=AXIS, lw=1, zorder=0)
    ax.set_xlabel('discount depth')
    ax.set_ylabel(ylabel.replace('$', '\\$'))
    ax.set_title(title)
    ax.legend(loc='upper left')
    return ax


# ==================================================================================================
# PART 3: JOINING MODEL
# ==================================================================================================
def fit_logit(formula, data):
    # Logit with standard errors clustered by customer (a customer appears in several seasons)
    return smf.logit(formula, data=data).fit(disp=0, cov_type='cluster', cov_kwds={'groups': data['customer_id']})

