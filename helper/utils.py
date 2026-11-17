"""Small helpers for the Lorehaven notebooks.

The notebooks run every step themselves. This file only holds the pieces that would otherwise
be copied for each metric or analysis.

Part 1 (01_price_cut.ipynb): region-season cells, the WLS regression, coefficients, and the
levels / placebo plots.
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
from scipy import stats
from scipy.special import expit

SEASONS = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
SEASON_ABBR = ['Win', 'Spr', 'Sum', 'Fall']
FIRST_YEAR = 2017
#  t runs from 0 (winter 2017) to 11 (fall 2019).
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
def fit_ltv(data, y, fixed_effects=False):
    # LTV on discount depth ($1 off is the base), SEs clustered by customer.
    # Depth is randomized, so by default this is depth-only: the coefficients are differences of raw means,
    # and the regression only attaches confidence intervals. fixed_effects=True is a robustness check.
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
# PART 3&4: JOINING & CANCELLATION MODEL
# ==================================================================================================
def fit_logit(formula, data):
    # Logit with standard errors clustered by customer (a customer appears in several seasons)
    return smf.logit(formula, data=data).fit(disp=0, cov_type='cluster', cov_kwds={'groups': data['customer_id']})

def add_terms_join(x):
    """Price and exposure terms of the joining model. Used for the data and for tool scenarios."""
    x = x.copy()
    x['price_j'] = x['price'] - 12
    x['price_sq'] = x['price_j'] ** 2
    x['pod_soc'] = x['pod'] * x['soc']
    x['sqrt_pod_soc'] = np.sqrt(x['pod_soc'])
    x['pod_zero'] = x['pod'].eq(0).astype(int)
    x['soc_zero'] = x['soc'].eq(0).astype(int)
    return x

def add_terms_cancel(x):
    """Price and exposure terms of the model, rebuilt from price, price_lag, pod, soc, pod_last and soc_last."""
    x = x.copy()
    x['price_c'] = x['price'] - 12                    # centred at today's $12 base, as in the joining model
    x['price_sq'] = x['price_c'] ** 2
    x['lag_c'] = x['price_lag'] - 12
    for a, b, s in [('pod', 'soc', ''), ('pod_last', 'soc_last', '_last')]:
        x['pod_soc' + s] = x[a] * x[b]
        x['sqrt_pod_soc' + s] = np.sqrt(x['pod_soc' + s])
        x['pod_zero' + s] = x[a].eq(0).astype(int)
        x['soc_zero' + s] = x[b].eq(0).astype(int)
    return x


def price_checks(formula, data, base):
    """Three checks on the price terms of `formula`: shape, a change at the cut, and the slope by season."""
    dummies = fit_logit(formula.replace('price_c + price_sq', 'C(price)'), data)
    lr, dof = 2 * (dummies.llf - base.llf), dummies.df_model - base.df_model
    print(f"Price dummies vs quadratic: LR = {lr:.1f} on {dof:.0f} df, p = {stats.chi2.sf(lr, dof):.2f}")
    linear = fit_logit(formula.replace('price_c + price_sq', 'price_c'), data)
    print(f"Quadratic vs straight line: LR = {2 * (base.llf - linear.llf):.1f} on 1 df")
    at_cut = fit_logit(formula.replace('price_sq', 'price_sq + price_post'),
                       data.assign(price_post=data['price_c'] * data['post']))
    print(f"Separate slope after the cut: {at_cut.params['price_post']:+.3f}, LR = {2 * (at_cut.llf - base.llf):.1f} on 1 df")
    seasons = sorted(data['t'].unique())
    slope_terms = [f'price_t{k}' for k in seasons]
    by_season = data.assign(**{f'price_t{k}': data['price_c'] * data['t'].eq(k) for k in seasons})
    fit_s = fit_logit(formula.replace('price_c', ' + '.join(slope_terms), 1), by_season)
    return pd.Series(fit_s.params[slope_terms].to_numpy(), index=[period_label(k) for k in seasons], name='slope at $10')


def coef_table_logit(model, terms):
    """Estimate, CI and p-value for the given terms of a logit (Parts 3 and 4). Separate name: coef_table is Part 1's."""
    ci = model.conf_int()
    return pd.DataFrame({'estimate': model.params[terms], 'ci_low': ci.loc[terms, 0], 'ci_high': ci.loc[terms, 1],
                         'p_value': model.pvalues[terms]}).round(4)

def match_season(target, candidates, log_spend):
    """The candidate season whose planned podcast and social spend (in logs) is closest to the target season's."""
    return ((log_spend.loc[candidates] - log_spend.loc[target]) ** 2).sum(axis=1).idxmin()


def with_exposure_from(x, season, exposure, add_terms=add_terms_join):
    """Replace each customer's exposure with their own exposure in `season`, and rebuild the model terms
    with `add_terms` (add_terms_join for the joining model, add_terms_cancel for the cancellation model)."""
    past = exposure[exposure['t'].eq(season)].drop(columns='t')
    x = x.drop(columns=['pod', 'soc']).merge(past, on='customer_id', how='left', validate='one_to_one')
    assert x[['pod', 'soc']].notna().all().all()
    return add_terms(x)
