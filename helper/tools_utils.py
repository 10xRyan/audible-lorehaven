"""Code behind pricing_forecast_tool.ipynb: load the data and the Part 3 / Part 4 models, forecast, report.

The model terms (price centring, exposure terms) are built by utils.add_terms_join and utils.add_terms_cancel,
the same functions the models were fitted with, so the tool cannot drift from Parts 3 and 4.
"""
import re
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

import helper.utils as u

PANEL, AGGREGATE = 'data/customer_panel.csv', 'data/aggregate.csv'
JOIN_PARAMS, CANCEL_PARAMS = 'outputs/join_model_params.csv', 'outputs/cancel_model_params.csv'      # Parts 3 and 4
HORIZON = 4                                     # the forecast window
AFTER = 4                                       # four more seasons, run only to show what the window leaves out
PRICES = np.arange(7, 16)                       # every price the forecast can charge, $7 to $15
TODAY_OFFER_RATE, TODAY_DEPTHS = 0.25, (1, 2, 3, 4, 5)
HARBOR, BRASS, INK, GREY = '#0086a8', '#b8862b', '#14233b', '#7b8798'
JOIN_EXPOSURE = ['pod', 'soc', 'sqrt_pod_soc', 'pod_soc', 'pod_zero', 'soc_zero']
CANCEL_EXPOSURE = JOIN_EXPOSURE + [name + '_last' for name in JOIN_EXPOSURE]


# ---------------------------------------------------------------- pricing policies
def policy(base, join_discount, offer_rate, member_discount):
    """One uniform setting for a season: every offered prospect gets `join_discount` off, every member `member_discount` off."""
    return dict(base=base, offer_rate=offer_rate, join_depths=(join_discount,), member_depths=(member_discount,))


def today_policy(base=12):
    """Today's policy: a quarter of prospects offered, and a fresh random $1-$5 off every season for offers and members."""
    return dict(base=base, offer_rate=TODAY_OFFER_RATE, join_depths=TODAY_DEPTHS, member_depths=TODAY_DEPTHS)


# ---------------------------------------------------------------- data and model terms
def load_seasons(path=PANEL):
    """One frame per season for the five cut regions, every frame in the same customer order."""
    p = u.load_panel(path, extra_cols=('customer_id', 'social_ad_impressions', 'podcast_ad_listen'))
    p = (p[p['region_name'].ne(u.CONTROL)].sort_values(['customer_id', 't'])
         .rename(columns={'podcast_ad_listen': 'pod', 'social_ad_impressions': 'soc'}))
    prev = p.groupby('customer_id')['sign_up_flag'].shift(1)
    spell = (p['sign_up_flag'].eq(1) & prev.fillna(0).eq(0)).groupby(p['customer_id']).cumsum()
    p['tenure'] = np.where(p['sign_up_flag'].eq(1), p.groupby(['customer_id', spell]).cumcount() + 1, 0)
    return {k: g.set_index('customer_id').sort_index() for k, g in p.groupby('t')}


def _value(params, terms, names):
    """Sum of coefficient x term over `names`."""
    return sum(params[name] * terms[name].to_numpy(float) for name in names)


def _region_index(params, region):
    out = np.zeros(len(region))
    for name, beta in params.items():
        m = re.fullmatch(r'C\(region_name\)\[T\.(.+)\]', name)
        if m:
            out += beta * (region == m.group(1))
    return out


class Models:
    """The saved Part 3 and Part 4 coefficients, with the price parts of both models."""

    def __init__(self, join_path=JOIN_PARAMS, cancel_path=CANCEL_PARAMS):
        self.jp = pd.read_csv(join_path, index_col=0)['coefficient']
        self.cp = pd.read_csv(cancel_path, index_col=0)['coefficient']
        self.tenure_fx = np.array([0.0] + [self.cp[f'C(tenure_group)[T.{g}]'] for g in range(2, 9)])   # groups 1 ... 8+

    @lru_cache(maxsize=None)
    def join_price(self, price):
        x = u.add_terms_join(pd.DataFrame({'price': [price], 'pod': [0], 'soc': [0]}))
        return float(_value(self.jp, x, ['price_j', 'price_sq'])[0])

    @lru_cache(maxsize=None)
    def cancel_price(self, price):
        return float(_value(self.cp, self._cancel_terms(price, 12), ['price_c', 'price_sq'])[0])

    @lru_cache(maxsize=None)
    def cancel_lag(self, last_price):
        return float(_value(self.cp, self._cancel_terms(12, last_price), ['lag_c'])[0])

    @staticmethod
    def _cancel_terms(price, last_price):
        zero = [0]
        return u.add_terms_cancel(pd.DataFrame({'price': [price], 'price_lag': [last_price], 'pod': zero, 'soc': zero,
                                                'pod_last': zero, 'soc_last': zero}))


def load(ad_plan=None):
    """Everything the notebook needs: data, models, and the forecast inputs for the eight seasons after fall 2019.

    The first four are the forecast window; the next four are run only to show what the window leaves out.
    Advertising repeats 2019 season for season, or, with `ad_plan`, is copied from the past seasons whose planned
    spend is closest (see `match_ad_plan`). The same seasons are used again for the four seasons after the window.
    """
    seasons = load_seasons()
    spend = pd.read_csv(AGGREGATE)
    spend['t'] = (spend['year'] - u.FIRST_YEAR) * 4 + spend['season'].str.lower().map(u.SEASONS)
    ctx = dict(seasons=seasons, models=Models(), spend=spend.groupby('t')[['podcast_spend', 'social_spend']].first())
    last = max(seasons)                                                     # fall 2019
    ctx.update(last=last, future=[last + 1 + i for i in range(HORIZON)], n=len(seasons[last]))
    s = seasons[last]
    ctx['start_members'] = int(s['sign_up_flag'].sum())
    ctx['start_revenue'] = (s['membership_price'] * s['sign_up_flag']).sum() / ctx['n']
    ctx['ad_seasons'] = [k - 4 for k in ctx['future']] if ad_plan is None else match_ad_plan(ctx, ad_plan)
    ctx['inputs'] = prepare(ctx, last, ctx['ad_seasons'] * 2)
    ctx['today'] = forecast(ctx, ctx['inputs'], [today_policy()] * (HORIZON + AFTER))
    ctx['backtest_gap'] = _backtest_record(ctx)[1]                          # how far off levels ran on 2019
    print(f"Ready: {ctx['n']:,} customers in five regions, {ctx['start_members']:,} of them members in "
          f"{u.period_label(last)}.")
    return ctx


def match_ad_plan(ctx, ad_plan):
    """Advanced option. For each forecast season, the past season at the same time of year whose planned podcast and
    social spend is closest to the plan (in logs). Its customers' exposure is then copied.

    This changes the advertising backdrop for every pricing setting alike. It is not a spend lever: exposure was not
    randomized, so a difference between two ad plans is not the return on ad spend.
    """
    spend, future = ctx['spend'], ctx['future']
    if len(ad_plan) != len(future) or any(len(pair) != 2 for pair in ad_plan):
        raise ValueError('AD_PLAN must be four (podcast, social) pairs: winter, spring, summer and fall.')
    low, high = spend.min(), spend.max()
    problems, matched = [], []
    for k, (podcast, social) in zip(future, ad_plan):
        if not (low['podcast_spend'] <= podcast <= high['podcast_spend'] and low['social_spend'] <= social <= high['social_spend']):
            problems.append(f"{u.period_label(k)}: ({podcast}, {social}) is outside the planned spend ever seen "
                            f"(podcast {low['podcast_spend']}-{high['podcast_spend']}, social {low['social_spend']}-{high['social_spend']}).")
            continue
        same_time = [t for t in spend.index if t % 4 == k % 4]
        distance = ((np.log(spend.loc[same_time]) - np.log([podcast, social])) ** 2).sum(axis=1)
        matched.append(int(distance.idxmin()))
    if problems:
        raise ValueError('Please change AD_PLAN at the top:\n  ' + '\n  '.join(problems))
    print('Advanced option: ad exposure copied from the past season at the same time of year with the closest planned spend.')
    for k, (podcast, social), m in zip(future, ad_plan, matched):
        print(f"  {u.period_label(k)}: your plan ({podcast}, {social}) -> {u.period_label(m)} "
              f"(planned {spend.loc[m, 'podcast_spend']}, {spend.loc[m, 'social_spend']})")
    print("This changes the advertising backdrop for every setting alike. Differences between two ad plans are not the "
          "return on ad spend: exposure was not randomized.\n")
    return matched


def prepare(ctx, start, matched):
    """Everything about each customer that does not depend on the pricing choices.

    start: last observed season. matched: for each forecast season, the past season whose ad exposure stands in for it.
    """
    seasons, m = ctx['seasons'], ctx['models']
    s0 = seasons[start]
    region = s0['region_name'].to_numpy()
    join_index, cancel_index = [], []
    last = s0[['pod', 'soc']].to_numpy(float)                                # last observed season: real exposure
    for k in matched:
        this = seasons[k][['pod', 'soc']].to_numpy(float)
        x = pd.DataFrame({'pod': this[:, 0], 'soc': this[:, 1], 'pod_last': last[:, 0], 'soc_last': last[:, 1],
                          'price': 12, 'price_lag': 12})
        join_index.append(m.jp['Intercept'] + _region_index(m.jp, region) + _value(m.jp, u.add_terms_join(x), JOIN_EXPOSURE))
        cancel_index.append(m.cp['Intercept'] + _region_index(m.cp, region)
                            + _value(m.cp, u.add_terms_cancel(x), CANCEL_EXPOSURE))
        last = this                                                          # the next season looks back at this one
    member = s0['sign_up_flag'].to_numpy().astype(bool)
    return dict(join_index=join_index, cancel_index=cancel_index, member=member, n=len(s0),
                tenure=np.clip(s0['tenure'].to_numpy(), 1, 8), last_price=s0['membership_price'].to_numpy().astype(int))


# ---------------------------------------------------------------- forecast
def forecast(ctx, inp, policies):
    """Roll forward one season at a time; one policy per season (see `policy` and `today_policy`).

    Each customer carries the probability of being a non-member, or a member with a given tenure (1 ... 8+ seasons)
    and a given last price. The last price matters because the cancellation model uses it, and under today's policy it
    differs from member to member. Each season starts from the season before.
    """
    m, n = ctx['models'], inp['n']
    member = inp['member']
    state = np.zeros((8, len(PRICES), n))                                    # [tenure group, last price, customer]
    idx = np.where(member)[0]
    state[inp['tenure'][idx] - 1, inp['last_price'][idx] - PRICES[0], idx] = 1.0
    p_out = (~member).astype(float)
    rows = []
    for s, pol in enumerate(policies):
        base, rate = pol['base'], pol['offer_rate']
        new = np.zeros_like(state)
        stay, cancel, joined, revenue = (np.zeros(n) for _ in range(4))

        # Members: cancel or stay at this season's member price, then move up one tenure group
        w = 1 / len(pol['member_depths'])
        for g in range(8):
            for j, last_price in enumerate(PRICES):
                here = state[g, j]
                if not here.any():
                    continue
                index = inp['cancel_index'][s] + m.tenure_fx[g] + m.cancel_lag(int(last_price))
                for depth in pol['member_depths']:
                    price = base - depth
                    c = expit(index + m.cancel_price(price))
                    kept = here * w * (1 - c)
                    new[min(g + 1, 7), price - PRICES[0]] += kept
                    stay += kept
                    cancel += here * w * c
                    revenue += kept * price

        # Non-members: join on an offer (at each offered depth) or at the base price
        offers = [(rate / len(pol['join_depths']), base - d, 1) for d in pol['join_depths']] + [(1 - rate, base, 0)]
        for share, price, offered in offers:
            p = expit(inp['join_index'][s] + m.join_price(price) + m.jp['offer'] * offered)
            j = p_out * share * p
            new[0, price - PRICES[0]] += j
            joined += j
            revenue += j * price

        p_out = p_out - joined + cancel
        state = new
        rows.append({'members': (stay + joined).sum(), 'joined': joined.sum(), 'cancelled': cancel.sum(),
                     'revenue': revenue.sum()})
    out = pd.DataFrame(rows)
    out['revenue per customer'] = out['revenue'] / n
    return out


# ---------------------------------------------------------------- the notebook's outputs
def run(ctx, base, join_discount, share_offered, member_discount):
    """Check the choices, forecast them, and print the headline against today's policy."""
    join_price, member_price = base - join_discount, base - member_discount
    problems = []
    if not 10 <= base <= 15: problems.append('BASE_PRICE must be between 10 and 15.')
    if not 0 <= join_discount <= 5: problems.append('JOINING_DISCOUNT must be between 0 and 5.')
    if not 0 <= member_discount <= 5: problems.append('MEMBER_DISCOUNT must be between 0 and 5.')
    if not 0 <= share_offered <= 100: problems.append('SHARE_OFFERED must be between 0 and 100.')
    if min(join_price, member_price) < 7: problems.append('No price can go below $7, the lowest price ever charged. Use a smaller discount.')
    if problems:
        raise ValueError('Please change the pricing choices at the top:\n  ' + '\n  '.join(problems))

    mine = forecast(ctx, ctx['inputs'], [policy(base, join_discount, share_offered / 100, member_discount)] * (HORIZON + AFTER))
    window, after = summarize(mine), summarize(ctx['today'])
    word = lambda x, up, down: up if x > 0 else down
    pct = lambda a, b: 100 * (a / b - 1)
    print(f"Your choices: ${base} base price; ${join_discount} off the first season for {share_offered}% of non-members "
          f"(they pay ${join_price}); ${member_discount} off for members (they pay ${member_price}).\n")
    print(f"Over the next four seasons each customer brings in ${window['revenue']:.2f},")
    print(f"and {window['members']:,.0f} are members by {u.period_label(ctx['future'][-1])}.\n")
    gap, member_gap = window['revenue'] - after['revenue'], window['members'] - after['members']
    print(f"Compared with today's policy (a random $1-$5 off, re-drawn every season): ${abs(gap):.2f} "
          f"{word(gap, 'more', 'less')} per customer ({pct(window['revenue'], after['revenue']):+.1f}%), "
          f"and {abs(member_gap):,.0f} {word(member_gap, 'more', 'fewer')} members.")
    print(f"In total: ${window['revenue'] * ctx['n']:,.0f} against ${after['revenue'] * ctx['n']:,.0f}.\n")
    later_gap = window['revenue_after'] - after['revenue_after']
    print(f"The four seasons after that, if nothing changes: ${abs(later_gap):.2f} {word(later_gap, 'more', 'less')} per "
          f"customer than today's policy ({pct(window['revenue_after'], after['revenue_after']):+.1f}%). This shows whether "
          f"the lead holds once the member base has adjusted.")
    lean = ctx['backtest_gap']
    print(f"In the back-test, levels ran {abs(lean):.0%} {word(lean, 'high', 'low')}; the difference from today's policy "
          f"is the more reliable number.")

    notes = []
    if member_price >= 15: notes.append('No member has ever been charged $15, so the member side goes one dollar beyond the data.')
    if base < 12: notes.append('The base price has never been below $12. Prices this low were only seen as discounts, so treat this as rougher.')
    if share_offered > 50: notes.append('Offers have only ever gone to about a quarter of non-members. This assumes the rest would respond the same way.')
    if notes:
        print('\nCaution:\n  - ' + '\n  - '.join(notes))
    return mine


def summarize(run):
    """Revenue per customer over the window and over the four seasons after it, and members at the end of the window."""
    return {'revenue': run['revenue per customer'].iloc[:HORIZON].sum(), 'members': run['members'].iloc[HORIZON - 1],
            'revenue_after': run['revenue per customer'].iloc[HORIZON:].sum()}


def _grid(ctx, inputs, base, share_offered, seasons):
    """Every joining and member discount (prices of $7 or more), each forecast for `seasons` seasons."""
    for join_discount in range(6):
        for member_discount in range(6):
            if min(base - join_discount, base - member_discount) >= 7:
                setting = policy(base, join_discount, share_offered / 100, member_discount)
                yield join_discount, member_discount, summarize(forecast(ctx, inputs, [setting] * seasons))


def best_settings(ctx, base, share_offered, top=10):
    """Every joining and member discount at the given base price and offer share, ranked by four-season revenue."""
    today = summarize(ctx['today'])
    rows = [{'Joining discount': j, 'Member discount': m, 'Revenue per customer, 4 seasons': run['revenue'],
             'vs today': run['revenue'] / today['revenue'] - 1,
             'Members at the end': run['members'], 'Members vs today': run['members'] - today['members'],
             'Next 4 seasons, vs today': run['revenue_after'] / today['revenue_after'] - 1}
            for j, m, run in _grid(ctx, ctx['inputs'], base, share_offered, HORIZON + AFTER)]
    table = (pd.DataFrame(rows).sort_values('Revenue per customer, 4 seasons', ascending=False).head(top)
             .reset_index(drop=True))
    table.index = table.index + 1
    print(f"Top {top} of {len(rows)} settings at a ${base} base with {share_offered}% of non-members offered. "
          f"Today's policy: ${today['revenue']:.2f} per customer, {today['members']:,.0f} members at the end.")
    return table.style.format({'Joining discount': '${}'.format, 'Member discount': '${}'.format,
                               'Revenue per customer, 4 seasons': '${:,.2f}'.format, 'vs today': '{:+.1%}'.format,
                               'Members at the end': '{:,.0f}'.format, 'Members vs today': '{:+,.0f}'.format,
                               'Next 4 seasons, vs today': '{:+.1%}'.format})


def robustness(ctx, base, share_offered, top=8):
    """The settings ranking with advertising copied from each past year in turn (2017, 2018, 2019).

    If the same settings stay on top in every year, the pricing advice does not hinge on the marketing assumption.
    """
    seasons, spend = ctx['seasons'], ctx['spend']
    columns, winners = {}, []
    for year in sorted({t // 4 for t in seasons}):
        ad_seasons = [4 * year + k % 4 for k in ctx['future']]
        inputs = prepare(ctx, ctx['last'], ad_seasons)
        today = summarize(forecast(ctx, inputs, [today_policy()] * HORIZON))
        label = (f"{u.FIRST_YEAR + year} ads (podcast {spend.loc[ad_seasons, 'podcast_spend'].sum()}, "
                 f"social {spend.loc[ad_seasons, 'social_spend'].sum()})")
        columns[label] = pd.Series({(j, m): run['revenue'] / today['revenue'] - 1
                                    for j, m, run in _grid(ctx, inputs, base, share_offered, HORIZON)})
        best = columns[label].idxmax()
        winners.append(f"{u.FIRST_YEAR + year}: ${best[0]} to join, ${best[1]} for members")
    table = pd.DataFrame(columns)
    table['Average'] = table.mean(axis=1)
    ranks = table.drop(columns='Average').rank(ascending=False).astype(int)
    table['Worst rank'] = ranks.max(axis=1)
    table = table.sort_values('Average', ascending=False).head(top)
    table.index = [f'${j} to join, ${m} for members' for j, m in table.index]
    print(f"Four-season revenue per customer against today's policy, at a ${base} base with {share_offered}% offered, "
          f"under each past year's advertising.\nBest setting: " + '; '.join(winners) + '.')
    return table.style.format({c: '{:+.1%}'.format for c in table.columns if c != 'Worst rank'})


def season_table(ctx, mine):
    """Season-by-season table for the chosen settings, with today's policy alongside."""
    mine, today, n = mine.iloc[:HORIZON], ctx['today'].iloc[:HORIZON], ctx['n']
    table = pd.DataFrame({'Season': [u.period_label(k) for k in ctx['future']], 'Members': mine['members'],
                          'Joined': mine['joined'], 'Cancelled': mine['cancelled'],
                          'Revenue per customer': mine['revenue per customer'], 'Total revenue': mine['revenue'],
                          "Members, today's policy": today['members'],
                          "Revenue per customer, today's policy": today['revenue per customer']})
    first = {'Season': u.period_label(ctx['last']) + ' (actual)', 'Members': ctx['start_members'],
             'Revenue per customer': ctx['start_revenue'], 'Total revenue': ctx['start_revenue'] * n,
             "Members, today's policy": ctx['start_members'], "Revenue per customer, today's policy": ctx['start_revenue']}
    totals = {'Season': 'Four seasons', 'Joined': mine['joined'].sum(), 'Cancelled': mine['cancelled'].sum(),
              'Revenue per customer': mine['revenue per customer'].sum(), 'Total revenue': mine['revenue'].sum(),
              "Revenue per customer, today's policy": today['revenue per customer'].sum()}
    table = pd.concat([pd.DataFrame([first]), table, pd.DataFrame([totals])], ignore_index=True)[table.columns].set_index('Season')
    money, whole = '${:,.2f}'.format, '{:,.0f}'.format
    return table.style.format({'Members': whole, 'Joined': whole, 'Cancelled': whole, 'Revenue per customer': money,
                               'Total revenue': '${:,.0f}'.format, "Members, today's policy": whole,
                               "Revenue per customer, today's policy": money}, na_rep='')


def plot(ctx, mine):
    """Members at the end of each season, and revenue per customer, for the chosen settings against today's policy."""
    mine, today = mine.iloc[:HORIZON], ctx['today'].iloc[:HORIZON]
    short = lambda k: u.period_label(k)
    names = [short(k) for k in [ctx['last'], *ctx['future']]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))

    mine_m = np.r_[ctx['start_members'], mine['members']] / 1000
    today_m = np.r_[ctx['start_members'], today['members']] / 1000
    left.plot(names, today_m, color=BRASS, lw=2.5, ls=(0, (4, 3)), label="Today's policy")
    left.plot(names, mine_m, color=HARBOR, lw=2.5, marker='o', ms=7, mec='white', mew=1.5, label='Your settings')
    left.annotate(f'{mine_m[-1]:,.0f}k', (4, mine_m[-1]), xytext=(9, 0), textcoords='offset points', va='center',
                  color=INK, weight='bold')
    if abs(mine_m[-1] - today_m[-1]) > 0.04 * max(mine_m.max(), today_m.max()):
        left.annotate(f'{today_m[-1]:,.0f}k', (4, today_m[-1]), xytext=(9, 0), textcoords='offset points', va='center',
                      color=GREY)
    left.set_xlim(-0.3, 4.6)
    left.set_ylabel('Members (thousands)')
    left.set_title('Members at the end of each season', loc='left', color=INK, weight='bold')
    left.set_xticks(range(5), [names[0] + '\n(actual)', *names[1:]])

    x, w = np.arange(HORIZON), 0.36
    bars = right.bar(x - w / 2 - 0.01, mine['revenue per customer'], w, color=HARBOR, label='Your settings')
    right.bar(x + w / 2 + 0.01, today['revenue per customer'], w, color=BRASS, label="Today's policy")
    right.bar_label(bars, fmt='$%.2f', padding=3, color=INK, weight='bold', fontsize=9)
    right.set_xticks(x, names[1:])
    right.set_ylabel('$ per customer per season')
    right.set_title('Revenue per customer, members and non-members together', loc='left', color=INK, weight='bold')

    for ax in (left, right):
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#d9e0e8', lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, -0.2), ncols=2)
    fig.tight_layout()
    plt.show()


def backtest(ctx):
    """The back-test table, with a two-line summary."""
    record, gap = _backtest_record(ctx)
    member_gap = 100 * (record['Members, forecast'] - record['Members, actual']) / ctx['n']
    print(f"Four-season revenue per customer: ${record['Revenue per customer, forecast'].sum():.2f} forecast against "
          f"${record['Revenue per customer, actual'].sum():.2f} actual ({100 * gap:+.0f}%).")
    print(f"Members: off by {member_gap.min():+.1f} to {member_gap.max():+.1f} per 100 customers.")
    money, whole = '${:,.2f}'.format, '{:,.0f}'.format
    return record.style.format({'Members, actual': whole, 'Members, forecast': whole,
                                'Revenue per customer, actual': money, 'Revenue per customer, forecast': money})


def _backtest_record(ctx):
    """The same method started at the end of fall 2018 and asked to forecast 2019 under 2019's actual policy.

    It is given the 2019 prices (today's random discounts, with the $15 base until the cut). Advertising is taken
    from the same season a year earlier, exactly as the forecast takes 2019's. Nothing else from 2019 is used.
    Returns the season table and the four-season revenue gap (forecast / actual - 1).
    """
    seasons, n = ctx['seasons'], ctx['n']
    start = ctx['last'] - 4
    future = [start + 1 + i for i in range(HORIZON)]
    matched = [k - 4 for k in future]
    bt = forecast(ctx, prepare(ctx, start, matched), [today_policy(15 if k < u.CUT_T else 12) for k in future])
    record = pd.DataFrame({'Season': [u.period_label(k) for k in future],
                           'Ad exposure taken from': [u.period_label(k) for k in matched],
                           'Members, actual': [seasons[k]['sign_up_flag'].sum() for k in future],
                           'Members, forecast': bt['members'].round(),
                           'Revenue per customer, actual': [(seasons[k]['membership_price'] * seasons[k]['sign_up_flag']).sum() / n
                                                            for k in future],
                           'Revenue per customer, forecast': bt['revenue per customer']}).set_index('Season')
    gap = record['Revenue per customer, forecast'].sum() / record['Revenue per customer, actual'].sum() - 1
    return record, gap
