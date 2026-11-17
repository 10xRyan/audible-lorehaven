"""Code behind pricing_forecast_tool.ipynb: load the data and the Part 3 / Part 4 models, forecast, report.

The exposure terms are built by utils.add_terms_join and utils.add_terms_cancel, the same functions the models were
fitted with. The price terms are written out in `price_effects`: prices are centred at the $12 base, as in Parts 3
and 4, so if that centring ever changes in utils.py it must change there too.
"""
import numbers
from collections import defaultdict

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
HEAVY = 80                                      # listens x impressions above this = heavy exposure (Part 3)
AD_BACKDROPS = ('like 2017', 'like 2018', 'like 2019', 'lightest', 'heaviest')
HARBOR, BRASS, INK, GREY = '#0086a8', '#b8862b', '#14233b', '#7b8798'
JOIN_EXPOSURE = ['pod', 'soc', 'sqrt_pod_soc', 'pod_soc', 'pod_zero', 'soc_zero']
CANCEL_EXPOSURE = JOIN_EXPOSURE + [name + '_last' for name in JOIN_EXPOSURE]


# ---------------------------------------------------------------- pricing policies
def policy(base, join_discount, offer_rate, member_discount):
    """One uniform setting: every offered prospect gets `join_discount` off, every member `member_discount` off."""
    return dict(base=base, offer_rate=offer_rate, join_depths=(join_discount,), member_depths=(member_discount,))


def today_policy(base=12):
    """Today's policy: a quarter of prospects offered, and a fresh random $1-$5 off every season for offers and members."""
    return dict(base=base, offer_rate=TODAY_OFFER_RATE, join_depths=TODAY_DEPTHS, member_depths=TODAY_DEPTHS)


# ---------------------------------------------------------------- data and models
def load_seasons(path=PANEL):
    """One frame per season for the five cut regions, every frame in the same customer order, with tenure added."""
    p = u.load_panel(path, extra_cols=('customer_id', 'social_ad_impressions', 'podcast_ad_listen'))
    p = (p[p['region_name'].ne(u.CONTROL)].sort_values(['customer_id', 't'])
         .rename(columns={'podcast_ad_listen': 'pod', 'social_ad_impressions': 'soc'}))
    # Tenure = consecutive seasons as a member. A new spell starts each time someone joins after not being a member.
    prev = p.groupby('customer_id')['sign_up_flag'].shift(1)
    spell = (p['sign_up_flag'].eq(1) & prev.fillna(0).eq(0)).groupby(p['customer_id']).cumsum()
    p['tenure'] = np.where(p['sign_up_flag'].eq(1), p.groupby(['customer_id', spell]).cumcount() + 1, 0)
    return {t: g.set_index('customer_id').sort_index() for t, g in p.groupby('t')}


def score(params, terms, names):
    """Sum of coefficient x term over `names` (one value per row of `terms`)."""
    return sum(params[name] * terms[name].to_numpy(float) for name in names)


def region_effect(params, region):
    """Each customer's region coefficient (0 for the base region)."""
    return np.array([params.get(f'C(region_name)[T.{r}]', 0.0) for r in region])


def price_effects(jp, cp):
    """The price parts of both models (prices centred at the $12 base, as in Parts 3 and 4), for every price."""
    join = {p: jp['price_j'] * (p - 12) + jp['price_sq'] * (p - 12) ** 2 for p in PRICES}        # this season's price
    cancel = {p: cp['price_c'] * (p - 12) + cp['price_sq'] * (p - 12) ** 2 for p in PRICES}      # this season's price
    lag = {p: cp['lag_c'] * (p - 12) for p in PRICES}                                            # last season's price
    tenure = np.array([0.0] + [cp[f'C(tenure_group)[T.{g}]'] for g in range(2, 9)])        # groups 1 ... 8+
    return dict(join=join, cancel=cancel, lag=lag, tenure=tenure, offer=jp['offer'])


def load(ad_backdrop='like 2019'):
    # Everything the notebook needs: data, models, and the forecast inputs for the eight seasons after fall
    seasons = load_seasons()
    spend = pd.read_csv(AGGREGATE)
    spend['t'] = (spend['year'] - u.FIRST_YEAR) * 4 + spend['season'].str.lower().map(u.SEASONS)
    # order doesn't matter, col name saved as index
    jp = pd.read_csv(JOIN_PARAMS, index_col=0)['coefficient']
    cp = pd.read_csv(CANCEL_PARAMS, index_col=0)['coefficient']
    last = max(seasons)                                                     # fall 2019
    s = seasons[last]
    ctx = dict(seasons=seasons, jp=jp, cp=cp, price=price_effects(jp, cp),
               spend=spend.groupby('t')[['podcast_spend', 'social_spend']].first(),
               heavy=pd.Series({t: (x['pod'] * x['soc'] > HEAVY).mean() for t, x in seasons.items()}),
               last=last, future=[last + 1 + i for i in range(HORIZON)], n=len(s),
               start_members=int(s['sign_up_flag'].sum()),
               start_revenue=(s['membership_price'] * s['sign_up_flag']).sum() / len(s))
    ctx['ad_seasons'] = pick_ad_seasons(ctx, ad_backdrop)
    ctx['inputs'] = prepare(ctx, ctx['ad_seasons'] * 2)                     # the same backdrop for the extra four
    ctx['today'] = forecast(ctx, ctx['inputs'], [today_policy()] * (HORIZON + AFTER))
    print(f"Ready: {ctx['n']:,} customers in five regions, {ctx['start_members']:,} of them members in "
          f"{u.period_label(last)}.")
    print(f"Advertising backdrop '{ad_backdrop}': ads copied from "
          + ', '.join(u.period_label(t) for t in ctx['ad_seasons']) + '.')
    return ctx


def pick_ad_seasons(ctx, choice):
    """For each forecast season, the past season at the same time of year whose ad exposure stands in for it.

    'like 2017' / 'like 2018' / 'like 2019': that year, season for season.
    'lightest' / 'heaviest': the past season with the smallest / largest share of heavily exposed customers.
    A backdrop, not a spend lever: exposure was not randomized, so a gap between backdrops is not the return on ads.
    """
    if choice not in AD_BACKDROPS:
        raise ValueError(f"AD_BACKDROP must be one of {', '.join(repr(c) for c in AD_BACKDROPS)}.")
    if choice.startswith('like '):
        year = int(choice[5:]) - u.FIRST_YEAR
        return [4 * year + t % 4 for t in ctx['future']]
    heavy = ctx['heavy']
    same_time = [heavy[heavy.index % 4 == t % 4] for t in ctx['future']]
    return [int(h.idxmin() if choice == 'lightest' else h.idxmax()) for h in same_time]


def ad_menu(ctx):
    """Each past season's advertising, so a backdrop can be chosen. Marks the seasons the current backdrop uses."""
    order = sorted(ctx['heavy'].index, key=lambda t: (t % 4, t))           # by time of year, then year
    heavy = ctx['heavy'][order]
    table = pd.DataFrame({
        'Time of year': [u.period_label(t).split()[0] for t in order],
        'Year': [u.FIRST_YEAR + t // 4 for t in order],
        'Podcast spend (planned)': ctx['spend'].loc[order, 'podcast_spend'].to_numpy(),
        'Social spend (planned)': ctx['spend'].loc[order, 'social_spend'].to_numpy(),
        'Customers heavily exposed': heavy.to_numpy(),
        'Ad level': ['Light' if x < 0.05 else 'Heavy' if x >= 0.15 else 'Typical' for x in heavy],
        'Used now': ['yes' if t in ctx['ad_seasons'] else '' for t in order],
    }).set_index(['Time of year', 'Year'])
    return table.style.format({'Customers heavily exposed': '{:.0%}'.format})


def prepare(ctx, ad_seasons):
    """Everything about each customer that does not depend on the pricing choices.

    ad_seasons: for each forecast season, the past season whose exposure stands in for it.
    Returns, per forecast season, each customer's join and cancel log-odds before the price terms.
    """
    seasons, jp, cp = ctx['seasons'], ctx['jp'], ctx['cp']
    s0 = seasons[ctx['last']]                                                # fall 2019, the last real season
    region = s0['region_name'].to_numpy()
    join_index, cancel_index = [], []
    last = s0[['pod', 'soc']].to_numpy(float)                                # last observed season: real exposure
    for t in ad_seasons:
        this = seasons[t][['pod', 'soc']].to_numpy(float)
        x = pd.DataFrame({'pod': this[:, 0], 'soc': this[:, 1], 'pod_last': last[:, 0], 'soc_last': last[:, 1],
                          'price': 12, 'price_lag': 12})
        join_index.append(jp['Intercept'] + region_effect(jp, region) + score(jp, u.add_terms_join(x), JOIN_EXPOSURE))
        cancel_index.append(cp['Intercept'] + region_effect(cp, region)
                            + score(cp, u.add_terms_cancel(x), CANCEL_EXPOSURE))
        last = this                                                          # the next season looks back at this one
    return dict(join_index=join_index, cancel_index=cancel_index, member=s0['sign_up_flag'].to_numpy().astype(bool),
                tenure=np.clip(s0['tenure'].to_numpy(), 1, 8), last_price=s0['membership_price'].to_numpy().astype(int))


# ---------------------------------------------------------------- forecast
def forecast(ctx, inp, policies):
    """Roll forward one season at a time; one setting per season (see `policy` and `today_policy`).

    Each customer carries the probability of being a non-member, or a member with a given tenure (1 ... 8+ seasons)
    and a given last price. The last price matters because the cancellation model uses it, and under today's policy it
    differs from member to member. Each season starts from the season before.
    """
    n = ctx['n']
    price = ctx['price']

    # Initial non-member probabilities
    non_member = (~inp['member']).astype(float)

    # Initial member states: (tenure, last_price) -> each customer's probability of being in that state.
    # defaultdict creates an empty array the first time a new state is used.
    members = defaultdict(lambda: np.zeros(n))
    for i in np.where(inp['member'])[0]:
        tenure = int(inp['tenure'][i])
        last_price = int(inp['last_price'][i])
        members[(tenure, last_price)][i] = 1.0

    results = []
    for season, setting in enumerate(policies):
        # initialization for each simulated season
        next_members = defaultdict(lambda: np.zeros(n))
        joined = np.zeros(n)
        cancelled = np.zeros(n)
        stayed = np.zeros(n)
        revenue = np.zeros(n)
        discount_weight = 1 / len(setting['member_depths'])            # each member discount equally likely

        # STEP 1: Existing members stay or cancel
        for (tenure, last_price), prob in members.items():
            for discount in setting['member_depths']:
                current_price = setting['base'] - discount
                log_odds = (
                    inp['cancel_index'][season]
                    + price['tenure'][tenure - 1]
                    + price['lag'][last_price]
                    + price['cancel'][current_price]
                )
                p_cancel = expit(log_odds)

                p_stay = prob * discount_weight * (1 - p_cancel)
                p_churn = prob * discount_weight * p_cancel

                next_tenure = min(tenure + 1, 8)                        # 8 = 8 or more seasons
                next_members[(next_tenure, current_price)] += p_stay

                stayed += p_stay
                cancelled += p_churn
                revenue += p_stay * current_price

        # STEP 2: Non-members join, with an offer or at the base price
        offers = [
            (setting['offer_rate'] / len(setting['join_depths']), setting['base'] - d, 1)
            for d in setting['join_depths']
        ]
        offers.append((1 - setting['offer_rate'], setting['base'], 0))

        for share, current_price, offered in offers:
            log_odds = (
                inp['join_index'][season]
                + price['join'][current_price]
                + price['offer'] * offered
            )
            p_join = expit(log_odds)
            new_joiners = non_member * share * p_join

            next_members[(1, current_price)] += new_joiners

            joined += new_joiners
            revenue += new_joiners * current_price

        # STEP 3: Update customer states (cancellers can rejoin from next season)
        non_member = non_member - joined + cancelled
        members = next_members

        # STEP 4: Save results
        results.append({
            'members': (stayed + joined).sum(),
            'joined': joined.sum(),
            'cancelled': cancelled.sum(),
            'revenue': revenue.sum(),
            'revenue per customer': revenue.sum() / n,
        })

    return pd.DataFrame(results)


def summarize(run):
    """Revenue per customer over the window and over the four seasons after it, and members at the end of the window."""
    return {'revenue': run['revenue per customer'].iloc[:HORIZON].sum(), 'members': run['members'].iloc[HORIZON - 1],
            'revenue_after': run['revenue per customer'].iloc[HORIZON:].sum()}


# ---------------------------------------------------------------- the notebook's outputs
def run(ctx, base, join_discount, share_offered, member_discount):
    """Check the choices, forecast them, and print the headline against today's policy."""
    # Types first: the range checks below can't compare text or None with numbers, and prices are looked up by
    # whole dollar, so $2.50 off has no entry.
    is_number = lambda x: isinstance(x, numbers.Real) and not isinstance(x, bool)
    problems = [f'{name} must be a whole number of dollars (got {value!r}).'
                for name, value in [('BASE_PRICE', base), ('JOINING_DISCOUNT', join_discount),
                                    ('MEMBER_DISCOUNT', member_discount)]
                if not (is_number(value) and float(value).is_integer())]
    if not is_number(share_offered): problems.append(f'SHARE_OFFERED must be a number from 0 to 100 (got {share_offered!r}).')
    if problems:
        raise ValueError('Please change the pricing choices at the top:\n  ' + '\n  '.join(problems))
    base, join_discount, member_discount = int(base), int(join_discount), int(member_discount)

    join_price, member_price = base - join_discount, base - member_discount
    if not 10 <= base <= 15: problems.append('BASE_PRICE must be between 10 and 15.')
    if not 0 <= join_discount <= 5: problems.append('JOINING_DISCOUNT must be between 0 and 5.')
    if not 0 <= member_discount <= 5: problems.append('MEMBER_DISCOUNT must be between 0 and 5.')
    if not 0 <= share_offered <= 100: problems.append('SHARE_OFFERED must be between 0 and 100.')
    if min(join_price, member_price) < 7: problems.append('No price can go below $7, the lowest price ever charged. Use a smaller discount.')
    if problems:
        raise ValueError('Please change the pricing choices at the top:\n  ' + '\n  '.join(problems))

    mine = forecast(ctx, ctx['inputs'], [policy(base, join_discount, share_offered / 100, member_discount)] * (HORIZON + AFTER))
    window, today = summarize(mine), summarize(ctx['today'])
    more = lambda x: 'more' if x > 0 else 'less'
    pct = lambda a, b: 100 * (a / b - 1)
    gap, member_gap = window['revenue'] - today['revenue'], window['members'] - today['members']
    later_gap = window['revenue_after'] - today['revenue_after']
    print(f"Your choices: ${base} base price; ${join_discount} off the first season for {share_offered}% of non-members "
          f"(they pay ${join_price}); ${member_discount} off for members (they pay ${member_price}).\n")
    print(f"Over the next four seasons each customer brings in ${window['revenue']:.2f},")
    print(f"and {window['members']:,.0f} are members by {u.period_label(ctx['future'][-1])}.\n")
    print(f"Compared with today's policy (a random $1-$5 off, re-drawn every season): ${abs(gap):.2f} "
          f"{more(gap)} per customer ({pct(window['revenue'], today['revenue']):+.1f}%), "
          f"and {abs(member_gap):,.0f} {'more' if member_gap > 0 else 'fewer'} members.")
    print(f"In total: ${window['revenue'] * ctx['n']:,.0f} against ${today['revenue'] * ctx['n']:,.0f}.\n")
    print(f"The four seasons after that, if nothing changes: ${abs(later_gap):.2f} {more(later_gap)} per "
          f"customer than today's policy ({pct(window['revenue_after'], today['revenue_after']):+.1f}%). This shows whether "
          f"the lead holds once the member base has adjusted.")

    notes = []
    if member_price >= 15: notes.append('No member has ever been charged $15, so the member side goes one dollar beyond the data.')
    elif member_discount == 0: notes.append('Every member has always had a discount. Charging members the full base price assumes they respond to the price alone.')
    if 0 < share_offered < 1: notes.append(f'SHARE_OFFERED is a percentage, so {share_offered} means {share_offered}% of non-members. For a quarter of them, enter 25.')
    if base < 12: notes.append('The base price has never been below $12. Prices this low were only seen as discounts, so treat this as rougher.')
    if share_offered > 50: notes.append('Offers have only ever gone to about a quarter of non-members. This assumes the rest would respond the same way.')
    if notes:
        print('\nCaution:\n  - ' + '\n  - '.join(notes))
    return mine


def best_settings(ctx, base, share_offered, top=10):
    """Every joining and member discount at the given base price and offer share, ranked by four-season revenue."""
    today = summarize(ctx['today'])
    rows = []
    for j in range(6):
        for m in range(6):
            if min(base - j, base - m) < 7:
                continue
            run = summarize(forecast(ctx, ctx['inputs'], [policy(base, j, share_offered / 100, m)] * (HORIZON + AFTER)))
            rows.append({'Joining discount': j, 'Member discount': m, 'Revenue per customer, 4 seasons': run['revenue'],
                         'vs today': run['revenue'] / today['revenue'] - 1,
                         'Members at the end': run['members'], 'Members vs today': run['members'] - today['members'],
                         'Next 4 seasons, vs today': run['revenue_after'] / today['revenue_after'] - 1})
    table = (pd.DataFrame(rows).sort_values('Revenue per customer, 4 seasons', ascending=False).head(top)
             .reset_index(drop=True))
    table.index = table.index + 1
    print(f"Top {top} of {len(rows)} settings at a ${base} base with {share_offered}% of non-members offered. "
          f"Today's policy: ${today['revenue']:.2f} per customer, {today['members']:,.0f} members at the end.")
    return table.style.format({'Joining discount': '${}'.format, 'Member discount': '${}'.format,
                               'Revenue per customer, 4 seasons': '${:,.2f}'.format, 'vs today': '{:+.1%}'.format,
                               'Members at the end': '{:,.0f}'.format, 'Members vs today': '{:+,.0f}'.format,
                               'Next 4 seasons, vs today': '{:+.1%}'.format})


def season_table(ctx, mine):
    """Season-by-season table for the chosen settings, with today's policy alongside."""
    mine, today = mine.iloc[:HORIZON], ctx['today'].iloc[:HORIZON]
    columns = ['Season', 'Members', 'Joined', 'Cancelled', 'Revenue per customer', 'Total revenue',
               "Members, today's policy", "Revenue per customer, today's policy"]
    rows = [[u.period_label(ctx['last']) + ' (actual)', ctx['start_members'], None, None, ctx['start_revenue'],
             ctx['start_revenue'] * ctx['n'], ctx['start_members'], ctx['start_revenue']]]
    for t, (_, m), (_, d) in zip(ctx['future'], mine.iterrows(), today.iterrows()):
        rows.append([u.period_label(t), m['members'], m['joined'], m['cancelled'], m['revenue per customer'],
                     m['revenue'], d['members'], d['revenue per customer']])
    rows.append(['Four seasons', None, mine['joined'].sum(), mine['cancelled'].sum(), mine['revenue per customer'].sum(),
                 mine['revenue'].sum(), None, today['revenue per customer'].sum()])
    table = pd.DataFrame(rows, columns=columns).set_index('Season')
    money, whole = '${:,.2f}'.format, '{:,.0f}'.format
    return table.style.format({'Members': whole, 'Joined': whole, 'Cancelled': whole, 'Revenue per customer': money,
                               'Total revenue': '${:,.0f}'.format, "Members, today's policy": whole,
                               "Revenue per customer, today's policy": money}, na_rep='')


def plot(ctx, mine):
    """Members at the end of each season, and revenue per customer, for the chosen settings against today's policy."""
    mine, today = mine.iloc[:HORIZON], ctx['today'].iloc[:HORIZON]
    names = [u.period_label(t) for t in [ctx['last'], *ctx['future']]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))

    mine_m = np.r_[ctx['start_members'], mine['members']] / 1000
    today_m = np.r_[ctx['start_members'], today['members']] / 1000
    left.plot(names, today_m, color=BRASS, lw=2.5, ls=(0, (4, 3)), label="Today's policy")
    left.plot(names, mine_m, color=HARBOR, lw=2.5, marker='o', ms=7, mec='white', mew=1.5, label='Your settings')
    left.annotate(f'{mine_m[-1]:,.0f}k', (4, mine_m[-1]), xytext=(9, 0), textcoords='offset points', va='center',
                  color=INK, weight='bold')
    if abs(mine_m[-1] - today_m[-1]) > 0.04 * max(mine_m.max(), today_m.max()):       # label today's only if apart
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
