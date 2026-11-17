# Pricing forecast tool

Forecasts revenue per customer and member count for the next four seasons under a pricing setting, and compares
it with today's policy (a random $1–$5 off, re-drawn every season). It uses the joining model from Part 3 and the
cancellation model from Part 4. Nothing is fitted in the tool.

## Running it

Set `BASE_PRICE`, `JOINING_DISCOUNT`, `SHARE_OFFERED`, `MEMBER_DISCOUNT` and `AD_BACKDROP` in the first code cell
of `pricing_forecast_tool.ipynb` and run all (about a minute). Paths are relative, so always run from the repo root.

## Files

- `pricing_forecast_tool.ipynb`: inputs, explanations and one-line calls. Keep logic out of it.
- `helper/tools_utils.py`: all the tool code.
- `helper/utils.py`: shared with the analysis notebooks. Builds the exposure terms (`add_terms_join`, `add_terms_cancel`).
- `outputs/join_model_params.csv` and `outputs/cancel_model_params.csv`: the coefficients, written by
  `analysis/03_join_modeling.ipynb` and `analysis/04_cancel_modeling.ipynb`.
- `data/customer_panel.csv` (customer-season panel) and `data/aggregate.csv` (planned ad spend).

## The idea in one equation

Both models are logits, so each customer's chance of joining or cancelling in a season is `expit(log-odds)`,
and the log-odds is a sum. The tool splits that sum into the part that depends on the pricing choices and the part
that doesn't:

```
join log-odds   = [intercept + region + this season's exposure]                      ← prepare()
               + [price effect of the price offered + offer term]                    ← price_effects(), used in forecast()

cancel log-odds = [intercept + region + this and last season's exposure]             ← prepare()
                + [tenure effect + last price effect + this season's price effect]   ← price_effects(), used in forecast()
```

The first bracket is the same whatever price is chosen, so `prepare()` computes it once. The second bracket depends
on the setting and on each member's history, so `forecast()` adds it season by season. Four pieces make this work:

| Piece | Job | Called |
|---|---|---|
| `policy()` / `today_policy()` | describe a pricing setting | per setting |
| `price_effects()` | the price, tenure and offer parts of both models | once, in `load()` |
| `prepare()` | the per-customer part that doesn't depend on price | once, in `load()` |
| `forecast()` | roll every customer forward season by season | once per setting |

`load()` wires these together and returns `ctx`, a dictionary holding everything loaded once (data, coefficients,
price lookups, the `prepare()` output as `ctx['inputs']`, and today's policy already forecast as `ctx['today']`).
Everything else in `tools_utils.py` is reporting: `ad_menu`, `run`, `season_table`, `plot` and `best_settings`.

## Pricing settings: `policy()` and `today_policy()`

A setting is a plain dictionary. It is the only thing `forecast()` needs to know about prices:

```python
policy(12, 5, 0.25, 1)   # -> {'base': 12, 'offer_rate': 0.25, 'join_depths': (5,), 'member_depths': (1,)}
today_policy()           # -> {'base': 12, 'offer_rate': 0.25, 'join_depths': (1,2,3,4,5), 'member_depths': (1,2,3,4,5)}
```

- `base`: the list price.
- `offer_rate`: the share of non-members offered a joining discount.
- `join_depths`, `member_depths`: the discounts in use, each **equally likely**. One value is a uniform setting;
  five values is today's random $1–$5.

`forecast()` takes **one setting per season** (a list), so a setting that changes over time needs no change to
`forecast()`: pass a different dictionary for each season.

## `price_effects(jp, cp)`

The price-dependent parts of both models, looked up once for every price the tool can charge (`PRICES`, $7–$15):

| Key | What | Formula |
|---|---|---|
| `join[p]` | joining, price offered | `price_j·(p−12) + price_sq·(p−12)²` |
| `cancel[p]` | cancelling, this season's price | `price_c·(p−12) + price_sq·(p−12)²` |
| `lag[p]` | cancelling, last season's price | `lag_c·(p−12)` |
| `tenure[g−1]` | cancelling, tenure group g (1 … 8+) | `C(tenure_group)[T.g]`, 0 for group 1 |
| `offer` | joining, being offered at all | `offer` coefficient |

Coefficients are read **by name**, so their order in the CSV doesn't matter, and a renamed term raises a
`KeyError`. **The $12 centring is written out here and also in `utils.py`.** If the models are refitted with a
different centring, change both, or the tool keeps running with wrong numbers and no error (this happened once,
when the models moved from $10 to $12).

## `prepare(ctx, ad_seasons)`: everything that doesn't depend on price

**In:** `ad_seasons`, the past season whose ad exposure stands in for each forecast season, e.g.
`[8, 9, 10, 11, 8, 9, 10, 11]` for `'like 2019'` over eight seasons (season numbers: 0 = winter 2017 … 11 = fall 2019).

**What it does**, per forecast season:

1. Copies each customer's own exposure (`pod`, `soc`) from the backdrop season. This relies on every season's table
   having the same customers in the same order, which `load_seasons()` guarantees by sorting on `customer_id`.
2. Pairs it with last season's exposure: the real fall 2019 exposure for the first forecast season, then the
   previous forecast season's (`last = this`).
3. Builds the exposure terms with `add_terms_join` / `add_terms_cancel` and scores **only** the exposure terms
   (`JOIN_EXPOSURE`, `CANCEL_EXPOSURE`), plus intercept and region.
   The `price: 12, price_lag: 12` columns are placeholders: `add_terms_*` needs them to run, but the price terms
   are not scored here. At 12 the centred price terms are zero anyway.

**Out:** a dictionary.

| Key | Shape | What |
|---|---|---|
| `join_index` | list of 8 arrays, one value per customer | joining log-odds before price |
| `cancel_index` | list of 8 arrays, one value per customer | cancelling log-odds before price, tenure and last price |
| `member` | one value per customer | member in fall 2019 (starting state) |
| `tenure` | one value per customer | consecutive seasons as a member, capped at 8 (the model's 8+ group) |
| `last_price` | one value per customer | price paid in fall 2019 |

**Extend it here** when a new model variable depends on the customer but not on the price (for example age or
device): add the column in `load_seasons()`, build its term in `utils.add_terms_*`, and add the term name to
`JOIN_EXPOSURE` / `CANCEL_EXPOSURE` (or score it alongside them). Nothing in `forecast()` changes.

## `forecast(ctx, inp, policies)`: the season-by-season roll

**State.** Each customer is described by probabilities that always add up to 1:

- `non_member[i]`: the chance customer *i* is not a member;
- `members[(tenure, last_price)][i]`: the chance they are a member with that tenure who paid that price last season.

Tenure and last price are tracked because the cancellation model uses both, and under today's random policy
members differ in their last price. `members` is a `defaultdict`, so a state only exists once someone can be in it.
In fall 2019 every customer is certain: members put 1.0 in their own state, non-members have `non_member = 1`.

**Each season**, with that season's setting:

1. **Members stay or cancel.** For every state and every member discount in the setting (each with weight
   `1 / len(member_depths)`):
   `log_odds = cancel_index[season] + tenure[tenure−1] + lag[last_price] + cancel[current_price]`.
   Stayers move to `(min(tenure+1, 8), current_price)` and pay `current_price`; cancellers pay nothing.
2. **Non-members join.** `offers` lists each group: `offer_rate` split evenly across the joining discounts, and
   the rest at the base price with `offered = 0`. For each:
   `log_odds = join_index[season] + join[current_price] + offer·offered`, and
   `new_joiners = non_member × share × p_join`. Joiners start at `(1, current_price)` and pay `current_price`.
   Multiplying by `non_member` is what keeps current members from "joining".
3. **Update.** `non_member = non_member − joined + cancelled`, so people who cancel can rejoin from the next
   season, through the same joining model. `members = next_members`.
4. **Record** expected members, joins, cancels and revenue, summed over customers.

**Out:** one row per season with `members`, `joined`, `cancelled`, `revenue` and `revenue per customer`.
`summarize()` turns it into the four-season totals the reports use.

**Extend it here** when something changes as the forecast runs, like tenure does: add it to the state key
(`(tenure, last_price, new_thing)`), say how it moves each season in steps 1–2, and add its effect to the log-odds.
This is the most work, and each new dimension multiplies the number of states.

## Common changes, and where they go

| Change | Where |
|---|---|
| New season of data | Append to the two data files, rerun 03 and 04 (check their calibration and rolling tests), then run the tool. The start season and the backdrop years move forward on their own. Update the hard-coded counts in the notebook's assumptions cell. |
| Today's policy changes | `TODAY_OFFER_RATE` and `TODAY_DEPTHS` |
| A setting that changes by season | Pass a different `policy()` per season to `forecast()`; add the inputs in `run()` |
| Unequal discount weights | Replace the equal weights in `forecast()` (`discount_weight`, and `share` in `offers`) with weights from the setting |
| Targeted offers by customer | Make `share` in `offers` a per-customer array. Offers were only ever given by a business rule, so add a caution |
| New price-only model term | `price_effects()`, then use it in `forecast()` |
| New customer-level model term | `load_seasons()`, `utils.add_terms_*`, and the term lists `prepare()` scores |
| New term that changes over time | The state key in `forecast()` |
| New output | A function in `tools_utils.py`, called from a one-line notebook cell |
| Ad-spend scenarios | Not possible with these models: exposure wasn't randomized |

## Checks after a change

1. The notebook runs top to bottom.
2. First-season join and cancel probabilities match the models scored directly with `utils.add_terms_*`.
3. With a single discount for joiners and members, results match the previous version exactly.
4. The headline for $5 to join and $1 for members stays at $15.75 per customer (+13.7%) with the 2019 backdrop.

These would make good pytest tests.

## Known limitations

- No new customers arrive, and Panem is excluded because its price never varied.
- Customers are lined up by position across seasons. This works because every customer appears in every season;
  a data drop with missing customer-seasons would break it silently.
