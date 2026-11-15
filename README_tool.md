# Pricing forecast tool

Forecasts revenue per customer and member count for the next four seasons under a pricing setting, and compares
it with today's policy (a random $1–$5 off, re-drawn every season). It uses the joining model from Part 3 and the
cancellation model from Part 4. Nothing is fitted in the tool.

## Running it

From the repo root:

Set `BASE_PRICE`, `JOINING_DISCOUNT`, `SHARE_OFFERED` and `MEMBER_DISCOUNT` in the first code cell and run all
(about a minute). `AD_PLAN` is optional; leave it as `None` to repeat 2019's advertising. Paths are relative, so
always run from the repo root.

## Files

- `pricing_forecast_tool.ipynb`: inputs, explanations and one-line calls. Keep logic out of it.
- `helper/tools_utils.py`: all the tool code.
- `helper/utils.py`: shared with the analysis notebooks. Builds the model terms.
- `analysis/03_join_modeling.ipynb` writes `outputs/join_model_params.csv`.
- `analysis/04_cancel_modeling.ipynb` writes `outputs/cancel_model_params.csv`.
- `data/customer_panel.csv` (customer-season panel) and `data/aggregate.csv` (planned ad spend).

## How it works

Each customer carries the probability of being a non-member, or a member with a given tenure and last price. Starting from real membership in fall 2019, each season:

- members stay or cancel (cancellation model: price, last price, tenure, exposure, region);
- non-members join, with or without an offer (joining model);
- people who cancel can rejoin later.

Ad exposure for each future season is the customer's own exposure in the same season of 2019. So the tool assumes marketing, and how spend turns into exposure, stay as they were; it can't value a change in ad spend. `AD_PLAN` only swaps in a past season with similar planned spend.

`load()` does the setup and returns `ctx`; `forecast()` does the season-by-season roll. The rest of
`tools_utils.py` is reporting: `run`, `season_table`, `plot`, `best_settings`, `robustness` and `backtest`.

## Model terms

The tool builds its terms with `add_terms_join` and `add_terms_cancel` from `helper/utils.py`, the same
functions the models were fitted with. Don't re-type a formula in the tool. An earlier version hard-coded the price
centering, and when the models moved from $10 to $12 the tool kept running with wrong numbers and no error.

If you change a term, change it in `utils.py`, refit 03 and 04, and the tool picks it up. A missing term name
raises a `KeyError`; fix the name rather than adding a default.

## Maintaining it

New season of data: append to the two data files, rerun 03 and 04 (check their calibration and rolling tests), then run the tool. The start season, the "same season last year" advertising and the back-test all move forward on their own. Update the hard-coded counts in the assumptions cell and the back-test numbers in the Reading.

Today's policy changes: edit `TODAY_OFFER_RATE` and `TODAY_DEPTHS` in `tools_utils.py`.

New model variable:
- depends only on price: add it to `Models.join_price`, `cancel_price` or `cancel_lag`;
- depends on the customer but not on price: add it in `load_seasons` and `prepare`;
- changes as the forecast runs (like tenure): add it to the state in `forecast`. This is the most work.

## Extending it

- Different settings by season: `forecast` already takes one policy per season; it only needs inputs in `run`.
- Unequal discount weights or targeted offers: change the weights in `forecast`. Offers were only ever given by
  a business rule, so add a caution.
- New output: write a function in `tools_utils.py` and call it from a one-line cell.
- Ad-spend scenarios: not possible with these models. Exposure wasn't randomized.

## Checks after a change

1. The notebook runs top to bottom.
2. First-season join and cancel probabilities match the models scored directly with `utils.add_terms_*`.
3. With a single discount for joiners and members, results match the previous version exactly.
4. The back-test stays close to its current value (12% low on revenue). A big jump usually means a bug.

These would make good pytest tests.

## Known limitations

- Levels run about 12% low in the back-test. Differences between settings are more reliable.
- Deep joining discounts look slightly too good. Members who joined at $7–$8 cancel about 4 points more at their
  first renewal than predicted. Even in the worst case, the top setting still beats today's policy (+11.7%
  instead of +13.7%).
- No new customers arrive, and Panem is excluded because its price never varied.
