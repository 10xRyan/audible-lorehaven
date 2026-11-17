## Overall Usage

I first used AI to agentically reviewed my EDA and tried to used to code the price cut analysis by itself. It tended to provide dense information even though I only asked it to strictly focus on the detailed plan through the prompt. It later took me lots of time to trim and understand what it tried to do. So for the following parts, I mainly used it to implement every functionality seperatly instead of code the whole step from scratch. I used AI more or less among most of the files except this one. I also tried not to rely on AI for the writeup but used it to validated the numbers I bring up.

## Example of AI Prompt

1. Run a DiD event study using Winter 2019 as the reference. Use historical-gap bands and add model-based confidence interval as diagnostic.
2. Run placebo tests using only pre-cut data. Each fake cut should have at least 4 pre-cut and 4 post-cut seasons.
3. Run a standard DiD to estimate the average treatment effect per customer per season, using all pre- and post-cut periods.
4. Include region and time fixed effects, and weight by customer count. I also want to add a Westeros winter adjustment because it shows 0 impression but test the results without it too.
5. Make the code modular, write the helper functions into the `utils.py`, so I can easily run everything on three metrics:
   - Revenue per customer
   - Log revenue per customer (log after aggregation)
   - Membership percentage

Have some plots functions to show band and gap cleanly. Also mark the price cut time on the plots.

## Errors:
There are some errors that AI made along the way and I found them through both reviewing the code and spotting anomolies from the result. The errors are listed below:
* When calculating V4, AI treats every non-joiner as $0 for the whole observation window which understates the effect by ~$2.
* Overstating the effect from visualization, offered V4 vs. non-offered.
* Taking log points as percentage when the numbers are not small enough to be directly used.
