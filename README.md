# Bazar Sanket (বজাৰ সংকেত) — price intelligence and forecasting for Assam farmers

Prototype submitted to **NESFIC 2026, challenge NESFIC-D-35** ("Agricultural Commodity Price Intelligence and Forecasting for Farmer Decisions", Agriculture Department, Government of Assam).

Applicant: Arijit Gogoi (individual applicant).

The prototype uses only public data: Agmarknet daily mandi prices published on data.gov.in. It:

1. measures how incomplete the public Assam price record is (gaps, renamed markets, patchy reporting);
2. selects pilot series with a fixed coverage rule;
3. trains one **global quantile model** across all Assam markets of the 11 crops with good data, so that markets with short or gappy histories borrow patterns from other markets and crops (this configuration was chosen after the 12-month test; the one set before testing, the three pilot crops with neighbouring-State markets, is still run and reported beside it);
4. evaluates it with a 12-month **rolling-origin backtest** against no-change and seasonal-naive benchmarks;
5. **stress-tests** forecasting when local history is short, gappy or absent;
6. converts forecast bands into **Hold / Sell / Store** advice and backtests that rule;
7. estimates **season-ahead** price profiles and checks them out of sample;
8. tries **other methods** on the same forecasts: Holt-Winters exponential smoothing (statsmodels; additive on the log price, plus the textbook multiplicative form as a check), an average of the global model and Holt-Winters, **LightGBM** as the learner of the global model, and **volatility pricing** of downside risk (the expected shortfall below a price floor);
9. repeats the accuracy test on **every Assam commodity with good data** (a coverage rule fixed in advance: at least 1,000 market-weeks with a price in the last 156 weeks and 100 in the test year);
10. renders the proposal answers and the Proof of Work from the generated metrics, so no number is typed by hand.

The rendered report is in [`docs/index.html`](docs/index.html) (also published with GitHub Pages) and [`proof-of-work.pdf`](proof-of-work.pdf). The proposal text is in [`answers.md`](answers.md). Every commodity in the downloaded data, and how each is used, is listed in [`COMMODITIES.md`](COMMODITIES.md).

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/) and Google Chrome (for the PDF only).

```sh
uv sync
# optional: refresh the data (needs a free data.gov.in API key in .env as DATA_GOV_IN_API_KEY=...)
uv run python -I src/fetch.py pull --state Assam --since 2006-01-01 --limit 5000 --max-offset 600000
uv run python -I src/fetch.py consolidate
# analysis (marimo notebooks, runnable as scripts or with `uv run marimo edit nb/forecast.py`)
uv run python -I -X utf8 nb/profile.py
uv run python -I -X utf8 nb/forecast.py      # uses all CPU cores
uv run python -I -X utf8 nb/benchmarks.py    # Holt-Winters and volatility pricing, all CPU cores
uv run python -I -X utf8 nb/extended.py      # LightGBM, multiplicative Holt-Winters, more commodities
uv run python -I -X utf8 src/commodities.py  # COMMODITIES.md
# proposal answers, report and PDF
uv run python -I -X utf8 src/render.py --pdf
# or everything after the optional refresh, in one command
uv run python -I src/pipeline.py
```

The data snapshot used for the submission is `data/raw/ogd_prices.parquet`.

## Layout

| Path | Contents |
|---|---|
| `src/fetch.py` | data.gov.in OGD API client; caches every page with its query and retrieval time |
| `src/apa/clean.py` | market-name harmonisation, weekly aggregation, outlier flags, coverage, selection rule, good-data commodity rule |
| `src/apa/features.py` | past-only features, regional indices, targets |
| `src/apa/model.py` | quantile gradient boosting (scikit-learn; LightGBM as an option), parallel task runner, rolling-origin backtest, scores |
| `src/apa/advisory.py` | Hold / Sell / Store decision rule (illustrative parameters) |
| `src/apa/seasonal.py` | season-ahead price profiles and their out-of-sample check |
| `src/apa/smoothing.py` | Holt-Winters benchmark (statsmodels; log-additive or textbook multiplicative), re-estimated at every forecast origin, in parallel |
| `src/apa/volatility.py` | EWMA volatility, volatility bands, pricing of downside risk (Black-76 put value) |
| `src/apa/compare.py` | scores for point forecasts and 80% ranges on identical rows |
| `nb/profile.py`, `nb/forecast.py`, `nb/benchmarks.py`, `nb/extended.py` | marimo notebooks that run the analysis and write `out/metrics_*.json` and figures |
| `src/commodities.py` | writes `COMMODITIES.md` from the data snapshot and the generated metrics |
| `src/render.py`, `src/templates/`, `src/apa/report*.py` | proposal answers and report rendered from `out/metrics.json` |
| `src/pipeline.py` | runs every step above in order |

## Data and licence

Data: Agmarknet daily market prices, Directorate of Marketing & Inspection, Ministry of Agriculture & Farmers Welfare, via the data.gov.in Open Government Data API (resource `35985678-0d79-46b4-9ed6-6f13308a1d24`), used under the Government Open Data License – India. No departmental (AMIU) data are included.

Code: no licence is granted at this time. Licence terms will be agreed with the department.

## Status and limitations

This is a prototype. Decision-rule parameters are illustrative and must be set with Assam Agricultural University before use. Assamese text is a draft translation pending native-speaker review. The public dataset has no arrivals field. See Section J of the report.
