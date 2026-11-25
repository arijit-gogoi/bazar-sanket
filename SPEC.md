# SPEC

## §G GOAL
ship NESFIC-D-35 proposal by 2026-11-25 cutoff: 12 form answers (∀ ≤500 w) + proof-of-work PDF (≤10 MB) backed by real-data prototype of two-horizon Assam crop price advisory (short: 1–4 wk price band + Hold/Sell/Store ∀ commodity–market; long: season-ahead crop-planning outlook) + public GitHub repo & Pages site link.

## §C CONSTRAINTS
- deadline: submit 2026-11-25. spec written 17:19 IST. portal cutoff time unknown (user) ∴ assume early.
- order by risk: submittable pkg (answers + PDF w/ mock-ups & method) ! exist early; polish after.
- draft result-free answers (1,2,4,7,8,10,11) early, while fetch runs; result sections (3,9,12) after backtest.
- brief = `nesfic-d-35.md`. answers ! cover ∀ "should enable" bullet: incomplete-data handling, spot price + arrivals use, short vs long horizon split, farmer-usable output.
- applicant: individual, Arijit Gogoi (self). software developer, new to agri. ∀ other role "to be engaged". ⊥ invent credentials, degrees, past projects.
- data: public sources only (Agmarknet, data.gov.in, cited mirrors). ⊥ claim AMIU, AAU, ARIAS, NCDEX access | partnership → "to be requested".
- data probe time-box 45 min. no usable history → fallback: mock-ups + method demo on synthetic data, labelled; ⊥ accuracy numbers.
- pairs: top 3 Assam commodity–market pairs by weekly coverage, whatever they are. rule fixed before backtest.
- horizons distinct: short 1–4 wk → sell/hold/store; long season-ahead → crop choice.
- farmer output: plain EN + Assamese; band + signal + reason. ⊥ raw model output.
- futures optional: describe as method, ⊥ block on data. availability ? (some Indian agri futures suspended since 2021 — verify before any claim).
- stack locked (user): uv + marimo notebooks (pure `.py`, run as script) + polars LazyFrame + scikit-learn + seaborn (viz) + statsmodels (user-allowed 2026-11-25, Holt-Winters). pandas only @ seaborn boundary (`.to_pandas()`), ⊥ pandas for data processing.
- libs (user 2026-11-25): any Python library allowed. added: lightgbm.
- LightGBM (user 2026-11-25): alt learner of global model; settings mirrored from HGB, fixed before run, ⊥ tuned on test. not adopted (slightly worse).
- advisory model (user decision 2026-11-25, after seeing results): HGB global model trained on all Assam markets of the 11 good-data crops (Assam only) replaces pre-set config (3 pilot crops + neighbour States). post-hoc ∴ labelled in PDF §E + §J, answers §3 + §9 (pre-set result beside it). pre-set config still run (`out/backtest_prev.parquet`); stress test + seasonal profiles stay on pre-set config. advice only for pilot 3 (only crops w/ storage/shelf-life rules).
- Holt-Winters form (user hoped multiplicative): primary = add trend + add season on log price ≡ multiplicative on rupee price. textbook `seasonal="mul"` on rupee price added as labelled check.
- more commodities (user 2026-11-25: "choose commodities with good data"): rule = Assam market-weeks (dominant variety/market) ≥ 1,000 in last 156 complete wk ∧ ≥ 100 in last 52 (test yr). coverage only, fixed before any fit on them. Assam markets only (neighbour data = 3 pilot commodities only). = training data of advisory model; advice only for pilot 3 (shelf-life/storage rules exist for pilot 3 only).
- `COMMODITIES.md` (user): ∀ downloaded commodity name as published + rows, dates, markets, use | exclusion reason; generated, ⊥ hand edit.
- compute (user): ∀ heavy step ! use all CPU cores (joblib loky, 1 thread/worker).
- licence (user 2026-11-25): none granted now. README + answers §11 → "licence terms to be agreed with the department". ⊥ LICENSE file.
- other methods (user 2026-11-25): try Holt-Winters + "volatility pricing" (interpreted: EWMA vol → 80% band + put-value price of downside below floor; exploratory). ∀ result reported, win | lose. figures → PNG/SVG for PDF & site. PDF = HTML → `%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe --headless --print-to-pdf`.
- pace (user): ⊥ hurry; quality > speed. still keep submittable pkg early.
- downloads → `data/raw/` only. readers run `python -I`. scripts ∉ data dir.
- voice: SPEC.md caveman. answers, PDF, README, site → formal plain English for govt evaluators.
- timeline claim: 6 mo → validation-ready, then 1-season pilot (4–6 mo).
- cost claim: drivers + INR ranges labelled "indicative estimate, to confirm with department".
- publish: GitHub repo + Pages under `arijit-gogoi`. public visibility ! explicit user yes before `gh repo create` (yes given 2026-11-25; repo `arijit-gogoi/bazar-sanket` public, Pages from `main:/docs`). commits use GitHub noreply email.

## §I INTERFACES
- src: data.gov.in OGD API `https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24` (Agmarknet daily prices, §R.1). key env `DATA_GOV_IN_API_KEY` | git-ignored `.env`; ⊥ commit, ⊥ print.
- cmd: `uv run python -I src/fetch.py {probe|days|pull|consolidate}` → `data/raw/pages/**.json` (cached, w/ retrieval time) → `data/raw/ogd_prices.parquet`
- cmd: `uv run python -I -X utf8 nb/profile.py` → `out/selection.json`, `out/metrics_profile.json`, f1–f2
- cmd: `uv run python -I -X utf8 nb/forecast.py` (all cores) → `out/backtest.parquet` (advisory model, 11 crops), `out/backtest_prev.parquet` (pre-set config), `out/metrics_forecast.json`, f3–f6
- cmd: `uv run python -I -X utf8 nb/benchmarks.py` (all cores) → reads `out/backtest.parquet` (same rows) → `out/metrics_benchmarks.json`, f7
- cmd: `uv run python -I -X utf8 nb/extended.py` (all cores) → good-data commodity set (HGB, LightGBM, HW log, HW mul, naive, snaive); main rows: asserts advisory model ≡ its HGB, adds LightGBM + pre-set config → `out/backtest_extended.parquet`, `out/metrics_extended.json`, f8
- cmd: `uv run python -I -X utf8 src/commodities.py` → `COMMODITIES.md` (reads raw parquet, `out/selection.json`, `out/metrics_extended.json`)
- cmd: `uv run python -I -X utf8 src/render.py [--pdf]` → `out/metrics.json` (merge) → `answers.md`, `pow/proof-of-work.html`, `docs/`, `proof-of-work.pdf`
- cmd: `uv run python -I src/pipeline.py` → ∀ steps above in order (V15)
- file: `out/metrics.json` — sole source ∀ number in answers, PDF, site
- file: `answers.md` — 12 H2 sections, form order & titles, paste-ready prose
- file: `proof-of-work.pdf` ≤10 MB ← `pow/proof-of-work.html`
- file: `docs/index.html` + `docs/proof-of-work.pdf` → GitHub Pages
- form: Supporting PDF = `proof-of-work.pdf`; Product/Website URL = Pages URL; Video = empty; Additional = repo URL

## §R RESEARCH
id|topic|finding|src
R1|price data|OGD resource `35985678-0d79-46b4-9ed6-6f13308a1d24` "Variety-wise Daily Market Prices Data of Commodity" (DMI): Assam 553,821 rows 2006 → 2026-11-24; fields Arrival_Date (dd/mm/yyyy), State, District, Market, Commodity, Commodity_Code, Variety, Grade, Min/Max/Modal_Price (₹/quintal); ⊥ arrivals column|probe 2026-11-25
R2|API params|`filters[<Field>]` exact match (incl. `filters[Arrival_Date]=dd/mm/yyyy`), `sort[Arrival_Date]=desc`, `offset`, `limit`, `format=json\|csv` all work|probe 2026-11-25
R3|keys|public sample key → 10 rows/call & shared rate limit (HTTP 429 seen). personal key → `limit=10000` ok, deep `offset` ok (100000 → 2019); 10k page ≈ 1–3 min|probe 2026-11-25
R4|current-day API|resource `9ef84268-d588-465a-a308-a864a43d0070` = today only (Assam 111 rows on 2026-11-25), ⊥ history|probe 2026-11-25
R5|Agmarknet 2.0|agmarknet.gov.in = SPA (go-live 2025-11-07) over internal `api.agmarknet.gov.in/v1/`; ⊥ use: not a published API, call denied by policy|probe 2026-11-25
R6|dispersion|same day potato modal ₹850–2100/q across Assam markets (2026-11-23); multiple varieties per market ∴ series key = market × commodity × variety|probe 2026-11-25
R7|Assam gaps|Assam rows 2005-09 → 2026-11 = 515,000; 109 raw market names → 82 after harmonising (25 renamed, mostly "X" → "X APMC" at Agmarknet 2.0 cut-over); 223 of 1,106 weeks have no Assam price (longest gap 2020-01 → 2021-08, 86 wk; also 2025-09 → 2025-12)|`out/metrics_profile.json`
R8|selection|rule → Pamohi(Garchuk) × {Green Chilli, Cabbage, Pumpkin}, each 102/156 wk (65.4%); all perishable/semi-perishable vegetables|`out/selection.json`
R9|comparables|8 neighbour states fetched for 3 commodities, mostly since 2016 (336,760 rows; 17,305 = 5% older, back to 2001) (WB 195k rows, Tripura 82k, Nagaland 20k, Meghalaya 17k, Bihar 15k, Manipur, Mizoram, Arunachal small)|`data/raw/ogd_prices.parquet`
R10|Holt-Winters feasibility|statsmodels `ExponentialSmoothing` fit ≈ 0.1 s/origin; NaN ⊥ accepted → gaps must be filled. two unbroken yrs (gap ≤ 26 wk) @ only 84 of 1,290 Assam test origins; all gaps interpolated → 407|`out/metrics_benchmarks.json`
R11|downside pricing|EWMA vol (λ 0.94) √h-scaled overprices 90%-floor shortfall at 2–4 wk (weekly veg prices partly revert); ≈ right @ 1 wk|`out/metrics_benchmarks.json`
R12|commodity data|Assam 173 commodity names as published; good-data rule → 11 (pilot 3 + Tomato, Capsicum, Ginger(Green), Bitter gourd, Cauliflower, Brinjal, Bottle gourd, Bhindi). Lemon out (49 market-wk in test yr); Potato, Onion, Carrot, Beetroot out (few prices before test yr). some names duplicated by Agmarknet 2.0 rename (space before bracket), ⊥ merged, none modelled|`COMMODITIES.md`, `out/metrics_extended.json`
R13|LightGBM + more commodities|main rows: LightGBM skill −0.7% vs HGB −0.2%. 11 commodities (12,762 fc, 210 series): HGB +1.6%, beats naive 7/11; LightGBM +1.4%; HW log −4.5%, HW mul −4.7%; seasonal HW @ 221/3,523 origins, worse there. pilot rows w/ 11-commodity Assam-only pool +1.3% vs main −0.2% (post-hoc; pool vs ⊥ neighbour data not separated)|`out/metrics_extended.json`

## §V INVARIANTS
V1: `answers.md` ! exactly 12 H2 sections in form order; ∀ section ≤500 words (script check)
V2: `proof-of-work.pdf` ! single file ≤10 MB; ∀ page renders, ⊥ blank page, ⊥ missing-glyph boxes
V3: ∀ metric in answers, PDF, site ← `out/metrics.json` via `src/render.py`; ⊥ hand-typed number
V4: backtest ! rolling-origin out-of-sample; features @ origin t use data ≤ t only (incl. comparable-market & seasonal index); ⊥ lookahead
V5: ∀ model metric reported beside naive & seasonal-naive on same test set; model loses → say so
V6: ∀ selected pair reported; ⊥ cherry-pick
V7: incomplete-data test: good-coverage market masked/truncated → local-only vs pooled (comparable markets) compared & reported
V8: band = P10–P90; empirical coverage reported per horizon
V9: ∀ advisory shows current price, band, signal, reason, assumptions (storage cost, spoilage); perishable → ⊥ "store" past shelf life
V10: ∀ component labelled built+tested | mock-up | planned; ⊥ claim pilot, deployment, partner, data access, team member not real
V11: ∀ figure & table cites source + date range + retrieval date
V12: Assamese text labelled "draft translation, native review pending"; rendered via Chrome; ⊥ Assamese in matplotlib
V13: fallback mode → ∀ synthetic figure marked "illustrative"; ⊥ accuracy metric
V14: repo ⊥ secrets, API keys, user email
V15: one cmd regenerates processed data, metrics, figures from cached raw
V16: other methods (Holt-Winters log & mul, combination, volatility bands & pricing, LightGBM) scored on identical backtest rows; settings fixed before results; post-hoc variants labelled "check"; ∀ tried method reported; advisory model switched only by user decision, labelled post-hoc w/ pre-set result beside it
V17: ∀ answer section ≤500 words under whitespace, alnum & hyphen-split counts (portal counter unknown)
V18: extended commodity set ← coverage rule only (thresholds in `out/metrics_extended.json`); ∀ commodity name downloaded listed in `COMMODITIES.md` w/ use | exclusion reason

## §T TASKS
id|status|task|cites
T1|x|probe data ≤45 min: data.gov.in catalog (historical | variety-wise, Assam filter) → agmarknet.gov.in date-range export → Kaggle/GitHub mirror; record endpoint, fields (price + arrivals?), Assam coverage; decide real vs fallback|I.src,V13
T2|x|scaffold: `uv init` + deps (marimo, polars, pyarrow, scikit-learn, seaborn, requests); dirs `nb/ src/ data/raw/ data/processed/ out/ pow/ docs/`; `.gitignore`; git init w/ noreply email|V14
T3|x|fetch Assam all-commodity history + comparable markets (neighbour & major producer states) → `data/raw/`|I.src,V11
T4|x|clean → weekly series (median modal price, arrivals), gap flags; Assam coverage heatmap; pick top-3 pairs per rule|V6,V11
T5|x|model: pooled cross-market quantile model (log-return target, h=1..4 wk, NaN-native e.g. HistGradientBoosting) + naive + seasonal-naive|V4,V5
T6|x|rolling-origin backtest → MAPE, MAE ₹/q, direction hit-rate, P10–P90 coverage → `out/metrics.json`|V3,V4,V5,V8
T7|x|incomplete-data test: mask/truncate good-coverage market; local-only vs pooled|V7
T8|x|advisory rule: Hold/Sell/Store from band vs storage cost & shelf life; ? decision backtest vs always-sell-now|V9
T9|x|long-horizon card: pooled seasonal price calendar + volatility per commodity; labelled prototype|V10,V11
T10|x|mock-ups HTML/SVG: SMS & WhatsApp card (EN + AS), IVR script, officer dashboard, AMIU data-quality view; user journey & architecture diagrams|V10,V12
T11|x|figures: coverage heatmap, forecast band vs actual, metric tables|V3,V11
T12|x|answers template, 12 sections → `src/render.py` → `answers.md`; formal English|V1,V3,V10
T13|x|proof-of-work HTML → Chrome headless → PDF; check size & pages|V2,V3,V12
T14|x|verify: word counts, number trace, PDF size, labels, no secrets; regen from one cmd|V1,V2,V3,V14,V15
T15|x|git init + Conventional Commits; on user yes → `gh repo create` public + Pages from `docs/`; URLs → form|V14
T16|x|Holt-Winters (statsmodels, damped add trend + add 52-wk season, log price, re-fit ∀ origin, parallel all cores) + pre-set equal-weight avg w/ global model; same rows → `nb/benchmarks.py`|V5,V6,V16
T17|x|volatility pricing: EWMA σ → 80% band (interval score vs model quantiles) + Black-76 put value of shortfall below 90% / 100% floor, backtest predicted vs realised|V5,V16
T18|x|one-cmd runner `src/pipeline.py`; site nav links + `.nojekyll`|V15
T19|x|`COMMODITIES.md` generator `src/commodities.py` (in pipeline; README + site nav link)|V3,V15,V18
T20|x|LightGBM learner (`model.make(learner="lgbm")`), mirrored settings; scored on main rows (identical-row assert) + extended set|V5,V16
T21|x|textbook multiplicative HW (`smoothing.forecast_origin(form="mul")`) on main rows (benchmarks) + extended set; labelled check|V5,V16
T22|x|good-data commodity rule (`clean.commodity_coverage`) → `nb/extended.py`, Assam-only, accuracy only; f8 + 2 PDF pages + §4/§9/§11 text|V3,V5,V16,V18
T23|x|full pipeline rerun, ∀ page checked, word counts, secret scan, commit, push, site hash|V1,V2,V14,V15,V17
T24|x|switch advisory to 11-crop Assam model (user); pre-set config kept as comparison; post-hoc labels; ∀ text naming neighbour-State training fixed; rerun, verify, push|V3,V5,V10,V16

## §B BUGS
id|date|cause|fix
B1|2026-11-25|PDF said neighbour data "since 2016"; 5% of rows older (API pages returned earlier records)|wording → "mostly since 2016"; R9 amended
B2|2026-11-25|decision text 1 sentence/crop → section H spilled onto near-blank page|aggregate sentence when no holds advised; check every PDF page after render
B3|2026-11-25|polars `interpolate()` skips NaN (fills null only) → statsmodels init crash in probe|`np.interp` over observed weeks
