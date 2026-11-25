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
- stack locked (user): uv + marimo notebooks (pure `.py`, run as script) + polars LazyFrame + scikit-learn + seaborn (viz). pandas only @ seaborn boundary (`.to_pandas()`), ⊥ pandas for data processing. figures → PNG/SVG for PDF & site. PDF = HTML → `%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe --headless --print-to-pdf`.
- pace (user): ⊥ hurry; quality > speed. still keep submittable pkg early.
- downloads → `data/raw/` only. readers run `python -I`. scripts ∉ data dir.
- voice: SPEC.md caveman. answers, PDF, README, site → formal plain English for govt evaluators.
- timeline claim: 6 mo → validation-ready, then 1-season pilot (4–6 mo).
- cost claim: drivers + INR ranges labelled "indicative estimate, to confirm with department".
- publish: GitHub repo + Pages under `arijit-gogoi`. public visibility ! explicit user yes before `gh repo create`. commits use GitHub noreply email.

## §I INTERFACES
- src: data.gov.in OGD API `https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24` (Agmarknet daily prices, §R.1). key env `DATA_GOV_IN_API_KEY` | git-ignored `.env`; ⊥ commit, ⊥ print.
- cmd: `uv run python -I src/fetch.py {probe|days|pull|consolidate}` → `data/raw/pages/**.json` (cached, w/ retrieval time) → `data/raw/ogd_prices.parquet`
- cmd: `uv run python -I nb/pipeline.py` (marimo notebook) → `data/processed/*.parquet`, `out/metrics.json`, `out/fig/*`
- cmd: `uv run python -I src/render.py` → `answers.md`, `pow/proof-of-work.html` (metrics injected from `out/metrics.json`)
- cmd: `uv run marimo export html nb/pipeline.py -o docs/notebook.html` → notebook page on site
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
R9|comparables|8 neighbour states fetched for 3 commodities since 2016 (WB 195k rows, Tripura 82k, Nagaland 20k, Meghalaya 17k, Bihar 15k, Manipur, Mizoram, Arunachal small)|`data/raw/ogd_prices.parquet`
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

## §T TASKS
id|status|task|cites
T1|x|probe data ≤45 min: data.gov.in catalog (historical | variety-wise, Assam filter) → agmarknet.gov.in date-range export → Kaggle/GitHub mirror; record endpoint, fields (price + arrivals?), Assam coverage; decide real vs fallback|I.src,V13
T2|x|scaffold: `uv init` + deps (marimo, polars, pyarrow, scikit-learn, seaborn, requests); dirs `nb/ src/ data/raw/ data/processed/ out/ pow/ docs/`; `.gitignore`; git init w/ noreply email|V14
T3|x|fetch Assam all-commodity history + comparable markets (neighbour & major producer states) → `data/raw/`|I.src,V11
T4|x|clean → weekly series (median modal price, arrivals), gap flags; Assam coverage heatmap; pick top-3 pairs per rule|V6,V11
T5|.|model: pooled cross-market quantile model (log-return target, h=1..4 wk, NaN-native e.g. HistGradientBoosting) + naive + seasonal-naive|V4,V5
T6|.|rolling-origin backtest → MAPE, MAE ₹/q, direction hit-rate, P10–P90 coverage → `out/metrics.json`|V3,V4,V5,V8
T7|.|incomplete-data test: mask/truncate good-coverage market; local-only vs pooled|V7
T8|.|advisory rule: Hold/Sell/Store from band vs storage cost & shelf life; ? decision backtest vs always-sell-now|V9
T9|.|long-horizon card: pooled seasonal price calendar + volatility per commodity; labelled prototype|V10,V11
T10|.|mock-ups HTML/SVG: SMS & WhatsApp card (EN + AS), IVR script, officer dashboard, AMIU data-quality view; user journey & architecture diagrams|V10,V12
T11|.|figures: coverage heatmap, forecast band vs actual, metric tables|V3,V11
T12|.|answers template, 12 sections → `src/render.py` → `answers.md`; formal English|V1,V3,V10
T13|.|proof-of-work HTML → Chrome headless → PDF; check size & pages|V2,V3,V12
T14|.|verify: word counts, number trace, PDF size, labels, no secrets; regen from one cmd|V1,V2,V3,V14,V15
T15|.|git init + Conventional Commits; on user yes → `gh repo create` public + Pages from `docs/`; URLs → form|V14

## §B BUGS
id|date|cause|fix
