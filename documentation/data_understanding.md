# Data understanding and source precedence

## Current architecture — November 2026 migration

`data/RM_Prices_XHEC_Challenge_v1.xlsx` is authoritative wherever its contents overlap with prior sources. Actual inspection found one sheet, `BOM_Materials`, with 22 material rows and five summary rows. It contains no monthly prices, source catalog, currencies or publication timestamps. The loader separates summary metadata from material observations.

The retained `data/data/RM_Prices_XHEC_Challenge.xlsx` supplies the price panels, FX, source catalog and dictionary; its embedded BOM is superseded. The independent energy workbook supplies energy and regional FX history. The prior standalone BOM and CSV exports are preserved locally in `data/.archive/before_v1/`; uncertain cloud exports are archived without permanent deletion. Supplementary market and supplier sources remain available for reconciliation and context.

The generated current catalog, numerical quality findings, BOM differences, EDA comparisons and executed notebook are under ignored `data/eda_presentation/`. The inventory and sheet inspections are under ignored `data/audit/`. See [README](../README.md) for reproducible commands. September-origin selection uses the explicit September 30 decision cutoff and separately labeled lag assumptions; it does not establish verified historical availability.

## Historical Issue #2 audit (superseded snapshot)

The audit below is preserved as historical evidence. Its old paths, row counts, BOM precedence and missing-field findings describe the pre-v1 snapshot and must not override the current architecture above or the generated current catalog.


## 1. Executive summary

The project supports raw-material and BOM cost forecasting for ETO pricing and gross-margin protection, while using market evidence to inform competitive pricing. The challenge forecast origin is **September 2026**. Required forecast targets are **December 2026 (3 months), March 2027 (6 months), and June 2027 (9 months)**.

The local data directory is lowercase `data/` and is Git-ignored. The audit found monthly raw-material and energy panels, BOM versions, auxiliary commodity and market inputs, competitor benchmarks, and supplier financial reports. Core RM and energy panels are structured and monthly; the auxiliary workbooks are heterogeneous and need additional source, licensing, availability, and schema validation.

By project decision, the standalone `data/RM_Prices_XHEC_Challenge.xlsx` is the authoritative source for **all BOM fields**. Its BOM differences with the nested workbook and CSV are resolved by precedence, but missing quantities/mappings, incomplete coverage, row-key ambiguity, and cost-weight limitations remain unresolved. Never fill authoritative blanks from secondary copies without explicit approval.

The readable nested RMU benchmark workbook is for competitive market context only: its model-labeled synthetic price bands are not confirmed actual transaction prices. The older protected OLE-format file at the root data path remains unchanged and was not opened by bypassing protection.

This is descriptive data understanding and preliminary EDA, not formal model validation. No preprocessing pipeline, baseline forecast, rolling backtest, or model was implemented or trained. The October 2026 raw-material market report and all other post-origin observations are excluded from the official September-origin experiment.

## 2. Dataset inventory

The following inventory covers 31 files found under `data/`. Dimensions for workbooks refer to used worksheet ranges and can include titles, blank columns, merged headers, or formulas; they are not necessarily clean table sizes. Supplier financial report filenames and observations are omitted from this public document.

### Core RM, BOM, and documentation

| Relative path | Format / size | Structure / coverage | Purpose and limits |
|---|---|---|---|
| `data/data/rm_prices_monthly_wide.csv` | CSV, 19,615 B | 120 × 26; monthly Nov 2016–Oct 2026. | Raw-material panel in wide form. Most price series end Sep 2026; FX continues to Oct. |
| `data/data/rm_prices_monthly_long.csv` | CSV, 181,258 B | 1,548 × 10; 13 observed source IDs. | Tidy RM panel, one observed value per month/series. |
| `data/data/source_catalog.csv` | CSV, 4,706 B | 23 × 13; one row per RM ID. | Series definition, geography, unit, source, access and proxy metadata. |
| `data/data/bom_materials.csv` | CSV, 2,022 B | 17 × 10; material and summary rows. | Secondary BOM representation only; not authoritative for BOM fields. |
| `data/data/data_dictionary.csv` | CSV, 2,821 B | 25 × 7. | Field descriptions and rules for the core challenge workbook. |
| `data/data/README.md` | Markdown, 21,295 B | Source guide for core RM and energy data. | Coverage, definitions, licensing and refresh notes; scripts it references are not present in the local repository tree. |
| `data/data/RM_Prices_XHEC_Challenge.xlsx` | XLSX, 118,628 B | README 21×1; BOM 18×10; source catalog 24×13; wide monthly 121×26; long monthly 1,549×10; dictionary 26×7. | Main RM panels and metadata. Its BOM sheet is lower-priority than the standalone workbook. |
| `data/RM_Prices_XHEC_Challenge.xlsx` | XLSX, 11,617 B | BOM_Materials only, 18×10 including header. | Authoritative BOM source by project decision. |
| `data/data.zip` | ZIP, 358,401 B | 11 core nested-data files. | Archive copies match the expanded core files byte-for-byte; not additional data. |

### Energy and auxiliary commodity inputs

| Relative path | Format / size | Structure / coverage | Purpose and limits |
|---|---|---|---|
| `data/data/energy_prices_monthly_wide.csv` | CSV, 31,157 B | 120×31; monthly Nov 2016–Oct 2026. | Candidate regional/global exogenous variables, not BOM prices. |
| `data/data/energy_prices_monthly_long.csv` | CSV, 404,184 B | 3,573×9; 30 IDs, with some series ending Sep or Jul 2026. | Tidy energy panel. |
| `data/data/energy_source_catalog.csv` | CSV, 7,395 B | 30×13; one row per energy ID. | Source, coverage, region, unit, access and relevance metadata. |
| `data/data/Energy_Prices_XHEC_Challenge.xlsx` | XLSX, 193,093 B | README 7×1; catalog 31×13; wide 121×31; long 3,574×9. | Workbook representation of the energy panel and catalog. |
| `data/LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx` | XLSX, 59,273 B | Price Trends 11×8; Monthly Prices 136×16; Method & Sources 17×8; Jan 2016–Nov 2026. | Copper/aluminium cross-check or candidate input. Nov 2026 is provisional month-to-date; Oct/Nov are post-origin. |
| `data/Epoxy for student.xlsx` | XLSX, 20,336 B | Feuil1 107×7; 106 nominally monthly rows Jan 2018–Oct 2026. | Resin/material cost-driver candidate; labels indicate EUR/tonne, but source and construction need validation. |
| `data/Copia de Europe Steel INDEXES.xlsx` | XLSX, 932,708 B | 13 heterogeneous sheets, approximately 2–665×6–73; recognized dates span 1997–2029. | Potential steel-price inputs. Mixed frequencies, multi-row headers, possible vendor forecasts and restricted source content require review. |

### Competitive, market, and supplier-risk inputs

| Relative path | Format / size | Structure / coverage | Purpose and limits |
|---|---|---|---|
| `data/data/RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx` | XLSX, 54,611 B | ML_Trend 64×7 (header row 4, 60 records); Chart_Data 13×22; six brands, 2017–2026. | Newly accessible model-labeled synthetic benchmark; use for cautious competitive context only, not realized selling prices or RM targets. |
| `data/RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx` | OLE/Compound File, 90,112 B | Not readable by the available standard XLSX reader. | Older protected copy. Retain unchanged; do not bypass protections or treat it as the readable nested workbook. |
| `data/benchmark-modules-fonctionnels-rm6-ne-iqi-3-fonctions-400a-appareillage-isolement-et-coupure-dans-le-gaz-sf6.xlsx` | XLSX, 65,718 B | Nine sheets: benchmark, six competitor/product BOMs, price sources and methodology; sheet sizes about 23–565×4–7. | Provisional RM6 competitor/teardown intelligence; not a clean time series, and some configurations are unpriced or inferred. |
| `data/benchmark-modules-fonctionnels-rm6-ne-iqi-3-fonctions-400a-appareillage-isolement-et-coupure-dans-le-gaz-sf6.pdf` | PDF 1.3, 1,621,354 B | Page structure not inspected; no PDF metadata/reading utility was available. | Companion market reference; contents and permissions remain unverified. |
| `data/Supplier_balancesheet/` | Nine XLSX reports; 39,240–88,256 B each. | Report-style sheets such as Overview, Income Statement, Balance Sheet, Cash Flow, Ratios, Deal Multiples, Disclaimer; layouts vary. | Potential supplier financial risk inputs only. Supplier names and financial observations are omitted; report periods, currencies, rights and BOM link are unverified. |
| `data/Supplier_balancesheet.zip` | ZIP, 434,905 B | Nine supplier report workbooks. | Members are byte-identical to expanded copies; not additional data. |
| `data/.DS_Store` | macOS metadata, 6,148 B | Finder metadata. | No analytical role. |

The separate October 2026 raw-material market report is post-origin information. It must be treated only as retrospective market intelligence and must not inform the September-origin experiment, feature selection, tuning, or model selection.

## 3. Dataset classification and business relevance

| Classification | Datasets | Business use and readiness |
|---|---|---|
| Raw-material forecasting targets | S01/S02 direct prices; D01/D02 derived EUR prices; selected material PPIs S07–S13 if explicitly modeled as indices. | Candidate series for 3/6/9-month forecasting. Targets must match validated BOM material, region, unit and business cost basis. |
| Cost drivers / exogenous variables | S03–S05; S06 FX; energy E/ED series; auxiliary LME, epoxy and steel series after validation. | Potential explanatory features, not automatically causal or available at forecast time. |
| BOM and product cost modeling | Standalone BOM workbook; RM catalog and price series. | Partial monetary index only where authoritative quantities and compatible direct prices are validated. Mass weighting is a separate proxy method. |
| Competitive benchmarking and pricing | Readable RMU synthetic benchmark; provisional RM6 benchmark workbook and PDF. | Market context and scenario comparison only after product/configuration, market, currency, tax and commercial-term comparability checks. |
| Market intelligence | RMU and RM6 material; any October 2026 market report. | Contextual evidence, not actual margin outcomes; October report is retrospective only. |
| Supplier financial risk | Nine supplier workbooks and archive. | Separate controlled analysis; no confirmed join to BOM suppliers and not a monthly price-forecast target. |
| Reference documentation | Core README, data dictionary, source catalogs, workbook README/method tabs. | Definitions and provenance; verify local availability and licensing. |

## 4. Data dictionary and series definitions

Field types are inferred from the files unless stated in the supplied dictionary. “Blank/not applicable” metadata is distinct from a missing observation. Catalog definitions below follow the actual local source catalogs; units, geography and access are series-specific.

### RM series: S01–S13, D01–D02, L01–L08

| ID | Catalog definition | Geography | Unit / currency | Source / type | Proposed role and local coverage |
|---|---|---|---|---|---|
| S01 | LME grade A cathode copper, monthly average | World | USD/t | IMF via FRED; direct | Target candidate / driver; Nov 2016–Sep 2026, 119 observations. |
| S02 | LME aluminium 99.5% minimum, monthly average | World | USD/t | IMF via FRED; direct | Target candidate / driver; Nov 2016–Sep 2026, 119. |
| S03 | LME nickel, monthly average | World | USD/t | IMF via FRED; stainless-steel driver | Feature candidate; Nov 2016–Sep 2026, 119. |
| S04 | Iron ore fines, 62% Fe, CFR China | World market | USD/dmt | IMF via FRED; carbon-steel driver | Feature candidate; Nov 2016–Sep 2026, 119. |
| S05 | LME zinc, monthly average | World | USD/t | IMF via FRED; galvanised-steel driver | Feature candidate; Nov 2016–Sep 2026, 119. |
| S06 | EUR/USD reference, expressed as USD per EUR | EU | USD per EUR | ECB; FX conversion | Exogenous conversion feature; Nov 2016–Oct 2026, 120. October is post-origin. |
| S07 | PPI, copper production (NACE C24.44) | EU27 | Index, 2021=100 | Eurostat; PPI proxy | Index target/proxy candidate, not currency cost; Nov 2016–Sep 2026, 119. |
| S08 | PPI, aluminium production (NACE C24.42) | EU27 | Index, 2021=100 | Eurostat; PPI proxy | Index target/proxy candidate; Nov 2016–Sep 2026, 119. |
| S09 | PPI, basic iron and steel / ferro-alloys (NACE C24.10) | EU27 | Index, 2021=100 | Eurostat; PPI proxy | Steel-index target/proxy candidate; Nov 2016–Sep 2026, 119. |
| S10 | PPI, basic iron and steel (NACE C24.10) | Germany | Index, 2021=100 | Eurostat; PPI proxy | Galvanised-steel proxy candidate; Nov 2016–Sep 2026, 119. |
| S11 | PPI, plastics in primary forms (NACE C20.16) | EU27 | Index, 2021=100 | Eurostat; PPI proxy | Plastics proxy candidate; Nov 2016–Sep 2026, 119. |
| S12 | PPI, steel mill products | USA | Index, 1982=100 | US BLS via FRED; PPI proxy | US steel proxy candidate; Nov 2016–Sep 2026, 119. |
| S13 | PPI, plastics materials and resins manufacturing | USA | Index, Dec 1980=100 | US BLS via FRED; PPI proxy | US plastics proxy candidate; Nov 2016–Sep 2026, 119. |
| D01 | Copper LME price converted to EUR (`S01 / S06`) | World price in EUR | EUR/t | Derived | EUR target candidate; Nov 2016–Sep 2026, 119. Depends on S01 and S06; not independent. |
| D02 | Aluminium LME price converted to EUR (`S02 / S06`) | World price in EUR | EUR/t | Derived | EUR target candidate; Nov 2016–Sep 2026, 119. Depends on S02 and S06; not independent. |
| L01 | Cold-rolled carbon-steel coil benchmark | Europe | EUR/t | Licensed direct price | Unavailable: 0 observations. Do not model or substitute silently. |
| L02 | 304L stainless-steel sheet benchmark | Europe | EUR/t | Licensed direct price | Unavailable: 0 observations. Do not model or substitute silently. |
| L03 | Hot-dip galvanised coil benchmark | Germany / EU | EUR/t | Licensed direct price | Unavailable: 0 observations. Do not model or substitute silently. |
| L04 | PBT resin benchmark | Europe | EUR/t | Licensed direct price | Unavailable: 0 observations. Do not model or substitute silently. |
| L05 | PET resin benchmark | Europe | EUR/t | Licensed direct price | Unavailable: 0 observations. Do not model or substitute silently. |
| L06 | Engineering thermoplastics basket benchmark | Europe | EUR/t | Licensed direct price | Unavailable: 0 observations. Restricted series definition/value details are not reproduced. |
| L07 | Aluminium duty-paid premium benchmark | Europe | USD/t | Licensed premium | Unavailable: 0 observations. Do not model or substitute silently. |
| L08 | Copper cathode premium benchmark | Europe | USD/t | Licensed premium | Unavailable: 0 observations. Do not model or substitute silently. |

`S01`–`S05` are direct commodity prices or material drivers; `S07`–`S13` are price indices rather than currency/tonne prices. S06 is FX. D01/D02 are arithmetic currency conversions, where USD per EUR is the divisor to obtain EUR from USD. L01–L08 are licensed and empty. Indices with different base years are not directly comparable in level.

### Energy series: E01–E24 and ED01–ED06

| ID | Catalog definition | Geography | Unit | Source / type | Local coverage and candidate role |
|---|---|---|---|---|---|
| E01 | Brent spot crude, monthly average | Europe / World | USD/bbl | IMF via FRED; commodity | Nov 2016–Sep 2026; possible energy feature. |
| E02 | WTI crude at Cushing, monthly average | USA | USD/bbl | IMF via FRED; commodity | Nov 2016–Sep 2026; possible energy feature. |
| E03 | Dubai crude (Fateh), monthly average | Asia / Middle East | USD/bbl | IMF via FRED; commodity | Nov 2016–Sep 2026; possible energy feature. |
| E04 | Crude petroleum import price index, yen basis | Japan | Index, 2020=100 | Bank of Japan; index | Nov 2016–Oct 2026; possible feature, October excluded at origin. |
| E05 | Netherlands TTF natural gas, monthly average | Europe | USD/MMBtu | IMF via FRED; commodity | Nov 2016–Sep 2026; possible energy feature. |
| E06 | Henry Hub natural gas, monthly average | USA | USD/MMBtu | IMF via FRED; commodity | Nov 2016–Sep 2026; possible energy feature. |
| E07 | Japan import price of LNG, monthly average | Japan / Asia | USD/MMBtu | IMF via FRED; commodity | Nov 2016–Sep 2026; possible energy feature. |
| E08 | Newcastle thermal coal | Asia-Pacific | USD/t | IMF via FRED; commodity | Nov 2016–Sep 2026; possible energy feature. |
| E09 | Wholesale day-ahead electricity, monthly average | Germany | EUR/MWh | Ember; electricity price | Nov 2016–Oct 2026; October excluded at origin. |
| E10 | Wholesale day-ahead electricity, monthly average | France | EUR/MWh | Ember; electricity price | Nov 2016–Oct 2026; October excluded at origin. |
| E11 | Wholesale day-ahead electricity, monthly average | Hungary | EUR/MWh | Ember; electricity price | Nov 2016–Oct 2026; October excluded at origin. |
| E12 | JEPX day-ahead system price, mean of half-hourly prices | Japan | JPY/kWh | JEPX; electricity price | Nov 2016–Oct 2026; October excluded at origin. |
| E13 | PPI, electricity, gas, steam and air-conditioning supply (NACE D35) | EU27 | Index, 2021=100 | Eurostat; PPI | Nov 2016–Sep 2026; possible energy feature. |
| E14 | PPI, NACE D35 | Germany | Index, 2021=100 | Eurostat; PPI | Nov 2016–Sep 2026; possible energy feature. |
| E15 | PPI, NACE D35 | France | Index, 2021=100 | Eurostat; PPI | Nov 2016–Sep 2026; possible energy feature. |
| E16 | PPI, electric-power generation, transmission and distribution (NACE D35.1) | Germany | Index, 2021=100 | Eurostat; PPI | Nov 2016–Sep 2026; possible energy feature. |
| E17 | PPI, industrial electric power | USA | Index, 1982=100 | US BLS via FRED; PPI | Nov 2016–Sep 2026; possible energy feature. |
| E18 | PPI, natural gas | USA | Index, 1982=100 | US BLS via FRED; PPI | Nov 2016–Sep 2026; possible energy feature. |
| E19 | Corporate-goods PPI, electricity | Japan | Index, 2020=100 | Bank of Japan; PPI | Nov 2016–Oct 2026; October excluded at origin. |
| E20 | EU27 crude-oil imports from Russia | EU27 imports | thousand t | Eurostat; trade volume | Nov 2016–Jul 2026; lagged feature candidate, not a price. |
| E21 | EU27 crude-oil imports, all partners | EU27 imports | thousand t | Eurostat; trade volume | Nov 2016–Jul 2026; lagged feature candidate, not a price. |
| E22 | IMF aggregate energy price index (crude, gas, coal, propane) | World | Index, 2016=100 | IMF via FRED; index | Nov 2016–Sep 2026; possible energy feature. |
| E23 | EUR/USD reference, USD per EUR, monthly average | EU | USD per EUR | ECB; FX | Nov 2016–Oct 2026; October excluded at origin. |
| E24 | EUR/JPY reference, JPY per EUR, monthly average | EU | JPY per EUR | ECB; FX | Nov 2016–Oct 2026; October excluded at origin. |
| ED01 | Brent converted to EUR (`E01 / E23`) | Europe / World | EUR/bbl | Derived | Nov 2016–Sep 2026; feature candidate, dependent on E01/E23. |
| ED02 | TTF converted to EUR/MWh (`E05 × 3.412 / E23`) | Europe | EUR/MWh | Derived | Nov 2016–Sep 2026; feature candidate, dependent on E05/E23. |
| ED03 | Asia LNG converted to EUR/MWh (`E07 × 3.412 / E23`) | Japan / Asia | EUR/MWh | Derived | Nov 2016–Sep 2026; feature candidate, dependent on E07/E23. |
| ED04 | Henry Hub converted to EUR/MWh (`E06 × 3.412 / E23`) | USA | EUR/MWh | Derived | Nov 2016–Sep 2026; feature candidate, dependent on E06/E23. |
| ED05 | JEPX converted to EUR/MWh (`E12 × 1000 / E24`) | Japan | EUR/MWh | Derived | Nov 2016–Oct 2026; October excluded at origin. |
| ED06 | Russia share of EU27 crude imports (`E20 / E21`) | Russia → EU27 | Share, 0–1 | Derived | Nov 2016–Jul 2026; feature candidate, dependent on E20/E21. |

Catalog access labels are not a substitute for checking redistribution terms. Derived series inherit the availability and revision constraints of their inputs.

### Common structured fields and identifiers

| Field(s) | Meaning / type | Grain, units, missingness and key role |
|---|---|---|
| `month` | Reference month; date. | Monthly row in wide panels; part of `(month, series_id)` in long panels. |
| `year`, `month_num` | Calendar features; integer. | Wide-panel fields derived from `month`. |
| `series_id` | Source-series identifier; string. | Long observation key component; joins to the respective source catalog. |
| `value` | Observed value; numeric. | Units vary by series; no nulls among observed long rows. |
| `unit`, `currency`, `region`, `material`, `energy` | Series descriptors; text/category. | Currency may be blank or marked not applicable for FX and indices; units vary. |
| `source`, `provider`, `provider_code`, `retrieved_at` | Provenance metadata; text/datetime. | Retrieval timestamp is not publication time or historical availability. |
| Catalog `first_month`, `last_month`, `n_months`, `access`, `proxy_type`, `relevance` | Series metadata; date/integer/category. | Verify catalog coverage against observations and point-in-time eligibility. |
| BOM `product_ref`, `plant_code`, `plant_name` | Configuration and production-site identifiers; text. | Summary rows may omit plant fields. No stable unique BOM row key is documented. |
| BOM `material_label_source`, `material_family` | Original material label and normalized category; text. | Preserve source label; harmonized family is a candidate mapping, not proof of purchasing equivalence. |
| BOM `mass_kg`, `share_pct` | Quantity and mass share; numeric. | kg per product unit and fraction/percent per dictionary. `share_pct` is mass share, never cost weight. |
| BOM `eu_series_ids`, `ww_series_ids` | Semicolon-separated candidate series mappings; text. | IDs resolve in RM catalog, but business validity remains unconfirmed. |

## 5. Dataset relationships and join keys

| Relationship | Status | Notes |
|---|---|---|
| RM long `(month, series_id)` → RM source catalog `series_id` | Verified structural relationship | RM catalog IDs are unique. The long data includes 13 observed IDs; the catalog also defines derived and licensed IDs. |
| Energy long `(month, series_id)` → energy catalog `series_id` | Verified structural relationship | 30 observed IDs and 30 unique catalog IDs. |
| Authoritative BOM `eu_series_ids` / `ww_series_ids` → RM catalog | IDs verified; semantic mapping proposed | 40 references across 22 distinct IDs resolve in the catalog. This does not validate material-price suitability. |
| USD commodity price and FX → derived EUR commodity series | Documented formula relationship | `D01 = S01 / S06`; `D02 = S02 / S06`, where S06 is USD per EUR. Derived EUR prices are not independent observations. |
| Energy source and derived energy series | Documented formula relationship | ED01–ED06 depend on listed E series and, for currency conversion, FX. Do not treat parent and derived columns as independent evidence. |
| RM/energy/auxiliary prices → month | Candidate temporal alignment | Monthly periods can align, but publication vintages, region, unit and currency must be validated before joining. |
| RMU benchmark → BOM, product configuration or actual quote | Not established | No validated configuration, country, transaction, cost or realized-sales key. |
| RM6 competitor benchmark → RMU benchmark / BOM | Not established | No validated configuration bridge. RM6/RM AirSeT and SM6/SM AirSeT are not assumed interchangeable. |
| Supplier reports → BOM suppliers/materials | Not established | No validated supplier identifier or purchasing relationship. |

Do not join datasets merely because both contain a date or similar material label. Preserve each source's native grain until keys and business semantics are confirmed.

## 6. Data source hierarchy and version control

### BOM precedence

For **all BOM fields**, use this hierarchy:

1. **Authoritative:** `data/RM_Prices_XHEC_Challenge.xlsx` → `BOM_Materials`.
2. Lower-priority comparison only: `data/data/RM_Prices_XHEC_Challenge.xlsx` → `BOM_Materials`.
3. Lower-priority comparison only: `data/data/bom_materials.csv`.

The standalone workbook takes precedence for conflicting material labels, quantities, mass shares and material-series mappings. Do not replace a blank authoritative value from a secondary version without explicit justification and approval. This rule resolves which version to use; it does not resolve missing values, incompleteness, row keys or cost weights.

### Other files

- The nested RM workbook and core CSVs remain the sources for raw-material panels, catalogs and data dictionary; their BOM copies are not authoritative.
- Use the newly accessible nested RMU benchmark workbook for cautious competitive analysis. Retain the old protected root copy unchanged; do not bypass file protections or treat it as the new version.
- Archive files are byte-identical packaging duplicates of their expanded members; treat each observation only once.
- Record file path, version/date, source, licensing status, retrieval time and approved transformations for every future analytical derivative. Do not modify the original inputs.

### BOM discrepancies and applied resolution

Audit compared material rows in workbook order within each of the two BOM-bearing configurations; no unique material-row identifier is documented, so this is a field-level discrepancy count, not a claim of semantic row matching. No raw BOM values are reproduced.

| Affected scope | Lower-priority copies versus standalone source | Resolution | Potential costing impact |
|---|---|---|---|
| Configuration A, material rows | In both nested workbook and CSV: one material-label cell, five `share_pct` cells and five `mass_kg` cells differ. | Use standalone values. Do not reconcile or average versions. | Changes alter a mass-weighted proxy and possibly classification/quantity inputs; they do not establish monetary cost weights. |
| Configuration B, material rows | In both nested workbook and CSV: one material-label cell, six `share_pct` cells and six `mass_kg` cells differ. | Use standalone values. Do not reconcile or average versions. | Same proxy and partial-cost sensitivity; no claim of total product COGS. |
| Summary rows | In each lower-priority copy, one `product_ref` and three `mass_kg` cells differ. | Use standalone summary-row values; do not mix totals with material rows or use alternate totals to fill missing inputs. | Mistaking summaries for materials or double-counting them can distort aggregates. |

No positional differences were observed for regional mapping fields, but three authoritative rows have missing mappings. The secondary files contain related gaps and cannot validate the correct material association.

## 7. Data quality assessment

- **Calendar completeness:** RM and energy wide panels each have 120 consecutive monthly rows. Individual series have end-of-series blanks; no interior calendar gaps were found in their observed long series.
- **Missingness:** RM wide fields S01–S05 and S07–S13 have 119 observations each through Sep 2026; S06 has 120 through Oct. D01/D02 have 119 each. L01–L08 are empty. Energy series counts vary: nine extend to Oct 2026, 18 to Sep and three to Jul. Long tables contain only observed rows and have no null `value` fields.
- **Duplicate keys:** No duplicate `(month, series_id)` keys were found in RM or energy long data. Wide panels have unique months. RMU `Brand+Year` is unique; its `ID` field repeats by brand and is not a primary key.
- **Workbook/CSV consistency:** Keyed RM and energy long panels match their corresponding workbook panels. Wide RM CSV/workbook values have small representation/rounding differences. Do not treat duplicate layouts as additional observations.
- **Archive duplicates:** Both ZIP files pass integrity checks. Core archive members and supplier report members match their expanded copies byte-for-byte.
- **BOM conflicts:** Version conflicts are resolved by the source hierarchy in Section 6. Remaining authoritative-source gaps and lack of a stable row key remain open.
- **Units and currencies:** RM data include USD/t, USD/dmt, USD per EUR, EUR/t and PPIs with different base years. Energy data mix commodity units, USD/EUR/JPY pricing, FX, indices, trade volumes and shares. Convert compatible direct prices before monetary aggregation. Never add PPI index points to currency prices.
- **Irregular frequency:** Core RM/energy panels are monthly. Epoxy has continuous month coverage, but dates fall on different days of the month. Steel workbook sheets have mixed layouts/frequencies and cannot be treated as one monthly series without sheet-level validation.
- **Potential outliers:** A descriptive 3×IQR screen on RM levels/monthly changes through Sep 2026 flagged a small number of observations (including iron-ore levels and selected commodity/PPI changes). These are review flags, not proven errors; nothing was removed or altered.
- **Formula errors:** The steel mapping sheet contains two explicit `#N/A` cells. No explicit Excel error cells were found in other readable workbooks. Formula cells were not recalculated, so cached or formula-derived values are not independently validated.
- **Benchmark limitations:** RMU series are labeled synthetic/model-generated, but source, generation method, ML process, quantile calibration and price meaning are unconfirmed. RM6 material is explicitly provisional/incomplete. Neither establishes realized transaction price.
- **Inaccessible file:** The older root RMU OLE file is not readable using the available standard reader. No protection was bypassed; use only the readable nested workbook for the currently approved benchmark role.

## 8. Preliminary EDA findings

The descriptive screen uses core RM observations through September 2026 only. It does not assert that every September value was published by the challenge cutoff and is not a forecast or formal model evaluation.

- Direct commodity price series show positive endpoint changes over the available history, with substantial differences in monthly volatility. Nickel and iron ore are among the more volatile direct series in the simple descriptive screen.
- EUR-converted copper and aluminium co-move by construction with their USD commodity prices and FX.
- PPI series trend over their history but measure index movement, not currency cost; their base years and geographies differ.
- A simple month-of-year screen is inconclusive for stable, repeatable seasonality. Do not assume seasonal structure or seasonally adjust without later time-ordered validation.
- The coarse outlier flags require source, revision and event review. They do not justify deletion, smoothing or winsorization.
- Inflation adjustment is a candidate experiment, not an assumed improvement. Compare nominal forecasting, deflated/reinflated forecasts, and nominal forecasts with a suitable inflation feature where permitted and available at each origin.

These observations support prioritizing validated series for further EDA; they do not demonstrate forecast accuracy, calibration, causal feature value or margin improvement.

## 9. Forecasting target and feature candidates

| Candidate | Proposed role | Readiness for monthly forecasting and September origin |
|---|---|---|
| S01/S02 direct copper/aluminium | Primary raw-material target candidates if consistent with BOM and business procurement basis. | 119 monthly observations through Sep 2026. Validate September release availability and whether global benchmarks represent local costs. |
| D01/D02 EUR copper/aluminium | Alternative currency-specific targets for EU cost scenarios. | 119 months through Sep; derived from USD price and FX, not independent. Validate FX cutoff. |
| S03–S05 nickel, iron ore, zinc | Material drivers; target status only if client confirms direct exposure. | 119 months each. Monthly history available, but material-to-cost relation is indirect and units differ. |
| S06 FX | Currency conversion feature. | Sep observation present, Oct also present but post-origin. Confirm release vintage and exclude all post-cutoff data. |
| S07–S13 PPI series | Index targets or material proxy features, depending on approved use. | 119 months each through Sep. Different bases and regions; forecast relative index movement, not a direct monetary price. |
| L01–L08 licensed series | None until lawfully available. | Zero observations; not suitable for training or baseline evaluation. |
| E01–E24 / ED01–ED06 | Exogenous energy, FX, index or trade candidates, not BOM cost targets. | 117–120 calendar observations by series. Select based on causal/business relevance, publication timing, license and region; derived features inherit parent availability. |
| Auxiliary LME copper/aluminium | Cross-check or alternative target/input after source and unit reconciliation. | Monthly data through Sep are potentially relevant; Oct and provisional Nov are excluded at origin. Overlaps core prices. |
| Epoxy | Resin-related feature or target if its meaning matches a BOM material. | Jan 2018–Sep 2026 is the eligible date window; exclude Oct, validate non-first-day dates, source and publication vintage. |
| Steel indexes | Possible steel target/driver only after sheet-level identification. | Mixed history and future/vendor forecast dates through 2029. Confirm license, units, history-versus-forecast status, and as-of availability before use. |
| RMU synthetic benchmark | Market reference only. | Annual model outputs, no actual quote/trade target; not a raw-material series. |
| RM6 benchmark | Competitive intelligence only. | Incomplete, provisional and non-monthly; not a forecast target. |
| Supplier financial reports | Supplier-risk analysis only. | Different reporting periods/grains; not monthly material-price targets or validated exogenous features. |

Future baseline and rolling backtesting require approved targets, a point-in-time feature set, historical data availability, and enough rolling origins for each horizon. The business problem statement proposes LOCF, but this Issue #2 audit does not train or score it.

## 10. BOM costing readiness

The authoritative workbook contains material rows and summary rows, but material coverage, row identity and selected quantities/mappings are incomplete. Three authoritative rows have missing quantities, mass shares and regional series mappings; do not backfill them from secondary copies.

**Preferred future method — partial monetary cost index.** For validated authoritative quantities and compatible direct unit prices, convert units/currencies and calculate included-material spend; normalize against an agreed base. Disclose included materials, omissions, currency, base, and configuration. This remains partial and is not total COGS.

Direct commodity prices may be multiplied by compatible validated quantities. PPI proxies represent relative index movement; monetary estimation requires an approved baseline spend/anchor and applying relative movement, not treating index points as currency per tonne.

**Fallback — mass-weighted proxy.** Where monetary costing is not feasible, combine normalized price relatives using validated material masses or mass shares, clearly label it a mass-weighted proxy, and report omitted materials. Mass shares are not cost weights and this proxy does not establish a product cost.

## 11. Competitive benchmarking readiness

The readable RMU benchmark contains 60 annual brand-level rows across six brands and labels its outputs P10/P50/P90 (ML) with EUR/ex-VAT notes. It is structurally complete for the six principal fields, but `ID` is repeated; use brand-year only as the observed unique combination. Six initial year-over-year fields are placeholders.

The “synthetic” and “ML” labels do not establish how values were generated, what prices represent, how quantiles were calibrated, or whether the values correspond to comparable products or actual market transactions. No configuration, country, BOM cost, customer, transaction terms or realized quote identifier is available. Therefore:

- Do not use the benchmark as a raw-material forecasting target, actual selling-price history, margin label or evidence of realized competitive outcomes.
- Do not infer a SF6-free premium from the workbook’s chart-only marker fields.
- Use it only as a clearly qualified scenario/reference input after Schneider Electric validates source, generation, product comparability, regional scope, currency, VAT and commercial conditions.
- Keep the provisional RM6 benchmark separate. Do not assume RM6/RM AirSeT and SM6/SM AirSeT are interchangeable.

## 12. Data leakage and publication-lag considerations

The official origin is September 2026. Exclude October 2026 observations from all inputs, transformations, feature selection and model selection. The provisional November LME observation and any later market reports are also excluded. The October 2026 raw-material report is retrospective intelligence only.

An observation labeled September is not necessarily information available at a September decision cutoff. The nested data README reports source publication lags and possible revisions; `retrieved_at` records download time, not publication time. Before forecasting or historical backtesting:

1. Agree the exact September decision timestamp.
2. Obtain publication dates, historical releases/vintages, revision policy and, where possible, `available_at` for each source.
3. Construct every rolling-origin feature set using only values published by that fold’s cutoff.
4. Exclude future-dated steel vendor forecasts unless the intended task explicitly uses forecast-vintage inputs available at the time.
5. Keep post-origin market reports outside model training, validation, feature choice and model comparison.

## 13. Confidentiality and licensing constraints

The local `data/` directory is Git-ignored, but this is not permission to publish, redistribute or use its contents. This public-facing document reports schemas and aggregate QA only; it omits BOM quantities, proprietary competitor prices, supplier identities and financial figures.

Before any modeling or publication, confirm rights and attribution for external sources, licensed L-series, MEPS/CRU/other steel indexes, competitor price references, supplier financial reports, and derived outputs. Keep restricted raw inputs and sensitive derivatives out of public repositories, logs, screenshots and artifacts. The standalone BOM is a client-authorized working source for this project; its contents remain confidential unless disclosure is explicitly approved. Do not bypass file protection.

## 14. Remaining open questions

1. What durable identifier should distinguish material rows from summary rows in each authoritative BOM configuration?
2. Can Schneider Electric clarify the missing authoritative quantities, mass shares and regional price mappings without silently substituting secondary versions?
3. Which materials, purchase basis, geography, currencies and approved baseline spend define the intended partial monetary BOM index?
4. What is the exact September 2026 forecast cutoff, and can historical publication vintages/availability timestamps be supplied for each series?
5. Which RM series and energy features are approved targets/features for each product and market? Are global commodity benchmarks suitable to local purchasing exposure?
6. What source, synthetic-generation process, ML method and quantile interpretation apply to the RMU benchmark, and what uses are permitted?
7. Which competitor configurations, markets, VAT basis, currency and commercial terms make price comparisons valid?
8. What are the approved product mappings and distinctions among RM6/RM AirSeT and SM6/SM AirSeT?
9. Which inflation series is approved, and should it be tested as deflation/reinflation, an exogenous feature, or both?
10. What licenses, retention rules and publication restrictions apply to steel indices, supplier reports, competitor files and the October 2026 market report?

## 15. Recommended next steps for preprocessing, EDA, and baseline forecasting

1. Preserve original inputs and encode the BOM source hierarchy in future derived workflows: standalone workbook for all BOM fields; nested workbook/CSV for RM panel and metadata only.
2. Resolve BOM missing fields, row identity, product/plant mapping, material coverage, and approved cost basis. Never fill authoritative blanks from secondary versions without approval.
3. Establish an as-of data snapshot and publication-vintage ledger for the September 2026 cutoff; quarantine October/November observations, the October market report, and future-dated vendor forecasts from the official experiment.
4. Confirm source licenses, regional relevance, units, currency conversions, index bases, and direct-price versus PPI treatment before constructing features or costs.
5. Perform deeper series-level EDA on approved data: trends, returns, structural breaks, volatility, missingness, outlier review, and seasonality diagnostics. Keep descriptive findings separate from formal validation.
6. Once targets and as-of inputs are approved, prepare LOCF baseline inputs and rolling-origin evaluation folds for December 2026, March 2027 and June 2027. Compare candidate methods later using point-accuracy and uncertainty-calibration metrics; this audit did not train or score models.

### Reproducibility

The audit used read-only CSV parsing, workbook sheet/header and formula/error inspection, keyed panel comparisons, archive-integrity/hash checks, monthly-calendar checks, and descriptive outlier/seasonality screens. No dependencies were installed; no source files were changed, no cleaned or merged dataset was written, no model was trained, and nothing was committed or pushed.
