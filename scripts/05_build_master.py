"""
05_build_master.py — Build the balanced master parish-day panel dataset

PURPOSE:
    Create a complete balanced panel of 64 parishes x all calendar days in the
    study period, then merge in weather (with heat index and heat wave flags),
    OSHA incident counts, and census demographics. This is the single analysis-
    ready dataset that all downstream models and figures draw from.

READS:
    data/processed/weather_with_heat.csv    — Parish-day weather (from Script 02)
    data/processed/parish_day_accidents.csv — Parish-day incident counts (from Script 03)
    data/processed/census_processed.csv     — Parish demographics + outdoor_employment_share (from Script 04)

WRITES:
    data/processed/master_dataset.csv           — Balanced panel (64 parishes x N days)
    data/processed/sample_construction_stats.json — Key counts for report/notebook

FIXES APPLIED (vs. original pipeline):
    1. Panel is built from a clean cross-product of all 64 parishes x all
       calendar days (not derived from weather or OSHA date ranges)
    2. Weather merged on parish_fips + date with LEFT join (no data loss)
    3. OSHA merged on parish_fips + date with LEFT join; missing = 0 incidents
    4. Census merged on parish_fips ONLY (not parish_fips + parish_name,
       which caused LaSalle to fail due to name mismatch)
    5. NEW outdoor_employment_share from ACS data (replaces old outdoor_industry_share)
    6. NO unbounded forward-fill of weather data (old pipeline filled gaps
       of arbitrary length with stale values)
    7. Explicit tracking of data coverage at every stage
    8. Day-of-week and month columns use clear integer coding with documentation
    9. Lagged temperature and heat index variables added for modeling

METHODOLOGY:
    Panel construction:
        - Date range: 2015-01-01 to 2025-07-28 (OSHA data end date)
        - All 64 Louisiana parishes (from census file, which is authoritative)
        - Cross product: 64 parishes x 3,861 days = 247,104 rows
        - Weather: left join, NaN where no station data exists
        - Incidents: left join, 0 where no incidents on that parish-day
        - Census: left join on parish_fips only

    Day-of-week coding:
        - Python's dt.dayofweek: 0=Monday, 1=Tuesday, ..., 6=Sunday
        - This is documented here and in the output for clarity

AUTHOR: Emmanuel Adeniyi
DATE: 2026-03-26
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR = INPUT_DIR

# Study period
STUDY_START = "2015-01-01"
STUDY_END = "2025-07-28"  # Last OSHA event date


# ============================================================================
# BUILD BALANCED PANEL SKELETON
# ============================================================================

def build_panel_skeleton(census_df):
    """
    Create a balanced panel: every parish x every day in the study period.
    Uses parish list from census (authoritative, all 64 parishes).
    """
    print("--- Building Panel Skeleton ---")

    # Date range
    dates = pd.date_range(start=STUDY_START, end=STUDY_END, freq="D")
    n_days = len(dates)
    print(f"  Date range: {STUDY_START} to {STUDY_END}")
    print(f"  Days: {n_days}")

    # Parish list from census
    parishes = census_df[["parish_fips", "parish_name"]].drop_duplicates()
    n_parishes = len(parishes)
    print(f"  Parishes: {n_parishes}")

    # Cross product
    parishes["_key"] = 1
    dates_df = pd.DataFrame({"date": dates, "_key": 1})
    panel = parishes.merge(dates_df, on="_key").drop(columns="_key")

    expected = n_parishes * n_days
    print(f"  Expected rows: {n_parishes} x {n_days} = {expected:,}")
    print(f"  Actual rows: {len(panel):,}")
    assert len(panel) == expected, f"Panel size mismatch: {len(panel)} != {expected}"

    return panel


# ============================================================================
# MERGE WEATHER
# ============================================================================

def merge_weather(panel, weather_path):
    """
    Left-join weather data onto the panel. Parish-days without weather
    station coverage will have NaN for all weather columns.
    NO forward-fill is applied (old pipeline filled arbitrarily long gaps).
    """
    print("\n--- Merging Weather ---")

    weather = pd.read_csv(weather_path, parse_dates=["date"])
    print(f"  Weather rows loaded: {len(weather):,}")
    print(f"  Weather parishes: {weather['parish_fips'].nunique()}")

    # Select weather columns to merge (drop parish_name to avoid conflicts)
    weather_cols = [
        "parish_fips", "date",
        "tmax_f", "tmin_f", "dewpoint_c",
        "rh_final", "humidity_source",
        "heat_index_f", "hw_threshold_f",
        "above_threshold", "heat_wave_flag",
        "n_stations"
    ]
    if "assignment_type" in weather.columns:
        weather_cols.append("assignment_type")
    # Include cumulative heat stress and HW lag variables if present
    for extra_col in ["cumulative_heat_3d", "cumulative_heat_5d",
                      "heat_wave_flag_lag1", "heat_wave_flag_lag2"]:
        if extra_col in weather.columns:
            weather_cols.append(extra_col)
    weather_merge = weather[weather_cols].copy()

    # Merge
    panel = panel.merge(weather_merge, on=["parish_fips", "date"], how="left")

    # Fill incident-related flags with 0 where weather is missing
    # (no weather data = we can't determine heat wave status)
    # Leave heat_wave_flag as NaN where no weather, not 0
    # This is important: 0 means "not a heat wave day", NaN means "unknown"

    weather_coverage = panel["tmax_f"].notna().sum()
    pct = weather_coverage / len(panel) * 100
    print(f"  Parish-days with weather: {weather_coverage:,} ({pct:.1f}%)")
    print(f"  Parish-days without weather: {len(panel) - weather_coverage:,} ({100-pct:.1f}%)")

    # Parishes with no weather at all
    parish_has_weather = panel.groupby("parish_fips")["tmax_f"].apply(lambda x: x.notna().any())
    no_weather_parishes = parish_has_weather[~parish_has_weather].index.tolist()
    print(f"  Parishes with NO weather data: {len(no_weather_parishes)}")

    return panel


# ============================================================================
# MERGE OSHA INCIDENTS
# ============================================================================

def merge_incidents(panel, incidents_path):
    """
    Left-join parish-day incident counts onto the panel.
    Parish-days with no incidents get 0 for all count columns.
    """
    print("\n--- Merging OSHA Incidents ---")

    incidents = pd.read_csv(incidents_path, parse_dates=["date"])
    print(f"  Incident parish-day rows loaded: {len(incidents):,}")
    print(f"  Total incidents: {incidents['total_incidents'].sum():,}")
    print(f"  Heat-related incidents: {incidents['heat_related'].sum():,}")

    # Select columns (drop parish_name to avoid conflicts)
    inc_cols = [
        "parish_fips", "date",
        "total_incidents", "hospitalizations", "amputations",
        "loss_of_eye", "heat_related"
    ]
    if "outdoor_industry_incidents" in incidents.columns:
        inc_cols.append("outdoor_industry_incidents")
    inc_merge = incidents[inc_cols].copy()
    inc_merge["parish_fips"] = inc_merge["parish_fips"].astype(int)

    # Merge
    panel = panel.merge(inc_merge, on=["parish_fips", "date"], how="left")

    # Fill missing incident counts with 0 (no incidents on that day)
    count_cols = ["total_incidents", "hospitalizations", "amputations",
                  "loss_of_eye", "heat_related"]
    if "outdoor_industry_incidents" in panel.columns:
        count_cols.append("outdoor_industry_incidents")
    for col in count_cols:
        panel[col] = panel[col].fillna(0).astype(int)

    total_in_panel = panel["total_incidents"].sum()
    print(f"  Total incidents in panel: {total_in_panel:,}")
    print(f"  Heat-related in panel: {panel['heat_related'].sum():,}")
    print(f"  Parish-days with >= 1 incident: {(panel['total_incidents'] > 0).sum():,}")

    return panel


# ============================================================================
# MERGE CENSUS
# ============================================================================

def merge_census(panel, census_path):
    """
    Left-join census demographics onto the panel using parish_fips ONLY.
    (Old pipeline joined on parish_fips + parish_name, which failed for LaSalle.)
    Now includes outdoor_employment_share from ACS data.
    """
    print("\n--- Merging Census ---")

    census = pd.read_csv(census_path)
    print(f"  Census parishes: {len(census)}")

    # Select columns (parish_name already in panel from skeleton)
    census_cols = [
        "parish_fips",
        "population_total", "income_median",
        "housing_total", "housing_occupied",
        "population_16plus", "mean_travel_time",
        "labor_force", "unemployed", "unemployment_rate",
        "is_urban", "urban_rural"
    ]
    # Add outdoor_employment_share if available (new in 2026-03-26 rebuild)
    if "outdoor_employment_share" in census.columns:
        census_cols.append("outdoor_employment_share")

    census_merge = census[census_cols].copy()

    # Merge on FIPS only
    panel = panel.merge(census_merge, on="parish_fips", how="left")

    # Verify no nulls
    pop_nulls = panel["population_total"].isna().sum()
    inc_nulls = panel["income_median"].isna().sum()
    print(f"  Population nulls: {pop_nulls}")
    print(f"  Income nulls: {inc_nulls}")

    if "outdoor_employment_share" in panel.columns:
        oe_nulls = panel["outdoor_employment_share"].isna().sum()
        print(f"  Outdoor employment share nulls: {oe_nulls}")
        oe_mean = panel["outdoor_employment_share"].mean()
        print(f"  Mean outdoor employment share: {oe_mean:.4f}")

    if pop_nulls > 0:
        missing = panel[panel["population_total"].isna()]["parish_name"].unique()
        print(f"  WARNING: Parishes with missing census data: {missing}")

    return panel


# ============================================================================
# ADD TEMPORAL FEATURES
# ============================================================================

def add_temporal_features(panel):
    """
    Add day-of-week, month, and year columns.
    Day-of-week: 0=Monday, 1=Tuesday, ..., 6=Sunday (Python convention).
    """
    print("\n--- Adding Temporal Features ---")

    panel["day_of_week"] = panel["date"].dt.dayofweek  # 0=Mon, 6=Sun
    panel["month"] = panel["date"].dt.month
    panel["year"] = panel["date"].dt.year

    # Verify
    dow_counts = panel["day_of_week"].value_counts().sort_index()
    print(f"  Day-of-week distribution (0=Mon, 6=Sun):")
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for dow, count in dow_counts.items():
        print(f"    {dow} ({dow_labels[dow]}): {count:,}")

    return panel


def add_lagged_variables(panel):
    """
    Add lagged temperature and heat index variables for each parish.
    Creates: tmax_f_lag1, tmax_f_lag2, heat_index_lag1, heat_index_lag2

    Lags are computed within-parish (no cross-parish spillover) and are set to
    NaN at the beginning of each parish's time series.
    """
    print("\n--- Adding Lagged Variables ---")

    # Group by parish and create lags (within each parish)
    for parish_fips, group in panel.groupby("parish_fips"):
        parish_mask = panel["parish_fips"] == parish_fips

        # Lag 1 for tmax_f
        if "tmax_f" in panel.columns:
            panel.loc[parish_mask, "tmax_f_lag1"] = \
                panel.loc[parish_mask, "tmax_f"].shift(1)

        # Lag 2 for tmax_f
        if "tmax_f" in panel.columns:
            panel.loc[parish_mask, "tmax_f_lag2"] = \
                panel.loc[parish_mask, "tmax_f"].shift(2)

        # Lag 1 for heat_index_f
        if "heat_index_f" in panel.columns:
            panel.loc[parish_mask, "heat_index_lag1"] = \
                panel.loc[parish_mask, "heat_index_f"].shift(1)

        # Lag 2 for heat_index_f
        if "heat_index_f" in panel.columns:
            panel.loc[parish_mask, "heat_index_lag2"] = \
                panel.loc[parish_mask, "heat_index_f"].shift(2)

    # Report on coverage
    if "tmax_f_lag1" in panel.columns:
        lag1_coverage = panel["tmax_f_lag1"].notna().sum()
        print(f"  tmax_f_lag1: {lag1_coverage:,} non-null ({lag1_coverage/len(panel)*100:.1f}%)")
    if "tmax_f_lag2" in panel.columns:
        lag2_coverage = panel["tmax_f_lag2"].notna().sum()
        print(f"  tmax_f_lag2: {lag2_coverage:,} non-null ({lag2_coverage/len(panel)*100:.1f}%)")
    if "heat_index_lag1" in panel.columns:
        hi_lag1 = panel["heat_index_lag1"].notna().sum()
        print(f"  heat_index_lag1: {hi_lag1:,} non-null ({hi_lag1/len(panel)*100:.1f}%)")
    if "heat_index_lag2" in panel.columns:
        hi_lag2 = panel["heat_index_lag2"].notna().sum()
        print(f"  heat_index_lag2: {hi_lag2:,} non-null ({hi_lag2/len(panel)*100:.1f}%)")

    return panel


# ============================================================================
# COMPUTE SAMPLE CONSTRUCTION STATS
# ============================================================================

def compute_stats(panel):
    """
    Compute and save key statistics for the report and notebook.
    These are the authoritative numbers — all other documents should reference these.
    """
    print("\n--- Sample Construction Statistics ---")

    stats = {}

    # Full panel
    stats["full_panel_rows"] = len(panel)
    stats["full_panel_parishes"] = panel["parish_fips"].nunique()
    stats["full_panel_days"] = panel["date"].nunique()
    stats["study_start"] = str(panel["date"].min().date())
    stats["study_end"] = str(panel["date"].max().date())

    # Weather coverage
    has_weather = panel["tmax_f"].notna()
    stats["weather_linked_rows"] = int(has_weather.sum())
    stats["weather_coverage_pct"] = round(has_weather.mean() * 100, 1)
    stats["weather_parishes"] = int(panel.loc[has_weather, "parish_fips"].nunique())

    # Complete case (for regression: need tmax, population, income)
    complete = panel.dropna(subset=["tmax_f", "population_total", "income_median"])
    stats["complete_case_rows"] = len(complete)
    stats["complete_case_parishes"] = int(complete["parish_fips"].nunique())
    stats["complete_case_incidents"] = int(complete["total_incidents"].sum())
    stats["complete_case_heat_related"] = int(complete["heat_related"].sum())
    stats["complete_case_pct"] = round(len(complete) / len(panel) * 100, 1)

    # Heat wave stats in complete case
    cc_hw = complete[complete["heat_wave_flag"] == 1]
    stats["hw_parish_days_complete"] = len(cc_hw)
    stats["hw_parishes_in_complete"] = int(cc_hw["parish_fips"].nunique()) if len(cc_hw) > 0 else 0
    stats["incidents_during_hw_complete"] = int(cc_hw["total_incidents"].sum()) if len(cc_hw) > 0 else 0
    stats["heat_incidents_during_hw_complete"] = int(cc_hw["heat_related"].sum()) if len(cc_hw) > 0 else 0

    # Total OSHA stats
    stats["total_osha_incidents"] = int(panel["total_incidents"].sum())
    stats["total_heat_related"] = int(panel["heat_related"].sum())

    # Print summary
    print(f"  Full panel: {stats['full_panel_rows']:,} rows "
          f"({stats['full_panel_parishes']} parishes x {stats['full_panel_days']} days)")
    print(f"  Weather-linked: {stats['weather_linked_rows']:,} ({stats['weather_coverage_pct']}%)")
    print(f"  Complete case: {stats['complete_case_rows']:,} "
          f"({stats['complete_case_parishes']} parishes, {stats['complete_case_pct']}%)")
    print(f"  Incidents in complete case: {stats['complete_case_incidents']:,} "
          f"({stats['complete_case_heat_related']} heat-related)")
    print(f"  HW parish-days in complete case: {stats['hw_parish_days_complete']:,} "
          f"({stats['hw_parishes_in_complete']} parishes)")
    print(f"  Incidents during HW (complete case): {stats['incidents_during_hw_complete']}")
    print(f"  Total OSHA incidents: {stats['total_osha_incidents']:,}")
    print(f"  Total heat-related: {stats['total_heat_related']}")

    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("STEP 05: BUILD MASTER PANEL DATASET")
    print("=" * 70)
    start_time = datetime.now()

    # Load census first (need parish list for skeleton)
    census_path = os.path.join(INPUT_DIR, "census_processed.csv")
    census = pd.read_csv(census_path)

    # Build skeleton
    panel = build_panel_skeleton(census)

    # Merge weather
    weather_path = os.path.join(INPUT_DIR, "weather_with_heat.csv")
    panel = merge_weather(panel, weather_path)

    # Merge incidents
    incidents_path = os.path.join(INPUT_DIR, "parish_day_accidents.csv")
    panel = merge_incidents(panel, incidents_path)

    # Merge census
    panel = merge_census(panel, census_path)

    # Merge industry profile (parish-level outdoor share)
    industry_path = os.path.join(INPUT_DIR, "parish_industry_profile.csv")
    if os.path.exists(industry_path):
        print("\n--- Merging Industry Profile ---")
        industry = pd.read_csv(industry_path)
        industry_cols = ["parish_fips", "outdoor_industry_share", "dominant_sector"]
        panel = panel.merge(industry[industry_cols], on="parish_fips", how="left")
        # Parishes with no incidents get 0 outdoor share
        panel["outdoor_industry_share"] = panel["outdoor_industry_share"].fillna(0.0)
        panel["dominant_sector"] = panel["dominant_sector"].fillna("Unknown")
        print(f"  Merged industry profile for {industry['parish_fips'].nunique()} parishes")
        print(f"  Mean outdoor share: {panel['outdoor_industry_share'].mean():.3f}")

    # Add temporal features
    panel = add_temporal_features(panel)

    # Add lagged variables (for modeling)
    panel = add_lagged_variables(panel)

    # Sort for clean output
    panel = panel.sort_values(["parish_fips", "date"]).reset_index(drop=True)

    # Compute and save stats
    stats = compute_stats(panel)

    stats_path = os.path.join(OUTPUT_DIR, "sample_construction_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {stats_path}")

    # Save master dataset
    output_path = os.path.join(OUTPUT_DIR, "master_dataset.csv")
    panel.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"  Rows: {len(panel):,}")
    print(f"  Columns: {list(panel.columns)}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
