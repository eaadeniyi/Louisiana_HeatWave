"""
02_compute_heat_index.py — Compute humidity (ERA5 + GHCN), heat index, and identify heat waves

PURPOSE:
    Starting from the parish-day weather data produced by 01_parse_weather.py,
    this script merges in ERA5/Open-Meteo humidity data, computes relative humidity
    (from ERA5 primarily, with GHCN dew point as cross-check), calculates the NWS
    heat index from TMAX, and identifies heat wave events using a per-parish threshold
    based on the 1981-2010 NOAA climatological normal baseline.

READS:
    data/processed/weather_daily.csv              — Parish-day weather (from Script 01)
    data/processed/weather_baseline_1981_2010.csv — Historical Jul-Aug baseline (from Script 01)
    data/raw/era5/parish_daily_humidity.csv       — ERA5/Open-Meteo parish-day humidity (100% coverage)

WRITES:
    data/processed/weather_with_heat.csv  — Same data + humidity_source, rh_final,
                                            heat_index_f, hw_threshold_f, heat_wave_flag

CRITICAL CHANGE (NEW in 2026-03-26 rebuild):
    1. **ERA5 humidity is now PRIMARY source** (100% coverage for 64 parishes)
    2. GHCN dew point used as secondary validation source
    3. humidity_source column now shows "era5" for most days, not "assumed_75pct"

FIXES APPLIED (vs. original pipeline):
    1. Heat index computed from TMAX (daytime high), not TMIN (overnight low)
    2. ERA5 RH data used for ALL parish-days (100% coverage, not sparse GHCN)
    3. GHCN dew point data used as cross-check where available
    4. 75% constant only used as absolute fallback (should be rare/never)
    5. Each parish-day is tagged with its humidity source for transparency
    6. Full NWS Rothfusz regression with BOTH adjustments:
       - Low-humidity subtraction (RH < 13%, 80 < T < 112)
       - High-humidity addition (RH > 85%, 80 < T < 87)
    7. Heat wave threshold computed from 1981-2010 July-August baseline
       (NOAA 30-year climatological normal, NOT the study period)
    8. Per-parish thresholds (not statewide) where sufficient data exists
    9. Heat wave flag marks ALL days in a >=2-day consecutive sequence

METHODOLOGY:
    Heat Index:
        - NWS Rothfusz regression (standard since 1990)
        - Input: daily maximum temperature (TMAX) and relative humidity
        - For TMAX < 80F, uses the simpler Steadman formula
        - Capped at 140F (instrument/physiological limit)

    Humidity (NEW PRIORITY CASCADE):
        - Priority 1: ERA5/Open-Meteo relative_humidity_pct (100% coverage)
        - Priority 2: Compute RH from GHCN dew point via Magnus formula
          RH = 100 * exp((17.625*Td)/(243.04+Td)) / exp((17.625*T)/(243.04+T))
          where T = TMAX in Celsius, Td = dew point in Celsius
        - Priority 3: Use station-reported RHAV (daily average RH, if ERA5 unavailable)
        - Priority 4: Constant 75% (Louisiana average; flagged as assumed)

    Heat Waves:
        - Threshold: 85th percentile of July-August daily heat index per parish
          (computed from 1981-2010 historical baseline; parishes with < 30
          Jul-Aug observations use the statewide 85th percentile as fallback)
        - Definition: >= 2 consecutive days where heat index exceeds parish threshold
        - ALL days in the sequence are flagged (including day 1)

AUTHOR: Emmanuel Adeniyi
DATE: 2026-03-26
"""

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR = INPUT_DIR

HEAT_INDEX_CAP = 140.0  # Maximum plausible heat index (F)

# Minimum number of July-August observations needed for a parish-specific
# heat wave threshold. Below this, we fall back to the statewide threshold.
MIN_BASELINE_OBS = 30

# Heat wave definition parameters
HW_PERCENTILE = 85  # Percentile for threshold
HW_MIN_CONSECUTIVE_DAYS = 2  # Minimum consecutive days above threshold


# ============================================================================
# HUMIDITY COMPUTATION
# ============================================================================

def compute_rh_from_dewpoint(tmax_f, dewpoint_c):
    """
    Compute relative humidity from daily max temperature and dew point
    using the Magnus formula.

    Parameters:
        tmax_f: Daily maximum temperature in Fahrenheit
        dewpoint_c: Dew point temperature in Celsius

    Returns:
        Relative humidity as a percentage (0-100), or NaN if inputs are invalid.

    The Magnus formula:
        RH = 100 * exp((b * Td) / (c + Td)) / exp((b * T) / (c + T))
    where b = 17.625, c = 243.04 (August-Roche-Magnus approximation)

    We use TMAX (converted to Celsius) as T because:
    - Heat index is computed from TMAX
    - RH at the time of maximum temperature is what matters for heat stress
    - Using TMAX with a morning dew point gives a conservative (lower) RH estimate,
      which is appropriate since dew point typically drops slightly as temperature rises
    """
    if pd.isna(tmax_f) or pd.isna(dewpoint_c):
        return np.nan

    # Convert TMAX to Celsius
    t_c = (tmax_f - 32.0) * 5.0 / 9.0

    # Magnus formula constants
    b = 17.625
    c = 243.04

    try:
        rh = 100.0 * math.exp((b * dewpoint_c) / (c + dewpoint_c)) / \
             math.exp((b * t_c) / (c + t_c))
    except (OverflowError, ZeroDivisionError):
        return np.nan

    # Clamp to valid range
    rh = max(0.0, min(100.0, rh))
    return round(rh, 1)


def assign_humidity(df, era5_df=None):
    """
    Assign relative humidity to each parish-day using a priority cascade:
    1. Use ERA5/Open-Meteo RH (100% coverage, primary source)
    2. Compute from GHCN dew point (validation/fallback)
    3. Use station-reported RHAV
    4. Fall back to constant 75% (should be rare/never with ERA5)

    Parameters:
        df: Study-period weather data from script 01
        era5_df: ERA5 humidity data (if provided, merged and used as primary)

    Adds columns: rh_final, humidity_source
    """
    print("--- Humidity Assignment ---")

    # Priority 0: Merge ERA5 data if provided
    if era5_df is not None and len(era5_df) > 0:
        # Merge ERA5 RH onto the weather data
        df = df.merge(
            era5_df[["parish_fips", "date", "relative_humidity_pct"]],
            on=["parish_fips", "date"],
            how="left"
        )
        print(f"  Merged ERA5 data: {era5_df[['parish_fips', 'date', 'relative_humidity_pct']].shape[0]:,} parish-days")
    else:
        df["relative_humidity_pct"] = np.nan

    # Priority 1: Compute from dew point (for comparison/validation)
    has_dp = df["dewpoint_c"].notna() & df["tmax_f"].notna()
    df.loc[has_dp, "rh_from_dewpoint"] = df.loc[has_dp].apply(
        lambda row: compute_rh_from_dewpoint(row["tmax_f"], row["dewpoint_c"]),
        axis=1
    )

    # Build final RH column with priority cascade
    df["rh_final"] = np.nan
    df["humidity_source"] = "none"

    # Priority 1: ERA5 RH (now primary)
    mask_era5 = df["relative_humidity_pct"].notna()
    df.loc[mask_era5, "rh_final"] = df.loc[mask_era5, "relative_humidity_pct"]
    df.loc[mask_era5, "humidity_source"] = "era5"

    # Priority 2: dew point derived (where ERA5 not available)
    mask_dp = (~mask_era5) & df["rh_from_dewpoint"].notna()
    df.loc[mask_dp, "rh_final"] = df.loc[mask_dp, "rh_from_dewpoint"]
    df.loc[mask_dp, "humidity_source"] = "ghcn_dewpoint"

    # Priority 3: station-reported RHAV (where neither ERA5 nor dew point available)
    mask_rhav = (~mask_era5) & (~mask_dp) & df["rh_pct"].notna()
    df.loc[mask_rhav, "rh_final"] = df.loc[mask_rhav, "rh_pct"]
    df.loc[mask_rhav, "humidity_source"] = "ghcn_rhav"

    # Priority 4: constant 75% (only where nothing else available)
    mask_const = df["rh_final"].isna()
    df.loc[mask_const, "rh_final"] = 75.0
    df.loc[mask_const, "humidity_source"] = "assumed_75pct"

    # Drop intermediate columns
    df.drop(columns=["rh_from_dewpoint", "relative_humidity_pct"], inplace=True, errors="ignore")

    # Report
    source_counts = df["humidity_source"].value_counts()
    total = len(df)
    print(f"  Total parish-days: {total:,}")
    for source, count in sorted(source_counts.items()):
        pct = count/total*100
        print(f"  {source}: {count:,} ({pct:.1f}%)")

    # Summary stats on final RH
    print(f"  Final RH range: {df['rh_final'].min():.0f}-{df['rh_final'].max():.0f}%")
    print(f"  Final RH mean: {df['rh_final'].mean():.1f}%")

    return df


# ============================================================================
# HEAT INDEX COMPUTATION (NWS Rothfusz Regression)
# ============================================================================

def heat_index_nws(T, RH):
    """
    Compute the NWS heat index using the full Rothfusz regression
    with both adjustment terms.

    Parameters:
        T:  Air temperature in Fahrenheit (should be TMAX for this study)
        RH: Relative humidity in percent (0-100)

    Returns:
        Heat index in Fahrenheit, or NaN if inputs are invalid.

    Algorithm (from NWS Technical Attachment SR 90-23):
        1. If T < 80F: use simple Steadman formula
        2. If T >= 80F: use Rothfusz regression
        3. Apply low-humidity adjustment if RH < 13% and 80 < T < 112
        4. Apply high-humidity adjustment if RH > 85% and 80 < T < 87
        5. Cap at HEAT_INDEX_CAP (140F)

    Reference: Rothfusz, L.P. (1990). The heat index "equation"
    (or, more than you ever wanted to know about heat index).
    NWS Southern Region Headquarters, Fort Worth, TX.
    """
    if pd.isna(T) or pd.isna(RH):
        return np.nan

    # Step 1: Simple formula for T < 80F
    HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))

    if HI < 80.0:
        return round(HI, 2)

    # Step 2: Full Rothfusz regression
    HI = (-42.379
          + 2.04901523 * T
          + 10.14333127 * RH
          - 0.22475541 * T * RH
          - 0.00683783 * T * T
          - 0.05481717 * RH * RH
          + 0.00122874 * T * T * RH
          + 0.00085282 * T * RH * RH
          - 0.00000199 * T * T * RH * RH)

    # Step 3: Low-humidity adjustment
    # When RH < 13% and 80F < T < 112F, subtract a correction
    if RH < 13.0 and 80.0 < T < 112.0:
        adjustment = ((13.0 - RH) / 4.0) * math.sqrt((17.0 - abs(T - 95.0)) / 17.0)
        HI -= adjustment

    # Step 4: High-humidity adjustment
    # When RH > 85% and 80F < T < 87F, add a correction
    # This was MISSING in the original pipeline
    if RH > 85.0 and 80.0 < T < 87.0:
        adjustment = ((RH - 85.0) / 10.0) * ((87.0 - T) / 5.0)
        HI += adjustment

    # Step 5: Cap at maximum
    HI = min(HI, HEAT_INDEX_CAP)

    return round(HI, 2)


def compute_heat_index(df):
    """
    Compute heat index for all parish-days using TMAX and final RH.
    """
    print("\n--- Heat Index Computation ---")

    # Compute heat index from TMAX (not TMIN like the old pipeline)
    df["heat_index_f"] = df.apply(
        lambda row: heat_index_nws(row["tmax_f"], row["rh_final"]),
        axis=1
    )

    valid = df["heat_index_f"].notna()
    print(f"  Heat index computed: {valid.sum():,} parish-days")
    print(f"  Heat index range: {df.loc[valid, 'heat_index_f'].min():.1f} "
          f"to {df.loc[valid, 'heat_index_f'].max():.1f} F")
    print(f"  Heat index mean: {df.loc[valid, 'heat_index_f'].mean():.1f} F")

    # How many exceed common danger thresholds
    above_105 = (df["heat_index_f"] > 105).sum()
    above_110 = (df["heat_index_f"] > 110).sum()
    print(f"  Parish-days with HI > 105F (danger): {above_105:,}")
    print(f"  Parish-days with HI > 110F (extreme danger): {above_110:,}")

    return df


# ============================================================================
# HEAT WAVE IDENTIFICATION
# ============================================================================

def identify_heat_waves(df, baseline_df=None):
    """
    Identify heat wave events using per-parish thresholds.

    Steps:
    1. Compute the 85th percentile of July-August heat index for each parish
       using the 1981-2010 historical baseline (or study-period fallback).
    2. For parishes with < MIN_BASELINE_OBS July-August days, use the statewide
       85th percentile as fallback.
    3. Flag days where heat index exceeds the parish threshold.
    4. Mark heat waves as sequences of >= HW_MIN_CONSECUTIVE_DAYS consecutive
       above-threshold days. ALL days in the sequence are flagged.

    Parameters:
        df: Study-period weather data with heat_index_f column
        baseline_df: Historical 1981-2010 Jul-Aug weather data (with heat_index_f).
                     If None, falls back to study-period Jul-Aug data.

    Adds columns: hw_threshold_f, above_threshold, heat_wave_flag
    """
    print("\n--- Heat Wave Identification ---")

    # Ensure date is datetime and sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["parish_fips", "date"]).reset_index(drop=True)

    # Step 1: Compute baseline thresholds from 1981-2010 historical data
    if baseline_df is not None and len(baseline_df) > 0:
        baseline = baseline_df[baseline_df["heat_index_f"].notna()].copy()
        print(f"  Using 1981-2010 historical baseline: {len(baseline):,} Jul-Aug parish-days")
    else:
        # Fallback to study-period (not recommended — circular)
        print("  WARNING: No historical baseline provided, falling back to study-period")
        baseline_mask = (
            df["date"].dt.month.isin([7, 8]) &
            (df["date"].dt.year >= 2015) &
            (df["date"].dt.year <= 2024) &
            df["heat_index_f"].notna()
        )
        baseline = df[baseline_mask].copy()
        print(f"  Baseline observations (Jul-Aug 2015-2024): {len(baseline):,}")

    # Per-parish threshold
    parish_thresholds = baseline.groupby("parish_fips")["heat_index_f"].quantile(
        HW_PERCENTILE / 100.0
    )

    # Statewide fallback threshold
    statewide_threshold = baseline["heat_index_f"].quantile(HW_PERCENTILE / 100.0)
    print(f"  Statewide {HW_PERCENTILE}th percentile threshold: {statewide_threshold:.1f} F")

    # Count baseline obs per parish
    baseline_counts = baseline.groupby("parish_fips").size()

    # Apply parish-specific or statewide threshold
    threshold_map = {}
    for fips in df["parish_fips"].unique():
        n_obs = baseline_counts.get(fips, 0)
        if n_obs >= MIN_BASELINE_OBS and fips in parish_thresholds.index:
            threshold_map[fips] = round(parish_thresholds[fips], 2)
        else:
            threshold_map[fips] = round(statewide_threshold, 2)

    df["hw_threshold_f"] = df["parish_fips"].map(threshold_map)

    # Report thresholds
    unique_thresholds = df[["parish_fips", "parish_name", "hw_threshold_f"]].drop_duplicates()
    n_parish_specific = (unique_thresholds["hw_threshold_f"] != round(statewide_threshold, 2)).sum()
    print(f"  Parishes with own threshold: {n_parish_specific}")
    print(f"  Parishes using statewide fallback: {len(unique_thresholds) - n_parish_specific}")
    print(f"  Threshold range: {unique_thresholds['hw_threshold_f'].min():.1f} "
          f"to {unique_thresholds['hw_threshold_f'].max():.1f} F")

    # Step 2: Flag days above threshold
    df["above_threshold"] = (
        df["heat_index_f"].notna() &
        (df["heat_index_f"] > df["hw_threshold_f"])
    ).astype(int)

    total_above = df["above_threshold"].sum()
    print(f"  Parish-days above threshold: {total_above:,}")

    # Step 3: Identify consecutive sequences and mark heat waves
    df["heat_wave_flag"] = 0

    for fips, group in df.groupby("parish_fips"):
        idx = group.index.tolist()
        dates = group["date"].tolist()
        above = group["above_threshold"].tolist()

        # Track consecutive above-threshold sequences
        current_streak_start = None
        current_streak_indices = []

        for i, (row_idx, date, is_above) in enumerate(zip(idx, dates, above)):
            if is_above:
                if current_streak_start is None:
                    current_streak_start = i
                    current_streak_indices = [row_idx]
                else:
                    # Check if this is actually consecutive (next day)
                    prev_date = dates[i - 1] if i > 0 else None
                    if prev_date is not None and (date - prev_date).days == 1:
                        current_streak_indices.append(row_idx)
                    else:
                        # Gap in dates — end previous streak, start new one
                        if len(current_streak_indices) >= HW_MIN_CONSECUTIVE_DAYS:
                            df.loc[current_streak_indices, "heat_wave_flag"] = 1
                        current_streak_start = i
                        current_streak_indices = [row_idx]
            else:
                # End of streak
                if current_streak_indices and len(current_streak_indices) >= HW_MIN_CONSECUTIVE_DAYS:
                    df.loc[current_streak_indices, "heat_wave_flag"] = 1
                current_streak_start = None
                current_streak_indices = []

        # Handle streak that extends to end of data
        if current_streak_indices and len(current_streak_indices) >= HW_MIN_CONSECUTIVE_DAYS:
            df.loc[current_streak_indices, "heat_wave_flag"] = 1

    # Report
    hw_days = df["heat_wave_flag"].sum()
    hw_parishes = df[df["heat_wave_flag"] == 1]["parish_fips"].nunique()
    print(f"\n  Heat wave parish-days: {hw_days:,}")
    print(f"  Parishes with at least one HW day: {hw_parishes}")

    if hw_days > 0:
        hw_by_parish = df[df["heat_wave_flag"] == 1].groupby("parish_name").size().sort_values(ascending=False)
        print(f"  Heat wave days by parish:")
        for parish, count in hw_by_parish.items():
            print(f"    {parish}: {count} days")

    # Year distribution
    if hw_days > 0:
        hw_by_year = df[df["heat_wave_flag"] == 1].groupby(df["date"].dt.year).size()
        print(f"  Heat wave days by year:")
        for year, count in hw_by_year.items():
            print(f"    {year}: {count} days")

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("STEP 02: COMPUTE HEAT INDEX AND IDENTIFY HEAT WAVES")
    print("=" * 70)
    start_time = datetime.now()

    # Load weather data from Step 01
    input_path = os.path.join(INPUT_DIR, "weather_daily.csv")
    df = pd.read_csv(input_path, parse_dates=["date"])
    print(f"Loaded: {input_path}")
    print(f"  Rows: {len(df):,}, Parishes: {df['parish_fips'].nunique()}")

    # Load 1981-2010 historical baseline from Step 01
    baseline_path = os.path.join(INPUT_DIR, "weather_baseline_1981_2010.csv")
    baseline_df = pd.read_csv(baseline_path, parse_dates=["date"])
    print(f"Loaded baseline: {baseline_path}")
    print(f"  Rows: {len(baseline_df):,}, Parishes: {baseline_df['parish_fips'].nunique()}")

    # Load ERA5/Open-Meteo humidity data (NEW in 2026-03-26 rebuild)
    raw_era5_dir = os.path.join(os.path.dirname(INPUT_DIR), "raw", "era5")
    era5_path = os.path.join(raw_era5_dir, "parish_daily_humidity.csv")
    era5_df = None
    if os.path.exists(era5_path):
        era5_df = pd.read_csv(era5_path, parse_dates=["date"])
        # Convert ERA5 FIPS (1-64) to full Louisiana FIPS (22001-22127)
        era5_df["parish_fips"] = era5_df["parish_fips"] + 22000
        print(f"Loaded ERA5 humidity: {era5_path}")
        print(f"  Rows: {len(era5_df):,}, Parishes: {era5_df['parish_fips'].nunique()}")
    else:
        print(f"WARNING: ERA5 humidity file not found at {era5_path}")
        print("  Falling back to GHCN dew point and assumed RH")

    # Step 1: Assign humidity (study period) — now with ERA5 as primary
    df = assign_humidity(df, era5_df=era5_df)

    # Step 2: Compute heat index (study period)
    df = compute_heat_index(df)

    # Step 2b: Compute humidity and heat index for baseline too (without ERA5, use GHCN)
    print("\n--- Processing Historical Baseline ---")
    baseline_df = assign_humidity(baseline_df, era5_df=None)
    baseline_df = compute_heat_index(baseline_df)

    # Step 3: Identify heat waves using 1981-2010 baseline thresholds
    df = identify_heat_waves(df, baseline_df=baseline_df)

    # Step 4: Cumulative heat stress variables
    print("\n--- Cumulative Heat Stress Variables ---")
    df = df.sort_values(["parish_fips", "date"])

    # Flag days where TMAX exceeds parish-specific 90th percentile
    tmax_p90 = df.groupby("parish_fips")["tmax_f"].quantile(0.90)
    df["tmax_p90_threshold"] = df["parish_fips"].map(tmax_p90)
    df["exceeds_tmax_p90"] = (
        df["tmax_f"].notna() & (df["tmax_f"] > df["tmax_p90_threshold"])
    ).astype(float)

    # Rolling counts within each parish
    df["cumulative_heat_3d"] = (
        df.groupby("parish_fips", group_keys=False)["exceeds_tmax_p90"]
        .rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    df["cumulative_heat_5d"] = (
        df.groupby("parish_fips", group_keys=False)["exceeds_tmax_p90"]
        .rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    valid_cum = df["cumulative_heat_3d"].notna()
    print(f"  Mean 3-day stress score: {df.loc[valid_cum, 'cumulative_heat_3d'].mean():.2f}")
    print(f"  Mean 5-day stress score: {df.loc[valid_cum, 'cumulative_heat_5d'].mean():.2f}")
    print(f"  Share with 3/3 days above P90: "
          f"{(df['cumulative_heat_3d'] == 3).mean():.3f}")

    # Also add heat_wave_flag lags (needed for lag models in script 06)
    df["heat_wave_flag_lag1"] = (
        df.groupby("parish_fips", group_keys=False)["heat_wave_flag"]
        .shift(1)
    )
    df["heat_wave_flag_lag2"] = (
        df.groupby("parish_fips", group_keys=False)["heat_wave_flag"]
        .shift(2)
    )
    print(f"  heat_wave_flag_lag1: {df['heat_wave_flag_lag1'].notna().sum():,} non-null")
    print(f"  heat_wave_flag_lag2: {df['heat_wave_flag_lag2'].notna().sum():,} non-null")

    # Step 5: Select output columns and save
    output_cols = [
        "parish_fips", "parish_name", "date",
        "tmax_f", "tmin_f", "dewpoint_c",
        "rh_final", "humidity_source",
        "heat_index_f", "hw_threshold_f",
        "above_threshold", "heat_wave_flag",
        "n_stations",
        "cumulative_heat_3d", "cumulative_heat_5d",
        "heat_wave_flag_lag1", "heat_wave_flag_lag2",
    ]
    # Include assignment_type if present
    if "assignment_type" in df.columns:
        output_cols.append("assignment_type")
    output_df = df[output_cols].copy()

    output_path = os.path.join(OUTPUT_DIR, "weather_with_heat.csv")
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(output_df):,}")
    print(f"  Columns: {list(output_df.columns)}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
