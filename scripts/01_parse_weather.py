"""
01_parse_weather.py — Parse NOAA GHCN-Daily weather data for Louisiana

PURPOSE:
    Read raw .dly fixed-width files for ALL 158 weather stations, extract daily
    TMAX, TMIN, ADPT (dew point), and RHAV (relative humidity), convert units,
    apply quality filters, assign stations to parishes, and aggregate to
    parish-day level.

    Also parses 1981-2010 historical data for NOAA climatological normal baseline,
    and performs nearest-station interpolation for parishes without direct station coverage.

READS:
    data/raw/noaa/*.dly                     — 158 GHCN-Daily station files
    data/raw/noaa/STATION_REFERENCE.csv     — Station metadata with lat/lon
    data/raw/tiger_county/tl_2020_us_county — TIGER/Line shapefile for parish boundaries

WRITES:
    data/processed/weather_daily.csv             — Parish-day weather (all 64 parishes)
    data/processed/weather_baseline_1981_2010.csv — Jul-Aug historical baseline for HW thresholds
    data/processed/station_parish_mapping.csv     — Station-to-parish assignments
    data/processed/parish_station_assignments.csv — All 64 parishes with assigned station(s)

FIXES APPLIED (vs. original pipeline):
    1. Parse ALL 158 .dly files (expanded from 50)
    2. Respect GHCN quality flags: drop any daily value where QFlag != blank
    3. Use point-in-polygon with TIGER shapefile to assign stations to parishes
    4. Validate TMAX >= TMIN after conversion; flag/log anomalies
    5. Sanity bounds: drop tmax > 130°F or tmin < -20°F
    6. Filter to study period 2015-01-01 through 2025-07-28 (+ 1981-2010 baseline)
    7. Only include stations within Louisiana or within 25 km of a parish boundary
    8. NEAREST-STATION INTERPOLATION for parishes without direct station coverage
    9. 1981-2010 HISTORICAL BASELINE extraction for heat wave threshold computation

AUTHOR: Emmanuel Adeniyi
DATE: 2026-03-26
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_NOAA_DIR = os.path.join(PROJECT_DIR, "data", "raw", "noaa")
RAW_TIGER_DIR = os.path.join(PROJECT_DIR, "data", "raw", "tiger_county")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")

STUDY_START = "2015-01-01"
STUDY_END = "2025-07-28"

# Elements we need from GHCN-Daily
ELEMENTS_OF_INTEREST = {"TMAX", "TMIN", "ADPT", "RHAV"}

# Sanity bounds (Fahrenheit) — values outside these are dropped
TMAX_UPPER_BOUND = 130.0  # No Louisiana station should ever exceed this
TMIN_LOWER_BOUND = -20.0  # Extremely rare even in north Louisiana

# Maximum distance (km) for out-of-state stations to be included
MAX_BORDER_STATION_DISTANCE_KM = 25.0


# ============================================================================
# GHCN-DAILY PARSER
# ============================================================================

def parse_dly_file(filepath):
    """
    Parse a single GHCN-Daily .dly file into a DataFrame of daily observations.

    GHCN-Daily fixed-width format:
        Chars 0-10:  Station ID (11 chars)
        Chars 11-14: Year (4 chars)
        Chars 15-16: Month (2 chars)
        Chars 17-20: Element (4 chars)
        Chars 21+:   31 daily value groups, each 8 chars:
                     Value (5 chars, right-justified), MFlag (1), QFlag (1), SFlag (1)

    Values are in tenths of a unit (e.g., 256 = 25.6°C for temperature).
    -9999 means missing.
    QFlag = blank means the value passed all quality checks.
    """
    records = []
    station_id = os.path.basename(filepath).replace(".dly", "")

    with open(filepath, "r") as f:
        for line in f:
            if len(line) < 269:
                continue

            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21].strip()

            # Skip elements we don't need
            if element not in ELEMENTS_OF_INTEREST:
                continue

            # Include study period (2014-2025) AND historical baseline (1981-2010)
            if not ((1981 <= year <= 2010) or (2014 <= year <= 2025)):
                continue

            # Parse 31 daily values
            for day in range(1, 32):
                offset = 21 + (day - 1) * 8
                value_str = line[offset:offset + 5].strip()
                # mflag = line[offset + 5]  # Measurement flag (not used)
                qflag = line[offset + 6]     # Quality flag
                # sflag = line[offset + 7]  # Source flag (not used)

                # Parse value
                try:
                    value = int(value_str)
                except ValueError:
                    continue

                # Skip missing values
                if value == -9999:
                    continue

                # QUALITY FLAG CHECK: blank = passed QC, anything else = failed
                # This was completely ignored in the original pipeline
                if qflag != " ":
                    continue

                # Construct date (skip invalid dates like Feb 30)
                try:
                    date = datetime(year, month, day)
                except ValueError:
                    continue

                records.append({
                    "station_id": station_id,
                    "date": date,
                    "element": element,
                    "value": value  # Still in tenths of original unit
                })

    return pd.DataFrame(records)


def parse_all_stations(noaa_dir):
    """Parse all .dly files and combine into one DataFrame."""
    dly_files = sorted(glob.glob(os.path.join(noaa_dir, "*.dly")))
    print(f"Found {len(dly_files)} .dly files to parse")

    all_records = []
    for i, filepath in enumerate(dly_files):
        station_id = os.path.basename(filepath).replace(".dly", "")
        df = parse_dly_file(filepath)
        if len(df) > 0:
            all_records.append(df)
        if (i + 1) % 10 == 0:
            print(f"  Parsed {i + 1}/{len(dly_files)} stations...")

    combined = pd.concat(all_records, ignore_index=True)
    print(f"Total raw records (after QFlag filter): {len(combined):,}")
    return combined


# ============================================================================
# STATION-TO-PARISH ASSIGNMENT (Point-in-Polygon)
# ============================================================================

def assign_stations_to_parishes(station_ref_path, tiger_dir):
    """
    Assign each weather station to a Louisiana parish using point-in-polygon
    with the TIGER/Line shapefile. This replaces the old hardcoded dictionary
    that had known FIPS errors (Calhoun→Bossier, Carville→Ascension, etc.).

    For Mississippi border stations (Parish = "Unknown" in reference file),
    we check if they're within MAX_BORDER_STATION_DISTANCE_KM of any Louisiana
    parish centroid and assign to the nearest one if so.

    Returns a DataFrame with station_id, parish_fips, parish_name, lat, lon,
    assignment_method, distance_km.
    """
    # Load station reference
    stations = pd.read_csv(station_ref_path)
    print(f"Station reference: {len(stations)} stations")

    # Load TIGER shapefile, filter to Louisiana
    counties = gpd.read_file(tiger_dir)
    la_parishes = counties[counties["STATEFP"] == "22"].copy()
    la_parishes = la_parishes.to_crs("EPSG:4326")  # WGS84 for lat/lon matching
    print(f"Louisiana parishes in shapefile: {len(la_parishes)}")

    # Build station GeoDataFrame
    station_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["Longitude"], stations["Latitude"]),
        crs="EPSG:4326"
    )

    # Spatial join: point-in-polygon
    joined = gpd.sjoin(station_gdf, la_parishes[["GEOID", "NAME", "geometry"]],
                       how="left", predicate="within")

    results = []
    for _, row in joined.iterrows():
        station_id = row["Station_ID"]
        lat = row["Latitude"]
        lon = row["Longitude"]
        ref_parish = row["Parish"]  # From STATION_REFERENCE.csv

        if pd.notna(row.get("GEOID")):
            # Station falls within a Louisiana parish
            parish_fips = int(row["GEOID"])
            parish_name = row["NAME"]
            method = "point_in_polygon"
            dist_km = 0.0
        else:
            # Station is outside Louisiana (likely Mississippi border station)
            # Check distance to nearest parish centroid
            station_point = Point(lon, lat)

            # Use projected CRS for distance calculation
            la_projected = la_parishes.to_crs("EPSG:3857")
            station_projected = gpd.GeoSeries([station_point], crs="EPSG:4326").to_crs("EPSG:3857")

            distances = la_projected.geometry.distance(station_projected.iloc[0])
            min_idx = distances.idxmin()
            min_dist_m = distances[min_idx]
            min_dist_km = min_dist_m / 1000.0

            if min_dist_km <= MAX_BORDER_STATION_DISTANCE_KM:
                parish_fips = int(la_parishes.loc[min_idx, "GEOID"])
                parish_name = la_parishes.loc[min_idx, "NAME"]
                method = f"nearest_parish_{min_dist_km:.1f}km"
                dist_km = min_dist_km
            else:
                # Too far — skip this station
                print(f"  SKIPPING {station_id} ({row['Station_Name']}): "
                      f"{min_dist_km:.1f} km from nearest LA parish (>{MAX_BORDER_STATION_DISTANCE_KM} km)")
                continue

        results.append({
            "station_id": station_id,
            "parish_fips": parish_fips,
            "parish_name": parish_name,
            "latitude": lat,
            "longitude": lon,
            "ref_parish": ref_parish,
            "assignment_method": method,
            "distance_km": round(dist_km, 1)
        })

    mapping_df = pd.DataFrame(results)

    # Log comparison with original reference
    print(f"\nStation assignment results:")
    print(f"  Stations assigned to LA parishes: {len(mapping_df)}")
    mismatches = mapping_df[mapping_df["ref_parish"] != mapping_df["parish_name"]]
    mismatches = mismatches[mismatches["ref_parish"] != "Unknown"]
    if len(mismatches) > 0:
        print(f"  Parish assignment corrections (vs. STATION_REFERENCE.csv):")
        for _, m in mismatches.iterrows():
            print(f"    {m['station_id']} ({m['latitude']:.3f}, {m['longitude']:.3f}): "
                  f"was '{m['ref_parish']}' -> now '{m['parish_name']}' [{m['assignment_method']}]")

    return mapping_df


# ============================================================================
# UNIT CONVERSION AND VALIDATION
# ============================================================================

def convert_and_validate(weather_df, station_mapping, date_start=None, date_end=None):
    """
    Convert raw GHCN values to usable units, pivot to wide format,
    validate, and merge with parish assignments.

    Unit conversions:
        TMAX, TMIN: tenths of °C → °F  ((val/10) * 9/5 + 32)
        ADPT:       tenths of °C → °C  (val/10) — kept in Celsius for RH calc
        RHAV:       whole percent (no conversion needed, but stored as tenths in raw)

    Note on RHAV: In GHCN-Daily, RHAV values are stored as whole percent
    (not tenths), so value 77 = 77% RH.
    """
    # Filter to requested period
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    start = date_start or STUDY_START
    end = date_end or STUDY_END
    mask = (weather_df["date"] >= start) & (weather_df["date"] <= end)
    weather_df = weather_df[mask].copy()
    print(f"Records in period ({start} to {end}): {len(weather_df):,}")

    # Only keep stations that are in our parish mapping
    valid_stations = set(station_mapping["station_id"])
    before = len(weather_df)
    weather_df = weather_df[weather_df["station_id"].isin(valid_stations)].copy()
    print(f"Records from mapped stations: {len(weather_df):,} (dropped {before - len(weather_df):,})")

    # Convert units BEFORE pivoting
    # Cast value to float first to avoid dtype incompatibility warnings
    weather_df["value"] = weather_df["value"].astype(float)

    # Temperature elements: tenths of °C -> °F
    temp_mask = weather_df["element"].isin(["TMAX", "TMIN"])
    weather_df.loc[temp_mask, "value"] = (
        weather_df.loc[temp_mask, "value"] / 10.0 * 9.0 / 5.0 + 32.0
    )

    # Dew point: tenths of °C -> °C (keep Celsius for Magnus formula later)
    adpt_mask = weather_df["element"] == "ADPT"
    weather_df.loc[adpt_mask, "value"] = weather_df.loc[adpt_mask, "value"] / 10.0

    # RHAV: stored as whole percent in GHCN-Daily (no conversion)
    # But verify range
    rhav_mask = weather_df["element"] == "RHAV"
    rhav_values = weather_df.loc[rhav_mask, "value"]
    if len(rhav_values) > 0:
        print(f"RHAV range: {rhav_values.min():.0f} to {rhav_values.max():.0f} "
              f"(expect 0-100)")
        # If values are > 100, they might be in tenths — check
        if rhav_values.max() > 100:
            print("  WARNING: RHAV values > 100 detected, likely in tenths — dividing by 10")
            weather_df.loc[rhav_mask, "value"] = weather_df.loc[rhav_mask, "value"] / 10.0

    # Pivot to wide format: one row per (station_id, date)
    weather_wide = weather_df.pivot_table(
        index=["station_id", "date"],
        columns="element",
        values="value",
        aggfunc="mean"  # mean handles rare duplicates (better than 'first')
    ).reset_index()

    # Flatten column names
    weather_wide.columns.name = None

    # Rename for clarity
    rename_map = {}
    if "TMAX" in weather_wide.columns:
        rename_map["TMAX"] = "tmax_f"
    if "TMIN" in weather_wide.columns:
        rename_map["TMIN"] = "tmin_f"
    if "ADPT" in weather_wide.columns:
        rename_map["ADPT"] = "dewpoint_c"
    if "RHAV" in weather_wide.columns:
        rename_map["RHAV"] = "rh_pct"
    weather_wide = weather_wide.rename(columns=rename_map)

    print(f"Station-day observations after pivot: {len(weather_wide):,}")

    # --- SANITY CHECKS ---

    # 1. Drop extreme values
    if "tmax_f" in weather_wide.columns:
        bad_tmax = weather_wide["tmax_f"] > TMAX_UPPER_BOUND
        if bad_tmax.any():
            print(f"  Dropping {bad_tmax.sum()} rows with tmax > {TMAX_UPPER_BOUND}°F")
            weather_wide.loc[bad_tmax, "tmax_f"] = np.nan

    if "tmin_f" in weather_wide.columns:
        bad_tmin = weather_wide["tmin_f"] < TMIN_LOWER_BOUND
        if bad_tmin.any():
            print(f"  Dropping {bad_tmin.sum()} rows with tmin < {TMIN_LOWER_BOUND}°F")
            weather_wide.loc[bad_tmin, "tmin_f"] = np.nan

    # 2. Check TMAX >= TMIN
    if "tmax_f" in weather_wide.columns and "tmin_f" in weather_wide.columns:
        both_present = weather_wide["tmax_f"].notna() & weather_wide["tmin_f"].notna()
        swapped = both_present & (weather_wide["tmin_f"] > weather_wide["tmax_f"])
        n_swapped = swapped.sum()
        if n_swapped > 0:
            print(f"  WARNING: {n_swapped} station-days have TMIN > TMAX ({n_swapped/both_present.sum()*100:.2f}%)")
            # Swap them — this is the most likely cause (mislabeled in source)
            swap_idx = weather_wide[swapped].index
            tmax_temp = weather_wide.loc[swap_idx, "tmax_f"].copy()
            weather_wide.loc[swap_idx, "tmax_f"] = weather_wide.loc[swap_idx, "tmin_f"]
            weather_wide.loc[swap_idx, "tmin_f"] = tmax_temp
            print(f"  Swapped TMAX/TMIN for these {n_swapped} rows")

    # 3. Validate RH range
    if "rh_pct" in weather_wide.columns:
        bad_rh = (weather_wide["rh_pct"] < 0) | (weather_wide["rh_pct"] > 100)
        if bad_rh.any():
            print(f"  Dropping {bad_rh.sum()} rows with RH outside 0-100%")
            weather_wide.loc[bad_rh, "rh_pct"] = np.nan

    return weather_wide


# ============================================================================
# PARISH-LEVEL AGGREGATION
# ============================================================================

def aggregate_to_parish_day(weather_wide, station_mapping):
    """
    Merge station data with parish assignments and aggregate to parish-day level.

    For parishes with multiple stations, we take the mean of each variable.
    This is a standard approach for spatial averaging.
    """
    # Merge with parish mapping
    merged = weather_wide.merge(
        station_mapping[["station_id", "parish_fips", "parish_name"]],
        on="station_id",
        how="inner"
    )
    print(f"Station-days with parish assignment: {len(merged):,}")

    # Aggregate to parish-day
    agg_cols = {}
    for col in ["tmax_f", "tmin_f", "dewpoint_c", "rh_pct"]:
        if col in merged.columns:
            agg_cols[col] = "mean"

    parish_day = merged.groupby(["parish_fips", "parish_name", "date"]).agg(
        **{col: (col, func) for col, func in agg_cols.items()},
        n_stations=("station_id", "nunique")
    ).reset_index()

    # Round to reasonable precision
    for col in ["tmax_f", "tmin_f", "dewpoint_c", "rh_pct"]:
        if col in parish_day.columns:
            parish_day[col] = parish_day[col].round(2)

    # Summary stats
    parishes_with_data = parish_day["parish_fips"].nunique()
    date_range = f"{parish_day['date'].min().date()} to {parish_day['date'].max().date()}"
    total_parish_days = len(parish_day)

    print(f"\nParish-day aggregation complete:")
    print(f"  Parishes with weather data: {parishes_with_data}")
    print(f"  Date range: {date_range}")
    print(f"  Total parish-day observations: {total_parish_days:,}")
    print(f"  Stations per parish-day: min={parish_day['n_stations'].min()}, "
          f"max={parish_day['n_stations'].max()}, "
          f"mean={parish_day['n_stations'].mean():.1f}")

    # Coverage by parish
    print(f"\n  Coverage by parish (top 10 by observation count):")
    coverage = parish_day.groupby("parish_name").size().sort_values(ascending=False)
    for parish, count in coverage.head(10).items():
        print(f"    {parish}: {count:,} days")

    # Check how many parish-days have humidity data
    if "rh_pct" in parish_day.columns:
        rh_coverage = parish_day["rh_pct"].notna().sum()
        print(f"\n  Parish-days with real RH data: {rh_coverage:,} "
              f"({rh_coverage/total_parish_days*100:.1f}%)")
    if "dewpoint_c" in parish_day.columns:
        dp_coverage = parish_day["dewpoint_c"].notna().sum()
        print(f"  Parish-days with dew point data: {dp_coverage:,} "
              f"({dp_coverage/total_parish_days*100:.1f}%)")

    return parish_day


# ============================================================================
# NEAREST-STATION INTERPOLATION
# ============================================================================

def assign_nearest_stations_to_all_parishes(station_mapping, tiger_dir, active_stations=None):
    """
    For every Louisiana parish, assign the nearest weather station.
    Parishes that already have a station via point-in-polygon get that station.
    Parishes without direct coverage get weather from the nearest station.

    If active_stations is provided (set of station_ids with study-period data),
    parishes with only inactive direct stations will also get nearest-neighbor
    assignment from an active station.

    Returns a DataFrame with all 64 parishes and their assigned station(s),
    plus an assignment_type column ('direct' or 'nearest_neighbor').
    """
    print("\n--- Nearest-Station Assignment for All 64 Parishes ---")

    # Load TIGER shapefile for all LA parishes
    counties = gpd.read_file(tiger_dir)
    la_parishes = counties[counties["STATEFP"] == "22"].copy()
    la_parishes["parish_fips"] = la_parishes["GEOID"].astype(int)

    # Project to a meter-based CRS for distance calculations
    la_projected = la_parishes.to_crs("EPSG:3857")
    la_projected["centroid"] = la_projected.geometry.centroid

    # Build station points (unique stations from mapping)
    station_locs = station_mapping.drop_duplicates("station_id")[
        ["station_id", "latitude", "longitude", "parish_fips"]
    ].copy()
    station_gdf = gpd.GeoDataFrame(
        station_locs,
        geometry=gpd.points_from_xy(station_locs["longitude"], station_locs["latitude"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    # Build active-only station GeoDataFrame for fallback
    if active_stations is not None:
        active_gdf = station_gdf[station_gdf["station_id"].isin(active_stations)].copy()
        print(f"  Active stations (with study-period data): {len(active_gdf)}")
    else:
        active_gdf = station_gdf

    # Parishes that already have direct station coverage
    direct_parishes = set(station_mapping["parish_fips"].unique())

    assignments = []
    for _, parish_row in la_projected.iterrows():
        fips = parish_row["parish_fips"]
        name = parish_row["NAME"]
        centroid = parish_row["centroid"]

        if fips in direct_parishes:
            # Direct coverage — list the station(s) already assigned
            direct_stations = station_mapping[station_mapping["parish_fips"] == fips]

            # Check if any direct station is active
            has_active_direct = False
            if active_stations is not None:
                has_active_direct = any(
                    sid in active_stations for sid in direct_stations["station_id"]
                )
            else:
                has_active_direct = True

            if has_active_direct:
                for _, st in direct_stations.iterrows():
                    # Skip inactive direct stations when we know which are active
                    if active_stations is not None and st["station_id"] not in active_stations:
                        continue
                    assignments.append({
                        "parish_fips": fips,
                        "parish_name": name,
                        "station_id": st["station_id"],
                        "assignment_type": "direct",
                        "distance_km": st.get("distance_km", 0.0)
                    })
            else:
                # All direct stations are inactive — fall back to nearest active
                distances = active_gdf.geometry.distance(centroid)
                nearest_idx = distances.idxmin()
                nearest_dist_km = distances[nearest_idx] / 1000.0
                nearest_station = active_gdf.loc[nearest_idx, "station_id"]
                print(f"  {name} ({fips}): direct station inactive, "
                      f"using {nearest_station} at {nearest_dist_km:.1f} km")
                assignments.append({
                    "parish_fips": fips,
                    "parish_name": name,
                    "station_id": nearest_station,
                    "assignment_type": "nearest_neighbor",
                    "distance_km": round(nearest_dist_km, 1)
                })
        else:
            # No direct coverage — find nearest active station
            distances = active_gdf.geometry.distance(centroid)
            nearest_idx = distances.idxmin()
            nearest_dist_km = distances[nearest_idx] / 1000.0
            nearest_station = active_gdf.loc[nearest_idx, "station_id"]

            assignments.append({
                "parish_fips": fips,
                "parish_name": name,
                "station_id": nearest_station,
                "assignment_type": "nearest_neighbor",
                "distance_km": round(nearest_dist_km, 1)
            })

    assign_df = pd.DataFrame(assignments)

    n_direct = assign_df[assign_df["assignment_type"] == "direct"]["parish_fips"].nunique()
    n_nn = assign_df[assign_df["assignment_type"] == "nearest_neighbor"]["parish_fips"].nunique()
    nn_dists = assign_df[assign_df["assignment_type"] == "nearest_neighbor"]["distance_km"]

    print(f"  Parishes with direct station: {n_direct}")
    print(f"  Parishes using nearest-neighbor: {n_nn}")
    if len(nn_dists) > 0:
        print(f"  Nearest-neighbor distances: min={nn_dists.min():.1f} km, "
              f"max={nn_dists.max():.1f} km, mean={nn_dists.mean():.1f} km")

    return assign_df


def build_full_parish_weather(weather_station_day, all_parish_assignments):
    """
    Build weather data for ALL 64 parishes using station assignments.
    Direct-coverage parishes use their own station(s) (averaged if multiple).
    Nearest-neighbor parishes get weather from the assigned nearest station.

    Returns a parish-day DataFrame for all 64 parishes.
    """
    print("\n--- Building Weather for All 64 Parishes ---")

    # Merge station-day weather with ALL parish assignments
    merged = weather_station_day.merge(
        all_parish_assignments[["parish_fips", "parish_name", "station_id", "assignment_type"]],
        on="station_id",
        how="inner"
    )
    print(f"  Station-day records after assignment merge: {len(merged):,}")

    # Aggregate to parish-day level (mean if multiple stations)
    agg_cols = {}
    for col in ["tmax_f", "tmin_f", "dewpoint_c", "rh_pct"]:
        if col in merged.columns:
            agg_cols[col] = "mean"

    parish_day = merged.groupby(["parish_fips", "parish_name", "date"]).agg(
        **{col: (col, func) for col, func in agg_cols.items()},
        n_stations=("station_id", "nunique"),
        assignment_type=("assignment_type", "first")  # direct or nearest_neighbor
    ).reset_index()

    # Round to reasonable precision
    for col in ["tmax_f", "tmin_f", "dewpoint_c", "rh_pct"]:
        if col in parish_day.columns:
            parish_day[col] = parish_day[col].round(2)

    n_parishes = parish_day["parish_fips"].nunique()
    n_direct = parish_day[parish_day["assignment_type"] == "direct"]["parish_fips"].nunique()
    n_nn = parish_day[parish_day["assignment_type"] == "nearest_neighbor"]["parish_fips"].nunique()

    print(f"  Total parishes with weather: {n_parishes}")
    print(f"    Direct station coverage: {n_direct} parishes")
    print(f"    Nearest-neighbor interpolation: {n_nn} parishes")
    print(f"  Total parish-day observations: {len(parish_day):,}")
    print(f"  Date range: {parish_day['date'].min().date()} to {parish_day['date'].max().date()}")

    return parish_day


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("STEP 01: PARSE NOAA GHCN-DAILY WEATHER DATA")
    print("=" * 70)
    start_time = datetime.now()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Assign stations to parishes using point-in-polygon ---
    print("\n--- Station-to-Parish Assignment (Point-in-Polygon) ---")
    station_ref_path = os.path.join(RAW_NOAA_DIR, "STATION_REFERENCE.csv")
    station_mapping = assign_stations_to_parishes(station_ref_path, RAW_TIGER_DIR)

    mapping_path = os.path.join(OUTPUT_DIR, "station_parish_mapping.csv")
    station_mapping.to_csv(mapping_path, index=False)
    print(f"Saved: {mapping_path}")

    # --- 2. Parse all .dly files (1981-2010 + 2014-2025) ---
    print("\n--- Parsing GHCN-Daily .dly Files ---")
    raw_weather = parse_all_stations(RAW_NOAA_DIR)

    # --- 3. Convert units for STUDY PERIOD (2015-2025) ---
    print("\n--- Unit Conversion and Validation (Study Period) ---")
    study_weather = convert_and_validate(
        raw_weather.copy(), station_mapping,
        date_start=STUDY_START, date_end=STUDY_END
    )

    # Determine which stations actually have study-period data
    active_stations = set(study_weather["station_id"].unique())
    print(f"  Active stations (with 2015+ data): {len(active_stations)}")

    # --- 4. Assign nearest stations to ALL 64 parishes (using active stations) ---
    all_parish_assignments = assign_nearest_stations_to_all_parishes(
        station_mapping, RAW_TIGER_DIR, active_stations=active_stations
    )
    assign_path = os.path.join(OUTPUT_DIR, "parish_station_assignments.csv")
    all_parish_assignments.to_csv(assign_path, index=False)
    print(f"Saved: {assign_path}")

    # --- 5a. Convert units for HISTORICAL BASELINE (1981-2010, Jul-Aug only) ---
    print("\n--- Unit Conversion and Validation (Historical Baseline 1981-2010) ---")
    baseline_weather = convert_and_validate(
        raw_weather.copy(), station_mapping,
        date_start="1981-01-01", date_end="2010-12-31"
    )
    # Filter to July-August only for baseline
    baseline_weather["date"] = pd.to_datetime(baseline_weather["date"])
    baseline_weather = baseline_weather[
        baseline_weather["date"].dt.month.isin([7, 8])
    ].copy()
    print(f"  Jul-Aug baseline station-days: {len(baseline_weather):,}")

    # 5b. Build baseline parish assignments (stations active in 1981-2010)
    baseline_active = set(baseline_weather["station_id"].unique())
    print(f"  Stations with 1981-2010 Jul-Aug data: {len(baseline_active)}")
    baseline_parish_assignments = assign_nearest_stations_to_all_parishes(
        station_mapping, RAW_TIGER_DIR, active_stations=baseline_active
    )

    # --- 6. Build weather for ALL 64 parishes (study period) ---
    parish_day = build_full_parish_weather(study_weather, all_parish_assignments)

    # --- 7. Build baseline for ALL 64 parishes (1981-2010 Jul-Aug) ---
    print("\n--- Building Historical Baseline for All Parishes ---")
    baseline_parish_day = build_full_parish_weather(baseline_weather, baseline_parish_assignments)
    print(f"  Historical baseline parish-days (Jul-Aug): {len(baseline_parish_day):,}")

    # --- 8. Save outputs ---
    output_path = os.path.join(OUTPUT_DIR, "weather_daily.csv")
    parish_day.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(parish_day):,}")
    print(f"  Columns: {list(parish_day.columns)}")

    baseline_path = os.path.join(OUTPUT_DIR, "weather_baseline_1981_2010.csv")
    baseline_parish_day.to_csv(baseline_path, index=False)
    print(f"\nSaved: {baseline_path}")
    print(f"  Rows: {len(baseline_parish_day):,}")
    print(f"  Parishes: {baseline_parish_day['parish_fips'].nunique()}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
