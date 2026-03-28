"""
03_process_osha.py — Process OSHA Severe Injury Reports for Louisiana

PURPOSE:
    Parse the raw OSHA SIR download, geocode incidents to Louisiana parishes
    using point-in-polygon spatial join with TIGER/Line shapefile, classify
    heat-related incidents, and aggregate to parish-day level.

READS:
    data/raw/osha/SIRDataDownload.csv       — Raw OSHA Severe Injury Reports
    data/raw/tiger_county/tl_2020_us_county  — TIGER/Line shapefile

WRITES:
    data/processed/osha_processed.csv       — Individual incident records with parish assignment
    data/processed/parish_day_accidents.csv  — Aggregated parish-day incident counts

FIXES APPLIED (vs. original pipeline):
    1. Robust CSV parsing: handle rows with extra fields by merging overflow
       back into the FinalNarrative column (25 rows affected), instead of
       silently dropping them
    2. Point-in-polygon geocoding via geopandas spatial join against TIGER
       shapefile (old pipeline used nearest-centroid or hardcoded single point)
    3. Coordinate validation: reject lat/lon outside Louisiana bounding box
    4. Heat-related classification uses BOTH structured Event code 531 AND
       keyword regex with word boundaries (old pipeline ignored Event code
       and 'hot' matched 'shot', 'photo', etc.)
    5. Log exact counts at every step for audit trail
    6. NAICS industry classification with standard BLS sector mapping

METHODOLOGY:
    Geocoding:
        - Primary: point-in-polygon spatial join (incident lat/lon within parish boundary)
        - Fallback for invalid/missing coordinates: excluded (logged)
        - Coordinates outside Louisiana bbox (28.5-33.5 N, 94.1-88.7 W) are rejected

    Heat-related classification:
        - Tier 1: OSHA Event code == 531 ("Exposure to environmental heat")
        - Tier 2: Keyword regex for environmental heat illness phrases in
          FinalNarrative or EventTitle (heat stroke, heat stress, heat exhaustion,
          heat-related, dehydration, sunburn, etc.)
          Excludes industrial thermal burns (hot water, steam, hot metal, etc.)
        - A record is heat-related if EITHER tier matches
        - The classification tier is recorded for transparency

AUTHOR: Emmanuel Adeniyi
DATE: 2026-03-24
"""

import os
import csv
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_OSHA_PATH = os.path.join(PROJECT_DIR, "data", "raw", "osha", "SIRDataDownload.csv")
RAW_TIGER_DIR = os.path.join(PROJECT_DIR, "data", "raw", "tiger_county")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")

# Louisiana bounding box for coordinate validation
LA_LAT_MIN, LA_LAT_MAX = 28.5, 33.5
LA_LON_MIN, LA_LON_MAX = -94.1, -88.7

# Heat-related keyword pattern — focused on ENVIRONMENTAL heat exposure
# Excludes industrial thermal burns (hot water, hot metal, steam, hot asphalt, etc.)
# which are thermal contact injuries, not environmental heat illness.
# The broad terms 'hot' and 'sun' caused 66 false positives in v1 (hot water,
# steam burns, hot asphalt, etc.), so we now use specific compound phrases only.
HEAT_KEYWORDS_PATTERN = re.compile(
    r'heat.?stroke|heat.?stress|heat.?exhaust|heat.?relat|heat.?illness|'
    r'heat.?exposure|heat.?cramp|heat.?emergenc|'
    r'dehydrat|\bsunburn\b|\bsunstroke\b|\bsun.?exposure\b|'
    r'environmental.{0,10}heat|effects? of heat',
    re.IGNORECASE
)

# OSHA Event code for environmental heat exposure
HEAT_EVENT_CODE = "531"

# NAICS sector mapping (2-digit NAICS -> sector name)
NAICS_SECTOR_MAP = {
    "11": "Agriculture, Forestry, Fishing",
    "21": "Mining, Oil & Gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation & Warehousing",
    "49": "Transportation & Warehousing",
    "51": "Information",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional & Technical Services",
    "55": "Management of Companies",
    "56": "Administrative & Waste Services",
    "61": "Educational Services",
    "62": "Health Care & Social Assistance",
    "71": "Arts, Entertainment & Recreation",
    "72": "Accommodation & Food Services",
    "81": "Other Services",
    "92": "Public Administration",
}


# ============================================================================
# ROBUST CSV PARSING
# ============================================================================

def parse_osha_csv(filepath):
    """
    Parse the OSHA SIR CSV file, handling rows with extra fields caused by
    unescaped commas in the FinalNarrative column.

    The raw file has 27 columns. About 25 rows have 28-30 fields because
    the FinalNarrative (column index 16) contains commas that aren't properly
    quoted. We detect these rows and merge the extra fields back into the
    narrative.

    Returns a DataFrame with all 27 expected columns.
    """
    print("--- Parsing OSHA CSV ---")
    EXPECTED_COLS = 27
    NARRATIVE_COL_IDX = 16  # FinalNarrative is the 17th column (0-indexed: 16)

    rows = []
    repaired = 0
    skipped = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert len(header) == EXPECTED_COLS, f"Expected {EXPECTED_COLS} header cols, got {len(header)}"

        for line_num, row in enumerate(reader, start=2):
            if len(row) == EXPECTED_COLS:
                rows.append(row)
            elif len(row) > EXPECTED_COLS:
                # Extra fields — merge them back into the narrative column
                overflow = len(row) - EXPECTED_COLS
                # The narrative and the overflow fields should be joined
                narrative_parts = row[NARRATIVE_COL_IDX:NARRATIVE_COL_IDX + overflow + 1]
                merged_narrative = ", ".join(narrative_parts)
                repaired_row = (
                    row[:NARRATIVE_COL_IDX]
                    + [merged_narrative]
                    + row[NARRATIVE_COL_IDX + overflow + 1:]
                )
                if len(repaired_row) == EXPECTED_COLS:
                    rows.append(repaired_row)
                    repaired += 1
                else:
                    skipped += 1
                    print(f"  WARNING: Could not repair line {line_num} "
                          f"({len(row)} fields), skipping")
            else:
                skipped += 1
                print(f"  WARNING: Line {line_num} has only {len(row)} fields, skipping")

    df = pd.DataFrame(rows, columns=header)
    print(f"  Rows parsed: {len(df)}")
    print(f"  Rows repaired (extra fields merged): {repaired}")
    print(f"  Rows skipped: {skipped}")

    return df


# ============================================================================
# COORDINATE VALIDATION AND GEOCODING
# ============================================================================

def validate_and_geocode(df, tiger_dir):
    """
    Validate incident coordinates and assign to Louisiana parishes using
    point-in-polygon spatial join with TIGER/Line shapefile.

    Steps:
    1. Parse lat/lon to float, drop non-numeric
    2. Reject points outside Louisiana bounding box
    3. Spatial join against TIGER parish polygons
    4. Log incidents that couldn't be assigned
    """
    print("\n--- Geocoding via Point-in-Polygon ---")

    # Parse coordinates
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    total = len(df)
    has_coords = df["Latitude"].notna() & df["Longitude"].notna()
    print(f"  Records with coordinates: {has_coords.sum()} / {total}")

    # Validate Louisiana bounding box
    in_bbox = (
        has_coords &
        (df["Latitude"] >= LA_LAT_MIN) & (df["Latitude"] <= LA_LAT_MAX) &
        (df["Longitude"] >= LA_LON_MIN) & (df["Longitude"] <= LA_LON_MAX)
    )
    out_of_bbox = has_coords & ~in_bbox
    if out_of_bbox.any():
        print(f"  Records outside Louisiana bbox: {out_of_bbox.sum()}")
        for _, row in df[out_of_bbox].head(5).iterrows():
            print(f"    ID {row['ID']}: ({row['Latitude']}, {row['Longitude']}) "
                  f"- {row['city']}")

    # Load TIGER shapefile
    counties = gpd.read_file(tiger_dir)
    la_parishes = counties[counties["STATEFP"] == "22"].copy()
    la_parishes = la_parishes.to_crs("EPSG:4326")
    print(f"  Louisiana parishes in shapefile: {len(la_parishes)}")

    # Create GeoDataFrame from valid-bbox records
    valid_df = df[in_bbox].copy()
    valid_gdf = gpd.GeoDataFrame(
        valid_df,
        geometry=gpd.points_from_xy(valid_df["Longitude"], valid_df["Latitude"]),
        crs="EPSG:4326"
    )

    # Spatial join
    joined = gpd.sjoin(
        valid_gdf,
        la_parishes[["GEOID", "NAME", "geometry"]],
        how="left",
        predicate="within"
    )

    # Check for unmatched (point didn't fall in any parish — could be offshore)
    unmatched = joined["GEOID"].isna()
    if unmatched.any():
        print(f"  Records in bbox but not in any parish polygon: {unmatched.sum()}")
        for _, row in joined[unmatched].head(5).iterrows():
            print(f"    ID {row['ID']}: ({row['Latitude']}, {row['Longitude']}) "
                  f"- {row['city']}")

    # Build result
    joined["parish_fips"] = pd.to_numeric(joined["GEOID"], errors="coerce").astype("Int64")
    joined["parish_name"] = joined["NAME"]

    # Drop geometry and spatial join artifacts
    result = pd.DataFrame(joined.drop(columns=["geometry", "index_right", "GEOID", "NAME"],
                                       errors="ignore"))

    # Add back records that couldn't be geocoded (no coords or out of bbox)
    no_geocode = df[~in_bbox].copy()
    no_geocode["parish_fips"] = pd.NA
    no_geocode["parish_name"] = None

    result = pd.concat([result, no_geocode], ignore_index=True)

    geocoded = result["parish_fips"].notna()
    print(f"\n  Geocoding summary:")
    print(f"    Total records: {len(result)}")
    print(f"    Successfully geocoded: {geocoded.sum()}")
    print(f"    Not geocoded: {(~geocoded).sum()}")
    print(f"    Unique parishes: {result.loc[geocoded, 'parish_fips'].nunique()}")

    return result


# ============================================================================
# HEAT-RELATED CLASSIFICATION
# ============================================================================

def classify_heat_related(df):
    """
    Classify incidents as heat-related using two tiers:
    Tier 1: OSHA Event code 531 (structured field)
    Tier 2: Keyword regex in FinalNarrative or EventTitle (with word boundaries)

    Records the classification tier for transparency.
    """
    print("\n--- Heat-Related Classification ---")

    # Tier 1: Event code 531
    df["Event"] = df["Event"].astype(str).str.strip()
    tier1 = df["Event"] == HEAT_EVENT_CODE
    print(f"  Tier 1 (Event code 531): {tier1.sum()} records")

    # Tier 2: Keyword regex
    narrative = df["FinalNarrative"].fillna("").astype(str)
    event_title = df["EventTitle"].fillna("").astype(str)

    nar_match = narrative.str.contains(HEAT_KEYWORDS_PATTERN, na=False)
    title_match = event_title.str.contains(HEAT_KEYWORDS_PATTERN, na=False)
    tier2 = (nar_match | title_match) & ~tier1  # Only count if not already tier 1
    print(f"  Tier 2 (keyword only, not in tier 1): {tier2.sum()} records")

    # Combined
    df["is_heat_related"] = tier1 | nar_match | title_match
    df["heat_classification"] = "none"
    df.loc[tier1, "heat_classification"] = "event_code_531"
    df.loc[tier2, "heat_classification"] = "keyword_match"

    total_heat = df["is_heat_related"].sum()
    print(f"  Total heat-related: {total_heat} ({total_heat/len(df)*100:.1f}%)")

    # Show some keyword-only matches to verify no false positives
    keyword_only = df[tier2].head(10)
    if len(keyword_only) > 0:
        print(f"\n  Sample keyword-only matches (verify no false positives):")
        for _, row in keyword_only.iterrows():
            narrative_snippet = str(row["FinalNarrative"])[:100]
            print(f"    ID {row['ID']}: {narrative_snippet}...")

    return df


# ============================================================================
# INDUSTRY CLASSIFICATION
# ============================================================================

def classify_industry(df):
    """Classify incidents by industry sector using 2-digit NAICS code."""
    print("\n--- Industry Classification ---")

    df["Primary_NAICS"] = df["Primary_NAICS"].astype(str).str.strip()

    def get_sector(naics):
        if naics and len(naics) >= 2:
            prefix = naics[:2]
            return NAICS_SECTOR_MAP.get(prefix, "Unknown")
        return "Unknown"

    df["industry_sector"] = df["Primary_NAICS"].apply(get_sector)

    sector_counts = df["industry_sector"].value_counts()
    print(f"  Incidents by sector:")
    for sector, count in sector_counts.items():
        heat_in_sector = df[df["industry_sector"] == sector]["is_heat_related"].sum()
        print(f"    {sector}: {count} total, {heat_in_sector} heat-related")

    return df


# ============================================================================
# PARISH-DAY AGGREGATION
# ============================================================================

# Outdoor-exposed industry sectors (construction, agriculture, mining, transport)
OUTDOOR_INDUSTRIES = {
    "Construction",
    "Agriculture, Forestry, Fishing",
    "Mining, Oil & Gas",
    "Transportation & Warehousing",
    "Utilities",
    "Administrative & Waste Services",  # Includes landscaping, waste collection
}


def aggregate_to_parish_day(df):
    """
    Aggregate individual incidents to parish-day counts.
    Only includes geocoded records (parish_fips is not null).
    Now includes outdoor industry incident counts.
    """
    print("\n--- Parish-Day Aggregation ---")

    # Filter to geocoded records only
    geocoded = df[df["parish_fips"].notna()].copy()
    geocoded["EventDate"] = pd.to_datetime(geocoded["EventDate"])
    geocoded["is_outdoor_industry"] = geocoded["industry_sector"].isin(OUTDOOR_INDUSTRIES).astype(int)
    print(f"  Geocoded records for aggregation: {len(geocoded)}")
    print(f"  Outdoor-industry incidents: {geocoded['is_outdoor_industry'].sum()}")

    parish_day = geocoded.groupby(
        ["parish_fips", "parish_name", "EventDate"]
    ).agg(
        total_incidents=("ID", "count"),
        hospitalizations=("Hospitalized", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        amputations=("Amputation", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        loss_of_eye=("Loss_Of_Eye", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        heat_related=("is_heat_related", "sum"),
        outdoor_industry_incidents=("is_outdoor_industry", "sum"),
    ).reset_index()

    parish_day = parish_day.rename(columns={"EventDate": "date"})
    parish_day["parish_fips"] = parish_day["parish_fips"].astype(int)

    # Cast count columns to int
    for col in ["total_incidents", "hospitalizations", "amputations", "loss_of_eye",
                "heat_related", "outdoor_industry_incidents"]:
        parish_day[col] = parish_day[col].astype(int)

    print(f"  Parish-day rows: {len(parish_day)}")
    print(f"  Parishes with incidents: {parish_day['parish_fips'].nunique()}")
    print(f"  Date range: {parish_day['date'].min().date()} to {parish_day['date'].max().date()}")
    print(f"  Total incidents in aggregation: {parish_day['total_incidents'].sum()}")
    print(f"  Total heat-related in aggregation: {parish_day['heat_related'].sum()}")
    print(f"  Outdoor-industry incidents in aggregation: {parish_day['outdoor_industry_incidents'].sum()}")

    return parish_day


def compute_parish_industry_profile(df):
    """
    Compute parish-level industry composition from OSHA incidents.
    Returns a DataFrame with one row per parish containing:
    - outdoor_industry_share: fraction of incidents in outdoor-exposed industries
    - dominant_sector: most common industry sector
    - n_sectors: number of distinct industry sectors
    """
    print("\n--- Parish Industry Profile ---")

    geocoded = df[df["parish_fips"].notna()].copy()
    geocoded["is_outdoor"] = geocoded["industry_sector"].isin(OUTDOOR_INDUSTRIES).astype(int)

    profiles = geocoded.groupby(["parish_fips", "parish_name"]).agg(
        total_parish_incidents=("ID", "count"),
        outdoor_incidents=("is_outdoor", "sum"),
        n_sectors=("industry_sector", "nunique"),
        dominant_sector=("industry_sector", lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown"),
    ).reset_index()

    profiles["outdoor_industry_share"] = (
        profiles["outdoor_incidents"] / profiles["total_parish_incidents"]
    ).round(4)
    profiles["parish_fips"] = profiles["parish_fips"].astype(int)

    print(f"  Parishes with industry data: {len(profiles)}")
    print(f"  Mean outdoor share: {profiles['outdoor_industry_share'].mean():.3f}")
    print(f"  Dominant sectors: {profiles['dominant_sector'].value_counts().to_dict()}")

    return profiles


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("STEP 03: PROCESS OSHA SEVERE INJURY REPORTS")
    print("=" * 70)
    start_time = datetime.now()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Parse CSV
    df = parse_osha_csv(RAW_OSHA_PATH)

    # Step 2: Geocode
    df = validate_and_geocode(df, RAW_TIGER_DIR)

    # Step 3: Classify heat-related
    df = classify_heat_related(df)

    # Step 4: Classify industry
    df = classify_industry(df)

    # Step 5: Save individual records
    output_cols = [
        "ID", "EventDate", "Employer", "city", "Zip",
        "Latitude", "Longitude", "parish_fips", "parish_name",
        "Primary_NAICS", "industry_sector",
        "Hospitalized", "Amputation", "Loss_Of_Eye",
        "is_heat_related", "heat_classification",
        "Event", "EventTitle", "FinalNarrative"
    ]
    osha_out = df[output_cols].copy()
    osha_path = os.path.join(OUTPUT_DIR, "osha_processed.csv")
    osha_out.to_csv(osha_path, index=False)
    print(f"\nSaved: {osha_path}")
    print(f"  Rows: {len(osha_out)}")

    # Step 6: Aggregate to parish-day
    parish_day = aggregate_to_parish_day(df)
    parish_day_path = os.path.join(OUTPUT_DIR, "parish_day_accidents.csv")
    parish_day.to_csv(parish_day_path, index=False)
    print(f"\nSaved: {parish_day_path}")
    print(f"  Rows: {len(parish_day)}")

    # Step 7: Compute parish industry profile
    industry_profile = compute_parish_industry_profile(df)
    industry_path = os.path.join(OUTPUT_DIR, "parish_industry_profile.csv")
    industry_profile.to_csv(industry_path, index=False)
    print(f"\nSaved: {industry_path}")
    print(f"  Rows: {len(industry_profile)}")

    # Final summary
    geocoded_count = df["parish_fips"].notna().sum()
    not_geocoded = df["parish_fips"].isna().sum()
    print(f"\n--- Final Summary ---")
    print(f"  Total records parsed: {len(df)}")
    print(f"  Geocoded to parish: {geocoded_count}")
    print(f"  Not geocoded: {not_geocoded}")
    print(f"  Heat-related (Event 531): {(df['heat_classification'] == 'event_code_531').sum()}")
    print(f"  Heat-related (keyword only): {(df['heat_classification'] == 'keyword_match').sum()}")
    print(f"  Heat-related (total): {df['is_heat_related'].sum()}")
    print(f"  Unique parishes with incidents: {df.loc[df['parish_fips'].notna(), 'parish_fips'].nunique()}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
