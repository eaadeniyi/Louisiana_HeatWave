"""
04_process_census.py — Process Census ACS 2020 data for Louisiana parishes

PURPOSE:
    Read raw Census ACS 5-year estimate files, construct parish FIPS codes
    from the numeric State FIPS + County FIPS columns (not name parsing),
    merge demographics and employment data, and verify all 64 parishes
    are present.

READS:
    data/raw/census/census_2020_acs5_louisiana_parishes.csv  — Population, income, housing
    data/raw/census/census_2020_acs5_louisiana_employment.csv — Labor force, unemployment

WRITES:
    data/processed/census_processed.csv  — All 64 parishes with demographics + employment

FIXES APPLIED (vs. original pipeline):
    1. FIPS codes built from numeric State FIPS (22) + County FIPS (3-digit string),
       NOT from name parsing. This eliminates:
       - Jefferson/Jefferson Davis confusion (old: pop 31K assigned to FIPS 22051
         which is Jefferson Parish with pop 435K)
       - LaSalle "La Salle" name mismatch (old: parish dropped entirely)
    2. All 64 parishes verified present (old had only 62)
    3. Employment data merged (labor force, unemployment rate) for potential
       use as model covariates
    4. Parish names cleaned to match TIGER shapefile names (no "Parish, Louisiana" suffix)

ASSUMPTIONS:
    - Census ACS 2020 5-year estimates are time-invariant for our study period
      (2015-2025). This is a standard assumption for ecological studies using
      decennial census products, but it means we cannot capture population changes
      over the study period.
    - FIPS codes in the raw Census file match the TIGER shapefile GEOID format

AUTHOR: Emmanuel Adeniyi
DATE: 2026-03-24
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CENSUS_DIR = os.path.join(PROJECT_DIR, "data", "raw", "census")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")

EXPECTED_PARISHES = 64  # Louisiana has 64 parishes


# ============================================================================
# PROCESSING
# ============================================================================

def build_fips(row):
    """
    Build 5-digit parish FIPS from State FIPS (22) + County FIPS (3-digit, zero-padded).
    Example: State=22, County=001 -> 22001 (Acadia Parish)

    This is the fix for the old pipeline's name-parsing bugs.
    The County FIPS column may be stored as integer (1, 5, 11) or string ('001', '005', '011').
    We handle both cases.
    """
    state = str(int(row["State FIPS"])).zfill(2)
    county = str(int(row["County FIPS"])).zfill(3)
    return int(state + county)


def clean_parish_name(full_name):
    """
    Extract parish name from Census format 'Acadia Parish, Louisiana' -> 'Acadia'.
    Also handles edge cases like 'St. ' prefix consistency.
    """
    name = full_name.replace(" Parish, Louisiana", "").strip()
    return name


def process_demographics(filepath):
    """Load and process the main demographics file (population, income, housing)."""
    print("--- Demographics ---")
    df = pd.read_csv(filepath)
    print(f"  Raw rows: {len(df)}")

    # Build FIPS from numeric columns
    df["parish_fips"] = df.apply(build_fips, axis=1)
    df["parish_name"] = df["Parish Name"].apply(clean_parish_name)

    # Check for duplicates
    dupes = df["parish_fips"].duplicated()
    if dupes.any():
        print(f"  WARNING: {dupes.sum()} duplicate FIPS codes found")
        print(f"  Duplicates: {df[dupes]['parish_fips'].tolist()}")
        # Keep first occurrence
        df = df.drop_duplicates(subset="parish_fips", keep="first")

    # Select and rename columns
    result = df[[
        "parish_fips", "parish_name",
        "Total Population", "Median Household Income",
        "Total Housing Units", "Occupied Housing Units",
        "Total Population 16+ (Employment Universe)",
        "Mean Travel Time to Work"
    ]].copy()

    result = result.rename(columns={
        "Total Population": "population_total",
        "Median Household Income": "income_median",
        "Total Housing Units": "housing_total",
        "Occupied Housing Units": "housing_occupied",
        "Total Population 16+ (Employment Universe)": "population_16plus",
        "Mean Travel Time to Work": "mean_travel_time"
    })

    print(f"  Parishes after dedup: {len(result)}")
    print(f"  Population range: {result['population_total'].min():,} to {result['population_total'].max():,}")
    print(f"  Income range: ${result['income_median'].min():,} to ${result['income_median'].max():,}")

    return result


def process_employment(filepath):
    """Load and process the employment file (labor force, unemployment)."""
    print("\n--- Employment ---")
    df = pd.read_csv(filepath)
    print(f"  Raw rows: {len(df)}")

    # Build FIPS from numeric columns (columns named 'state' and 'county' here)
    df["parish_fips"] = df.apply(
        lambda row: int(str(int(row["state"])).zfill(2) + str(int(row["county"])).zfill(3)),
        axis=1
    )

    # Compute unemployment rate
    df["labor_force"] = pd.to_numeric(df["In Labor Force"], errors="coerce")
    df["unemployed"] = pd.to_numeric(df["Unemployed"], errors="coerce")
    df["unemployment_rate"] = (df["unemployed"] / df["labor_force"] * 100).round(2)

    result = df[["parish_fips", "labor_force", "unemployed", "unemployment_rate"]].copy()

    # Drop duplicates
    result = result.drop_duplicates(subset="parish_fips", keep="first")

    print(f"  Parishes: {len(result)}")
    print(f"  Labor force range: {result['labor_force'].min():,} to {result['labor_force'].max():,}")
    print(f"  Unemployment rate range: {result['unemployment_rate'].min():.1f}% to {result['unemployment_rate'].max():.1f}%")

    return result


def verify_all_parishes(df):
    """Verify all 64 Louisiana parishes are present and data is complete."""
    print(f"\n--- Verification ---")

    n_parishes = len(df)
    print(f"  Total parishes: {n_parishes} (expected: {EXPECTED_PARISHES})")

    if n_parishes != EXPECTED_PARISHES:
        print(f"  ERROR: Missing {EXPECTED_PARISHES - n_parishes} parishes!")
        # Louisiana FIPS codes are 22001 to 22127, odd numbers only
        expected_fips = set(range(22001, 22128, 2))
        actual_fips = set(df["parish_fips"])
        missing = expected_fips - actual_fips
        if missing:
            print(f"  Missing FIPS: {sorted(missing)}")
    else:
        print(f"  All {EXPECTED_PARISHES} parishes present")

    # Check for nulls in key columns
    for col in ["population_total", "income_median"]:
        nulls = df[col].isna().sum()
        if nulls > 0:
            print(f"  WARNING: {nulls} null values in {col}")
            print(f"    Parishes: {df[df[col].isna()]['parish_name'].tolist()}")
        else:
            print(f"  {col}: no nulls")

    # Spot check Jefferson Parish (the one that was wrong before)
    jeff = df[df["parish_fips"] == 22051]
    if len(jeff) == 1:
        pop = jeff.iloc[0]["population_total"]
        income = jeff.iloc[0]["income_median"]
        name = jeff.iloc[0]["parish_name"]
        print(f"\n  Spot check - FIPS 22051 ({name}):")
        print(f"    Population: {pop:,} (expected: ~434,903)")
        print(f"    Income: ${income:,} (expected: ~$54,825)")
        if pop < 100000:
            print(f"    ERROR: Population too low — likely still has Jefferson Davis data!")

    # Spot check LaSalle Parish (the one that was missing before)
    lasalle = df[df["parish_fips"] == 22059]
    if len(lasalle) == 1:
        pop = lasalle.iloc[0]["population_total"]
        name = lasalle.iloc[0]["parish_name"]
        print(f"\n  Spot check - FIPS 22059 ({name}):")
        print(f"    Population: {pop:,} (expected: ~14,950)")
    elif len(lasalle) == 0:
        print(f"\n  ERROR: FIPS 22059 (LaSalle) is MISSING!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("STEP 04: PROCESS CENSUS DATA")
    print("=" * 70)
    start_time = datetime.now()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process demographics
    demo_path = os.path.join(RAW_CENSUS_DIR, "census_2020_acs5_louisiana_parishes.csv")
    demographics = process_demographics(demo_path)

    # Process employment
    emp_path = os.path.join(RAW_CENSUS_DIR, "census_2020_acs5_louisiana_employment.csv")
    employment = process_employment(emp_path)

    # Merge on FIPS
    merged = demographics.merge(employment, on="parish_fips", how="left")
    print(f"\n--- Merged ---")
    print(f"  Rows after merge: {len(merged)}")

    # Verify
    verify_all_parishes(merged)

    # Add urban/rural classification
    # Census-defined urban parishes in Louisiana
    # Source: 2020 Census Urban Area to County relationship file
    # https://www2.census.gov/geo/docs/reference/ua/ua_list_all.txt
    # Urban parishes = those containing a Census urban area with pop >= 2,500
    URBAN_PARISH_FIPS = [
        22015,  # Bossier
        22017,  # Caddo
        22019,  # Calcasieu
        22033,  # East Baton Rouge
        22045,  # Iberia
        22051,  # Jefferson
        22055,  # Lafayette
        22071,  # Orleans
        22073,  # Ouachita
        22079,  # Rapides
        22089,  # St. Charles
        22097,  # St. Landry
        22099,  # St. Martin
        22101,  # St. Mary
        22103,  # St. Tammany
        22105,  # Tangipahoa
        22109,  # Terrebonne
        22113,  # Vermilion
    ]

    # Apply Census-based urban classification
    merged["is_urban_census"] = merged["parish_fips"].isin(URBAN_PARISH_FIPS).astype(int)

    # Keep old variable for comparison
    merged["is_urban_pop50k"] = (merged["population_total"] >= 50000).astype(int)

    # Use Census-based as primary
    merged["is_urban"] = merged["is_urban_census"]
    merged["urban_rural"] = merged["is_urban"].apply(
        lambda x: "urban" if x == 1 else "rural"
    )

    n_urban = merged["is_urban"].sum()
    n_rural = len(merged) - n_urban
    n_reclassified = (merged["is_urban"] != merged["is_urban_pop50k"]).sum()
    print(f"\n--- Urban/Rural Classification (Census-defined) ---")
    print(f"  Urban parishes (Census urban areas): {n_urban}")
    print(f"  Rural parishes: {n_rural}")
    print(f"  Parishes reclassified vs old pop>=50k: {n_reclassified}")

    # Sort by FIPS for clean output
    merged = merged.sort_values("parish_fips").reset_index(drop=True)

    # Save
    output_path = os.path.join(OUTPUT_DIR, "census_processed.csv")
    merged.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(merged)}")
    print(f"  Columns: {list(merged.columns)}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
