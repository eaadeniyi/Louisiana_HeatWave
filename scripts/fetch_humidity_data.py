#!/usr/bin/env python3
"""
Fetch ERA5 or alternative humidity data for Louisiana parishes.
Uses Open-Meteo Historical Weather API (free, no authentication required).
"""

import geopandas as gpd
import pandas as pd
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TIGER_SHAPEFILE = '/sessions/compassionate-nice-einstein/mnt/heatWave/data/raw/tiger_county/tl_2020_us_county.shp'
OUTPUT_CSV = '/sessions/compassionate-nice-einstein/mnt/heatWave/data/raw/era5/parish_daily_humidity.csv'
START_DATE = '2015-01-01'
END_DATE = '2025-07-31'
OPEN_METEO_API = 'https://archive-api.open-meteo.com/v1/archive'

# Louisiana bounds and FIPS codes
LOUISIANA_FIPS = '22'  # Louisiana state FIPS code
LOUISIANA_LAT_BOUNDS = (28.5, 33.1)
LOUISIANA_LON_BOUNDS = (-94.1, -88.8)

def extract_louisiana_parishes():
    """Extract Louisiana parish centroids from TIGER shapefile."""
    logger.info("Loading TIGER shapefile...")
    gdf = gpd.read_file(TIGER_SHAPEFILE)

    # Filter for Louisiana (FIPS code starts with 22)
    louisiana = gdf[gdf['STATEFP'] == LOUISIANA_FIPS].copy()
    logger.info(f"Found {len(louisiana)} Louisiana parishes")

    # Calculate centroids
    louisiana['longitude'] = louisiana.geometry.centroid.x
    louisiana['latitude'] = louisiana.geometry.centroid.y

    # Filter by bounds as additional check
    louisiana = louisiana[
        (louisiana['latitude'] >= LOUISIANA_LAT_BOUNDS[0]) &
        (louisiana['latitude'] <= LOUISIANA_LAT_BOUNDS[1]) &
        (louisiana['longitude'] >= LOUISIANA_LON_BOUNDS[0]) &
        (louisiana['longitude'] <= LOUISIANA_LON_BOUNDS[1])
    ]

    logger.info(f"After bounds filtering: {len(louisiana)} parishes")

    # Extract relevant columns
    parishes = louisiana[['COUNTYFP', 'NAME', 'latitude', 'longitude']].copy()
    parishes.columns = ['parish_fips', 'parish_name', 'latitude', 'longitude']
    parishes['parish_fips'] = LOUISIANA_FIPS + parishes['parish_fips'].astype(str)

    return parishes.reset_index(drop=True)

def fetch_humidity_for_location(lat, lon, start_date, end_date):
    """
    Fetch daily dewpoint and relative humidity from Open-Meteo.

    Returns:
        DataFrame with columns: date, dewpoint_2m_c, relative_humidity_pct
    """
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': 'dewpoint_2m_mean,relative_humidity_2m_mean',
        'timezone': 'UTC'
    }

    try:
        response = requests.get(OPEN_METEO_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'daily' in data:
            daily = data['daily']
            df = pd.DataFrame({
                'date': pd.to_datetime(daily['time']),
                'dewpoint_2m_c': daily['dewpoint_2m_mean'],
                'relative_humidity_pct': daily['relative_humidity_2m_mean']
            })
            return df
        else:
            logger.warning(f"No daily data in response for lat={lat}, lon={lon}")
            return None
    except requests.RequestException as e:
        logger.error(f"Request failed for lat={lat}, lon={lon}: {e}")
        return None

def main():
    logger.info("Starting humidity data fetch...")

    # Extract Louisiana parishes
    parishes = extract_louisiana_parishes()
    logger.info(f"Processing {len(parishes)} Louisiana parishes")

    # Fetch humidity data for each parish
    all_data = []

    for idx, row in parishes.iterrows():
        parish_fips = row['parish_fips']
        parish_name = row['parish_name']
        lat = row['latitude']
        lon = row['longitude']

        logger.info(f"[{idx+1}/{len(parishes)}] Fetching {parish_name} ({parish_fips}) at ({lat:.3f}, {lon:.3f})")

        humidity_df = fetch_humidity_for_location(lat, lon, START_DATE, END_DATE)

        if humidity_df is not None and len(humidity_df) > 0:
            humidity_df['parish_fips'] = parish_fips
            humidity_df['parish_name'] = parish_name
            humidity_df['source'] = 'Open-Meteo'
            all_data.append(humidity_df)
            logger.info(f"  -> Retrieved {len(humidity_df)} days")
        else:
            logger.warning(f"  -> No data retrieved")

        # Rate limiting - be respectful to the API
        time.sleep(0.5)

    # Combine all data
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result = result[['parish_fips', 'parish_name', 'date', 'dewpoint_2m_c', 'relative_humidity_pct', 'source']]
        result = result.sort_values(['parish_fips', 'date']).reset_index(drop=True)

        # Save to CSV
        Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Saved {len(result)} records to {OUTPUT_CSV}")

        # Summary statistics
        logger.info("\n=== SUMMARY ===")
        logger.info(f"Total parish-days: {len(result)}")
        logger.info(f"Number of parishes: {result['parish_fips'].nunique()}")
        logger.info(f"Date range: {result['date'].min()} to {result['date'].max()}")

        # Check coverage
        start = pd.to_datetime(START_DATE)
        end = pd.to_datetime(END_DATE)
        expected_days = (end - start).days + 1
        expected_total = expected_days * len(parishes)
        coverage_pct = (len(result) / expected_total) * 100
        logger.info(f"Expected parish-days: {expected_total}")
        logger.info(f"Coverage: {coverage_pct:.1f}%")

        # Check for gaps
        result['date_only'] = result['date'].dt.date
        by_parish = result.groupby('parish_fips')

        gaps_found = False
        for parish_fips, group in by_parish:
            dates = sorted(group['date'].unique())
            if len(dates) > 1:
                date_diffs = pd.Series(dates[1:]) - pd.Series(dates[:-1])
                max_gap = date_diffs.max()
                if max_gap > timedelta(days=1):
                    logger.warning(f"Parish {parish_fips}: Max gap {max_gap.days} days")
                    gaps_found = True

        if not gaps_found:
            logger.info("No gaps found (continuous daily coverage)")

        return result
    else:
        logger.error("No data retrieved from Open-Meteo")
        return None

if __name__ == '__main__':
    main()
