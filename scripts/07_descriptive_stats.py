"""
07_descriptive_stats.py — Compute descriptive statistics and summary tables

PURPOSE:
    Generate all descriptive tables needed for the report, notebook, and figures.
    Each output is a self-contained CSV that can be directly referenced by the
    report and notebook without recomputation.

READS:
    data/processed/master_dataset.csv   — Balanced panel (from Script 05)
    data/processed/osha_processed.csv   — Individual OSHA records (from Script 03)
    data/processed/sample_construction_stats.json — Sample counts (from Script 05)

WRITES (all in data/processed/descriptive_tables/):
    hw_vs_nonhw_comparison.csv      — Parish-day incident rates on HW vs non-HW days with CIs
    state_day_comparison.csv        — State-level daily rates on HW vs non-HW days with CIs
    industry_breakdown.csv          — Incidents by NAICS sector (total + heat-related)
    monthly_pattern.csv             — Monthly incident counts and rates
    dow_pattern.csv                 — Day-of-week incident counts and rates
    yearly_summary.csv              — Year-by-year incident counts
    parish_summary.csv              — Per-parish incident counts and weather coverage
    heat_related_detail.csv         — Heat-related incidents by classification tier
    sample_flow.csv                 — Sample construction flow (for flow diagram figure)
    industry_hw_comparison.csv      — By-industry HW vs non-HW incident comparison
    parish_hw_details.csv           — Parish-level HW classification details
    interpolation_summary.csv       — Parish interpolation/assignment summary

METHODOLOGY:
    Confidence intervals for rates computed using the normal approximation:
        rate = events / person-days
        SE = sqrt(rate * (1 - rate) / n)  for proportions
        SE = sqrt(events) / person-days   for Poisson rates (more appropriate here)
        95% CI = rate +/- 1.96 * SE

    For incident counts (Poisson), we use:
        SE(rate) = sqrt(count) / exposure
        which gives the standard Poisson CI on the rate.

AUTHOR: Emmanuel Adeniyi
DATE: 2026-03-24
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
OUTPUT_DIR = os.path.join(INPUT_DIR, "descriptive_tables")


# ============================================================================
# HELPERS
# ============================================================================

def poisson_ci(count, exposure, alpha=0.05):
    """
    Compute Poisson rate and 95% CI.
    rate = count / exposure
    SE = sqrt(count) / exposure
    CI = rate +/- z * SE
    """
    z = 1.96
    if exposure == 0:
        return np.nan, np.nan, np.nan
    rate = count / exposure
    se = np.sqrt(max(count, 1)) / exposure  # Use max(1) to avoid zero SE
    ci_lower = max(0, rate - z * se)
    ci_upper = rate + z * se
    return rate, ci_lower, ci_upper


# ============================================================================
# HW vs NON-HW COMPARISON (PARISH-DAY LEVEL)
# ============================================================================

def compute_hw_comparison(master):
    """
    Compare incident rates on heat wave vs non-heat wave parish-days.
    Restricted to complete-case sample (has weather data).
    """
    print("--- HW vs Non-HW Comparison (Parish-Day) ---")

    cc = master.dropna(subset=["tmax_f", "heat_wave_flag"]).copy()

    groups = {}
    for hw_val, label in [(1, "Heat wave"), (0, "Non-heat wave")]:
        sub = cc[cc["heat_wave_flag"] == hw_val]
        n_days = len(sub)
        total_inc = sub["total_incidents"].sum()
        heat_inc = sub["heat_related"].sum()
        total_pop_days = sub["population_total"].sum()  # person-days

        rate_total, ci_lo_t, ci_hi_t = poisson_ci(total_inc, n_days)
        rate_heat, ci_lo_h, ci_hi_h = poisson_ci(heat_inc, n_days)
        rate_percap, ci_lo_p, ci_hi_p = poisson_ci(total_inc, total_pop_days)

        groups[label] = {
            "parish_days": n_days,
            "total_incidents": int(total_inc),
            "heat_related_incidents": int(heat_inc),
            "mean_incidents_per_parish_day": round(rate_total, 6),
            "ci_lower_per_parish_day": round(ci_lo_t, 6),
            "ci_upper_per_parish_day": round(ci_hi_t, 6),
            "heat_incidents_per_parish_day": round(rate_heat, 6),
            "heat_ci_lower": round(ci_lo_h, 6),
            "heat_ci_upper": round(ci_hi_h, 6),
            "rate_per_100k_person_days": round(rate_percap * 100000, 4),
            "rate_ci_lower_100k": round(ci_lo_p * 100000, 4),
            "rate_ci_upper_100k": round(ci_hi_p * 100000, 4),
            "mean_tmax_f": round(sub["tmax_f"].mean(), 1),
            "mean_heat_index_f": round(sub["heat_index_f"].mean(), 1),
            "n_parishes": sub["parish_fips"].nunique(),
        }

    result = pd.DataFrame(groups).T
    result.index.name = "group"

    for label, vals in groups.items():
        print(f"  {label}:")
        print(f"    Parish-days: {vals['parish_days']:,}")
        print(f"    Incidents: {vals['total_incidents']} (heat: {vals['heat_related_incidents']})")
        print(f"    Rate per parish-day: {vals['mean_incidents_per_parish_day']:.4f} "
              f"[{vals['ci_lower_per_parish_day']:.4f}, {vals['ci_upper_per_parish_day']:.4f}]")
        print(f"    Rate per 100k person-days: {vals['rate_per_100k_person_days']:.4f} "
              f"[{vals['rate_ci_lower_100k']:.4f}, {vals['rate_ci_upper_100k']:.4f}]")

    return result


# ============================================================================
# STATE-DAY COMPARISON
# ============================================================================

def compute_state_day_comparison(master):
    """
    Aggregate to state-day level, then compare HW days (any parish in HW)
    vs non-HW days. This gives a different perspective than parish-day.
    """
    print("\n--- State-Day Comparison ---")

    cc = master.dropna(subset=["tmax_f", "heat_wave_flag"]).copy()

    # Aggregate to state-day
    state_day = cc.groupby("date").agg(
        total_incidents=("total_incidents", "sum"),
        heat_related=("heat_related", "sum"),
        any_hw=("heat_wave_flag", "max"),
        mean_tmax=("tmax_f", "mean"),
        total_pop=("population_total", "sum"),
        n_parishes=("parish_fips", "nunique"),
    ).reset_index()

    groups = {}
    for hw_val, label in [(1, "Any parish in HW"), (0, "No parish in HW")]:
        sub = state_day[state_day["any_hw"] == hw_val]
        n_days = len(sub)
        total_inc = sub["total_incidents"].sum()
        heat_inc = sub["heat_related"].sum()
        mean_daily = sub["total_incidents"].mean()
        std_daily = sub["total_incidents"].std()
        mean_daily_heat = sub["heat_related"].mean()
        std_daily_heat = sub["heat_related"].std()

        groups[label] = {
            "state_days": n_days,
            "total_incidents": int(total_inc),
            "heat_related": int(heat_inc),
            "mean_daily_incidents": round(mean_daily, 4),
            "sd_daily_incidents": round(std_daily, 4),
            "ci_lower": round(mean_daily - 1.96 * std_daily / np.sqrt(n_days), 4),
            "ci_upper": round(mean_daily + 1.96 * std_daily / np.sqrt(n_days), 4),
            "mean_daily_heat_incidents": round(mean_daily_heat, 4),
            "sd_daily_heat_incidents": round(std_daily_heat, 4),
            "mean_tmax_f": round(sub["mean_tmax"].mean(), 1),
        }

    # Compute percent changes (HW relative to non-HW)
    hw_vals = groups["Any parish in HW"]
    nonhw_vals = groups["No parish in HW"]
    if nonhw_vals["mean_daily_incidents"] > 0:
        pct_change_total = round(
            (hw_vals["mean_daily_incidents"] - nonhw_vals["mean_daily_incidents"])
            / nonhw_vals["mean_daily_incidents"] * 100, 2)
    else:
        pct_change_total = np.nan
    if nonhw_vals["mean_daily_heat_incidents"] > 0:
        pct_change_heat = round(
            (hw_vals["mean_daily_heat_incidents"] - nonhw_vals["mean_daily_heat_incidents"])
            / nonhw_vals["mean_daily_heat_incidents"] * 100, 2)
    else:
        pct_change_heat = np.nan

    for label in groups:
        groups[label]["pct_change_mean_daily"] = pct_change_total if label == "Any parish in HW" else np.nan
        groups[label]["pct_change_mean_daily_heat"] = pct_change_heat if label == "Any parish in HW" else np.nan

    result = pd.DataFrame(groups).T
    result.index.name = "group"

    for label, vals in groups.items():
        print(f"  {label}:")
        print(f"    Days: {vals['state_days']}")
        print(f"    Mean daily incidents: {vals['mean_daily_incidents']:.3f} "
              f"(SD={vals['sd_daily_incidents']:.3f}) "
              f"[{vals['ci_lower']:.3f}, {vals['ci_upper']:.3f}]")
    print(f"  Percent change (HW vs non-HW): {pct_change_total:.1f}% total, "
          f"{pct_change_heat:.1f}% heat-related")

    return result


# ============================================================================
# INDUSTRY BREAKDOWN
# ============================================================================

def compute_industry_breakdown(osha):
    """Incidents by NAICS sector, with heat-related subset."""
    print("\n--- Industry Breakdown ---")

    geocoded = osha[osha["parish_fips"].notna()].copy()

    breakdown = geocoded.groupby("industry_sector").agg(
        total_incidents=("ID", "count"),
        heat_related=("is_heat_related", "sum"),
        hospitalizations=("Hospitalized", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        amputations=("Amputation", lambda x: pd.to_numeric(x, errors="coerce").sum()),
    ).reset_index()

    breakdown["heat_pct"] = (breakdown["heat_related"] / breakdown["total_incidents"] * 100).round(1)
    breakdown = breakdown.sort_values("total_incidents", ascending=False).reset_index(drop=True)

    print(f"  Sectors: {len(breakdown)}")
    print(f"  Top 5:")
    for _, row in breakdown.head(5).iterrows():
        print(f"    {row['industry_sector']}: {int(row['total_incidents'])} incidents "
              f"({int(row['heat_related'])} heat, {row['heat_pct']}%)")

    return breakdown


# ============================================================================
# TEMPORAL PATTERNS
# ============================================================================

def compute_monthly_pattern(master):
    """Monthly incident counts and rates across the complete-case panel."""
    print("\n--- Monthly Pattern ---")

    cc = master.dropna(subset=["tmax_f"]).copy()

    monthly = cc.groupby("month").agg(
        parish_days=("total_incidents", "count"),
        total_incidents=("total_incidents", "sum"),
        heat_related=("heat_related", "sum"),
        mean_tmax=("tmax_f", "mean"),
        hw_days=("heat_wave_flag", "sum"),
    ).reset_index()

    monthly["incidents_per_parish_day"] = (monthly["total_incidents"] / monthly["parish_days"]).round(6)

    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    monthly["month_name"] = monthly["month"].map(month_names)

    print(f"  Monthly range: {monthly['total_incidents'].min()} to {monthly['total_incidents'].max()} incidents")
    return monthly


def compute_dow_pattern(master):
    """Day-of-week incident counts and rates."""
    print("\n--- Day-of-Week Pattern ---")

    cc = master.dropna(subset=["tmax_f"]).copy()

    dow = cc.groupby("day_of_week").agg(
        parish_days=("total_incidents", "count"),
        total_incidents=("total_incidents", "sum"),
        heat_related=("heat_related", "sum"),
    ).reset_index()

    dow["incidents_per_parish_day"] = (dow["total_incidents"] / dow["parish_days"]).round(6)

    dow_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                 4: "Friday", 5: "Saturday", 6: "Sunday"}
    dow["day_name"] = dow["day_of_week"].map(dow_names)

    for _, row in dow.iterrows():
        print(f"  {row['day_name']}: {int(row['total_incidents'])} incidents "
              f"({row['incidents_per_parish_day']:.5f} per parish-day)")

    return dow


def compute_yearly_summary(master):
    """Year-by-year summary (complete-case only, consistent with monthly/DOW tables)."""
    print("\n--- Yearly Summary ---")

    cc = master.dropna(subset=["tmax_f"]).copy()
    yearly = cc.groupby("year").agg(
        total_incidents=("total_incidents", "sum"),
        heat_related=("heat_related", "sum"),
        has_weather=("tmax_f", lambda x: x.notna().sum()),
        hw_parish_days=("heat_wave_flag", lambda x: (x == 1).sum()),
    ).reset_index()

    for _, row in yearly.iterrows():
        print(f"  {int(row['year'])}: {int(row['total_incidents'])} incidents "
              f"({int(row['heat_related'])} heat), "
              f"{int(row['hw_parish_days'])} HW parish-days")

    return yearly


# ============================================================================
# PARISH SUMMARY
# ============================================================================

def compute_parish_summary(master):
    """Per-parish summary with weather coverage and incident counts."""
    print("\n--- Parish Summary ---")

    parish = master.groupby(["parish_fips", "parish_name"]).agg(
        total_days=("date", "count"),
        days_with_weather=("tmax_f", lambda x: x.notna().sum()),
        total_incidents=("total_incidents", "sum"),
        heat_related=("heat_related", "sum"),
        hw_days=("heat_wave_flag", lambda x: (x == 1).sum()),
        population=("population_total", "first"),
    ).reset_index()

    parish["weather_coverage_pct"] = (parish["days_with_weather"] / parish["total_days"] * 100).round(1)
    parish = parish.sort_values("total_incidents", ascending=False).reset_index(drop=True)

    print(f"  Parishes: {len(parish)}")
    print(f"  With weather: {(parish['days_with_weather'] > 0).sum()}")
    print(f"  With HW days: {(parish['hw_days'] > 0).sum()}")
    print(f"  Top 5 by incidents:")
    for _, row in parish.head(5).iterrows():
        print(f"    {row['parish_name']}: {int(row['total_incidents'])} incidents, "
              f"{row['weather_coverage_pct']}% weather coverage, "
              f"{int(row['hw_days'])} HW days")

    return parish


# ============================================================================
# HEAT-RELATED DETAIL
# ============================================================================

def compute_heat_detail(osha):
    """Heat-related incidents by classification tier and industry."""
    print("\n--- Heat-Related Detail ---")

    heat = osha[(osha["is_heat_related"] == True) & (osha["parish_fips"].notna())].copy()
    print(f"  Total heat-related (geocoded): {len(heat)}")

    # By tier
    tier_counts = heat["heat_classification"].value_counts()
    print(f"  By tier: {tier_counts.to_dict()}")

    # By industry
    by_industry = heat.groupby("industry_sector").agg(
        count=("ID", "count"),
        event_531=("heat_classification", lambda x: (x == "event_code_531").sum()),
        keyword=("heat_classification", lambda x: (x == "keyword_match").sum()),
    ).sort_values("count", ascending=False).reset_index()

    return by_industry


# ============================================================================
# SAMPLE FLOW
# ============================================================================

def compute_sample_flow(master, osha):
    """
    Sample construction flow for the flow diagram figure.
    Tracks how many records at each stage.
    """
    print("\n--- Sample Construction Flow ---")

    flow = []

    # Raw OSHA
    flow.append({"stage": "Raw OSHA records parsed", "records": len(osha), "unit": "incidents"})

    # Geocoded
    geocoded = osha[osha["parish_fips"].notna()]
    flow.append({"stage": "Geocoded to parish", "records": len(geocoded), "unit": "incidents"})

    # Full panel
    flow.append({"stage": "Full panel (64 parishes x 3,862 days)", "records": len(master), "unit": "parish-days"})

    # Weather-linked
    has_weather = master["tmax_f"].notna().sum()
    flow.append({"stage": "With weather data", "records": int(has_weather), "unit": "parish-days"})

    # Complete case
    cc = master.dropna(subset=["tmax_f", "population_total", "income_median", "heat_wave_flag"])
    flow.append({"stage": "Complete case (model sample)", "records": len(cc), "unit": "parish-days"})

    # Parishes in complete case
    flow.append({"stage": "Parishes in model sample", "records": cc["parish_fips"].nunique(), "unit": "parishes"})

    # Incidents in complete case
    flow.append({"stage": "Incidents in model sample", "records": int(cc["total_incidents"].sum()), "unit": "incidents"})

    flow_df = pd.DataFrame(flow)
    for _, row in flow_df.iterrows():
        print(f"  {row['stage']}: {row['records']:,} {row['unit']}")

    return flow_df


# ============================================================================
# INDUSTRY HW vs NON-HW COMPARISON
# ============================================================================

def compute_industry_hw_comparison(master, osha):
    """
    For each industry sector, compare incident counts and rates on HW days
    vs non-HW days. Uses osha_processed incidents joined to master panel
    to tag each incident's parish-day as HW or non-HW.
    """
    print("\n--- Industry HW vs Non-HW Comparison ---")

    # Complete-case parish-days from master
    cc = master.dropna(subset=["tmax_f", "heat_wave_flag"])[
        ["parish_fips", "date", "heat_wave_flag"]
    ].copy()
    cc["date"] = pd.to_datetime(cc["date"])

    # Prepare OSHA incidents
    osha_inc = osha[osha["parish_fips"].notna()].copy()
    osha_inc["parish_fips"] = osha_inc["parish_fips"].astype(int)
    osha_inc["date"] = pd.to_datetime(osha_inc["EventDate"])
    cc["parish_fips"] = cc["parish_fips"].astype(int)

    # Merge to tag each incident with HW status
    tagged = osha_inc.merge(cc, on=["parish_fips", "date"], how="inner")
    print(f"  Incidents matched to complete-case panel: {len(tagged)}")

    # Count parish-days per HW group for rate denominators
    hw_days_total = (cc["heat_wave_flag"] == 1).sum()
    nonhw_days_total = (cc["heat_wave_flag"] == 0).sum()

    rows = []
    for sector, grp in tagged.groupby("industry_sector"):
        hw_inc = (grp["heat_wave_flag"] == 1).sum()
        nonhw_inc = (grp["heat_wave_flag"] == 0).sum()
        hw_heat = ((grp["heat_wave_flag"] == 1) & (grp["is_heat_related"] == True)).sum()
        nonhw_heat = ((grp["heat_wave_flag"] == 0) & (grp["is_heat_related"] == True)).sum()

        hw_rate = hw_inc / hw_days_total if hw_days_total > 0 else np.nan
        nonhw_rate = nonhw_inc / nonhw_days_total if nonhw_days_total > 0 else np.nan

        rows.append({
            "industry_sector": sector,
            "hw_incidents": int(hw_inc),
            "nonhw_incidents": int(nonhw_inc),
            "hw_heat_related": int(hw_heat),
            "nonhw_heat_related": int(nonhw_heat),
            "hw_rate_per_parish_day": round(hw_rate, 8),
            "nonhw_rate_per_parish_day": round(nonhw_rate, 8),
            "rate_ratio": round(hw_rate / nonhw_rate, 4) if nonhw_rate and nonhw_rate > 0 else np.nan,
        })

    result = pd.DataFrame(rows).sort_values("hw_incidents", ascending=False).reset_index(drop=True)
    print(f"  Sectors: {len(result)}")
    for _, row in result.head(5).iterrows():
        print(f"    {row['industry_sector']}: HW={row['hw_incidents']}, "
              f"non-HW={row['nonhw_incidents']}, ratio={row['rate_ratio']}")

    return result


# ============================================================================
# PARISH HW CLASSIFICATION DETAILS
# ============================================================================

def compute_parish_hw_details(master):
    """
    For each parish: total days, HW days, non-HW days, HW incidents,
    non-HW incidents, HW rate, non-HW rate, assignment_type.
    Restricted to complete-case sample.
    """
    print("\n--- Parish HW Details ---")

    cc = master.dropna(subset=["tmax_f", "heat_wave_flag"]).copy()

    rows = []
    for (fips, name), grp in cc.groupby(["parish_fips", "parish_name"]):
        total_days = len(grp)
        hw_days = (grp["heat_wave_flag"] == 1).sum()
        nonhw_days = (grp["heat_wave_flag"] == 0).sum()

        hw_sub = grp[grp["heat_wave_flag"] == 1]
        nonhw_sub = grp[grp["heat_wave_flag"] == 0]

        hw_inc = hw_sub["total_incidents"].sum()
        nonhw_inc = nonhw_sub["total_incidents"].sum()

        hw_rate = hw_inc / hw_days if hw_days > 0 else np.nan
        nonhw_rate = nonhw_inc / nonhw_days if nonhw_days > 0 else np.nan

        # Assignment type (should be constant per parish)
        atype = grp["assignment_type"].iloc[0] if "assignment_type" in grp.columns else "unknown"

        rows.append({
            "parish_fips": int(fips),
            "parish_name": name,
            "total_days": int(total_days),
            "hw_days": int(hw_days),
            "nonhw_days": int(nonhw_days),
            "hw_incidents": int(hw_inc),
            "nonhw_incidents": int(nonhw_inc),
            "hw_rate": round(hw_rate, 6) if not np.isnan(hw_rate) else np.nan,
            "nonhw_rate": round(nonhw_rate, 6) if not np.isnan(nonhw_rate) else np.nan,
            "assignment_type": atype,
        })

    result = pd.DataFrame(rows).sort_values("hw_incidents", ascending=False).reset_index(drop=True)
    print(f"  Parishes: {len(result)}")
    print(f"  Direct assignments: {(result['assignment_type'] == 'direct').sum()}")
    print(f"  Nearest-neighbor: {(result['assignment_type'] == 'nearest_neighbor').sum()}")

    return result


# ============================================================================
# INTERPOLATION SUMMARY
# ============================================================================

def compute_interpolation_summary(master):
    """
    For each parish: assignment_type, distance_km, and weather coverage stats.
    Uses parish_station_assignments.csv for distance info.
    """
    print("\n--- Interpolation Summary ---")

    assignments_path = os.path.join(INPUT_DIR, "parish_station_assignments.csv")
    assignments = pd.read_csv(assignments_path)

    # One row per parish: take the first (primary) station assignment
    parish_assign = assignments.groupby("parish_fips").first().reset_index()
    parish_assign = parish_assign[["parish_fips", "parish_name", "station_id",
                                    "assignment_type", "distance_km"]]

    # Weather coverage from master
    coverage = master.groupby("parish_fips").agg(
        total_days=("date", "count"),
        days_with_weather=("tmax_f", lambda x: x.notna().sum()),
        days_with_hw_flag=("heat_wave_flag", lambda x: x.notna().sum()),
        mean_tmax=("tmax_f", "mean"),
    ).reset_index()

    coverage["weather_coverage_pct"] = (
        coverage["days_with_weather"] / coverage["total_days"] * 100
    ).round(1)
    coverage["hw_flag_coverage_pct"] = (
        coverage["days_with_hw_flag"] / coverage["total_days"] * 100
    ).round(1)

    # Merge
    result = parish_assign.merge(coverage, on="parish_fips", how="left")
    result = result.sort_values("distance_km", ascending=False).reset_index(drop=True)

    print(f"  Parishes: {len(result)}")
    print(f"  Direct: {(result['assignment_type'] == 'direct').sum()}, "
          f"Nearest-neighbor: {(result['assignment_type'] == 'nearest_neighbor').sum()}")
    print(f"  Mean distance: {result['distance_km'].mean():.1f} km")
    print(f"  Mean weather coverage: {result['weather_coverage_pct'].mean():.1f}%")

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("STEP 07: DESCRIPTIVE STATISTICS")
    print("=" * 70)
    start_time = datetime.now()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    master = pd.read_csv(os.path.join(INPUT_DIR, "master_dataset.csv"), parse_dates=["date"])
    osha = pd.read_csv(os.path.join(INPUT_DIR, "osha_processed.csv"))
    print(f"Master: {len(master):,} rows")
    print(f"OSHA: {len(osha):,} rows\n")

    # Compute all tables
    tables = {}

    tables["hw_vs_nonhw_comparison"] = compute_hw_comparison(master)
    tables["state_day_comparison"] = compute_state_day_comparison(master)
    tables["industry_breakdown"] = compute_industry_breakdown(osha)
    tables["monthly_pattern"] = compute_monthly_pattern(master)
    tables["dow_pattern"] = compute_dow_pattern(master)
    tables["yearly_summary"] = compute_yearly_summary(master)
    tables["parish_summary"] = compute_parish_summary(master)
    tables["heat_related_detail"] = compute_heat_detail(osha)
    tables["sample_flow"] = compute_sample_flow(master, osha)
    tables["industry_hw_comparison"] = compute_industry_hw_comparison(master, osha)
    tables["parish_hw_details"] = compute_parish_hw_details(master)
    tables["interpolation_summary"] = compute_interpolation_summary(master)

    # Save all
    index_tables = {"hw_vs_nonhw_comparison", "state_day_comparison"}
    print("\n--- Saving Tables ---")
    for name, df in tables.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df.to_csv(path, index=True if name in index_tables else False)
        print(f"  {name}.csv: {len(df)} rows")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
