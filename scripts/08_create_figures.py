"""
08_create_figures.py — Generate publication-quality figures (v3 — rebuilt pipeline)

PURPOSE:
    Create all 13 figures for the report at 300 DPI using the rebuilt data
    pipeline with ERA5 humidity, expanded stations, and ACS employment data.

READS:
    data/processed/descriptive_tables/*.csv
    data/processed/primary_model_coefficients.csv
    data/processed/quintile_model_results.csv
    data/processed/exploratory_hw_model_coefficients.csv
    data/processed/lag_model_coefficients.csv
    data/processed/lag_hw_model_coefficients.csv
    data/processed/robustness_results.json
    data/processed/model_diagnostics.json
    data/processed/sample_construction_stats.json
    data/processed/master_dataset.csv
    data/processed/parish_station_assignments.csv
    data/processed/descriptive_tables/interpolation_summary.csv
    data/raw/tiger_county/tl_2020_us_county.shp

WRITES (all in figures/):
    fig1_sample_flow.png            — Sample construction flow diagram
    fig2_parish_map.png             — Parish coverage map (3-panel)
    fig3_hw_comparison.png          — HW vs non-HW incident rates with CIs
    fig4_forest_plot.png            — Poisson IRR forest plot (all models)
    fig5_industry_breakdown.png     — Total + heat-related by sector
    fig6_temporal_patterns.png      — Monthly + DOW patterns (2-panel)
    fig7_yearly_trends.png          — Year-by-year incidents and HW days
    fig8_hw_by_year.png             — Heat wave parish-days by year and month
    fig9_station_coverage.png       — Weather station coverage map
    fig10_interpolation_accuracy.png — Interpolation quality assessment
    fig11_accidents_by_parish.png   — Top 20 parishes by accident count
    fig12_accidents_by_industry_parish.png — Industry x parish breakdown
    fig13_temperature_trends.png    — Temperature trends over time

AUTHOR: Emmanuel Adeniyi
DATE: 2026-03-26
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import ScalarFormatter
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(PROJECT_DIR, "data", "processed", "descriptive_tables")
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "processed")
SHAPE_PATH = os.path.join(PROJECT_DIR, "data", "raw", "tiger_county", "tl_2020_us_county.shp")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")

DPI = 300
FONT_SIZE = 10
TITLE_SIZE = 12

# Color palette — high-contrast, print-friendly
C = {
    "primary": "#2C3E50",
    "accent": "#E74C3C",
    "secondary": "#3498DB",
    "tertiary": "#1ABC9C",
    "green": "#27AE60",
    "light_gray": "#ECF0F1",
    "dark_gray": "#7F8C8D",
    "hw": "#E74C3C",
    "non_hw": "#3498DB",
    "heat": "#E67E22",
    "direct_station": "#1B4F72",   # dark steel blue
    "nn_station": "#F39C12",       # bright amber
}

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================================
# HELPER: load model CSV with consistent column naming
# ============================================================================
def load_model(path):
    df = pd.read_csv(path)
    if df.columns[0] == "Unnamed: 0":
        df = df.rename(columns={"Unnamed: 0": "Variable"})
    return df

def get_var(df, varname):
    """Get a single row from model dataframe by variable name."""
    row = df[df["Variable"] == varname]
    if len(row) == 0:
        return None
    return row.iloc[0]


# ============================================================================
# FIGURE 1: SAMPLE CONSTRUCTION FLOW DIAGRAM
# ============================================================================
def fig1_sample_flow():
    print("  Creating Figure 1: Sample Flow...")

    with open(os.path.join(MODEL_DIR, "sample_construction_stats.json")) as f:
        stats = json.load(f)

    # Define flow steps
    steps = [
        ("Raw OSHA Severe Injury Reports\n(Louisiana, 2015–2025)", stats["total_osha_incidents"], "incidents"),
        ("Geocoded to Louisiana Parishes\n(TIGER/Line spatial join)", stats["total_osha_incidents"], "incidents"),
        ("Full Parish-Day Panel\n(64 parishes × 3,862 days)", stats["full_panel_rows"], "parish-days"),
        ("Weather-Linked Sample\n(with GHCN + ERA5 data)", stats["weather_linked_rows"], "parish-days"),
        (f"Complete-Case Model Sample\n({stats['complete_case_parishes']} parishes)", stats["complete_case_rows"], "parish-days"),
    ]

    side_notes = [
        None,
        None,
        None,
        f"−{stats['full_panel_rows'] - stats['weather_linked_rows']:,} parish-days\n(missing weather data)",
        None,
    ]

    box_colors = ["#3498DB", "#2980B9", "#1ABC9C", "#16A085", "#27AE60"]

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, len(steps) + 0.5)
    ax.axis("off")

    box_w = 7
    box_h = 0.8
    x_center = 5

    for i, (label, count, unit) in enumerate(steps):
        y = len(steps) - i - 0.5
        color = box_colors[i]

        box = FancyBboxPatch(
            (x_center - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9,
        )
        ax.add_patch(box)

        text = f"{label}\nN = {count:,} {unit}"
        ax.text(x_center, y, text, ha="center", va="center",
                fontsize=FONT_SIZE, fontweight="bold", color="white")

        # Arrow between boxes
        if i < len(steps) - 1:
            ax.annotate("", xy=(x_center, y - box_h / 2 - 0.05),
                        xytext=(x_center, y - box_h / 2 - 0.3),
                        arrowprops=dict(arrowstyle="->", color=C["dark_gray"], lw=1.5))

        # Side notes
        if side_notes[i]:
            ax.text(x_center + box_w / 2 + 0.3, y, side_notes[i],
                    ha="left", va="center", fontsize=FONT_SIZE - 2,
                    color=C["accent"], style="italic")

    # Bottom summary
    summary = (f"Model sample: {stats['complete_case_incidents']:,} incidents total, "
               f"{stats.get('complete_case_heat_related', 81)} heat-related\n"
               f"{stats['hw_parish_days_complete']:,} heat-wave parish-days across "
               f"{stats['hw_parishes_in_complete']} parishes\n"
               f"{stats['incidents_during_hw_complete']} incidents during heat-wave days")
    ax.text(x_center, -0.3, summary, ha="center", va="top",
            fontsize=FONT_SIZE - 1, color=C["primary"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["light_gray"], edgecolor=C["dark_gray"]))

    ax.set_title("Sample Construction",
                 fontsize=TITLE_SIZE + 2, fontweight="bold", pad=15)

    fig.savefig(os.path.join(FIG_DIR, "fig1_sample_flow.png"))
    plt.close(fig)
    print("    -> fig1_sample_flow.png")


# ============================================================================
# FIGURE 2: PARISH COVERAGE MAP (3-panel)
# ============================================================================
def fig2_parish_map():
    print("  Creating Figure 2: Parish Map...")

    try:
        import geopandas as gpd
    except ImportError:
        print("    SKIP: geopandas not available")
        return

    interp = pd.read_csv(os.path.join(TABLE_DIR, "interpolation_summary.csv"))
    interp["parish_fips"] = interp["parish_fips"].astype(str).str.zfill(5)

    parish = pd.read_csv(os.path.join(TABLE_DIR, "parish_summary.csv"))

    gdf = gpd.read_file(SHAPE_PATH)
    la = gdf[gdf["STATEFP"] == "22"].copy()
    la = la.merge(interp[["parish_fips", "assignment_type", "weather_coverage_pct", "days_with_hw_flag"]],
                  left_on="GEOID", right_on="parish_fips", how="left")

    # Merge parish summary for hw days and incidents
    parish_fips_map = {}
    for _, r in interp.iterrows():
        parish_fips_map[r["parish_name"]] = r["parish_fips"]
    parish["parish_fips"] = parish["parish_fips"].astype(str).str.zfill(5)
    la = la.merge(parish[["parish_fips", "hw_days", "total_incidents"]],
                  left_on="GEOID", right_on="parish_fips", how="left", suffixes=("", "_parish"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Weather coverage
    ax = axes[0]
    la.plot(column="weather_coverage_pct", cmap="YlOrRd", linewidth=0.5,
            edgecolor="gray", legend=True, ax=ax, missing_kwds={"color": "lightgray"},
            legend_kwds={"label": "Weather coverage (%)", "shrink": 0.6})
    ax.set_title("A. Weather Station Coverage", fontweight="bold")
    ax.axis("off")

    # Panel B: Heat wave days
    ax = axes[1]
    la["HW_Days_fill"] = la["hw_days"].fillna(0)
    la.plot(column="HW_Days_fill", cmap="OrRd", linewidth=0.5,
            edgecolor="gray", legend=True, ax=ax, missing_kwds={"color": "lightgray"},
            legend_kwds={"label": "Heat wave days", "shrink": 0.6})
    ax.set_title("B. Heat Wave Parish-Days", fontweight="bold")
    ax.axis("off")

    # Panel C: Station assignment type
    ax = axes[2]
    la["assign_color"] = la["assignment_type"].map({
        "direct": C["direct_station"],
        "nearest_neighbor": C["nn_station"]
    }).fillna(C["light_gray"])
    la.plot(color=la["assign_color"], linewidth=0.5, edgecolor="gray", ax=ax)
    ax.set_title("C. Station Assignment", fontweight="bold")
    ax.axis("off")
    legend_elements = [
        mpatches.Patch(facecolor=C["direct_station"], edgecolor="gray", label=f"Direct station (n=37)"),
        mpatches.Patch(facecolor=C["nn_station"], edgecolor="gray", label=f"Nearest neighbor (n=34)"),
    ]
    # Count from actual data
    n_direct = (la["assignment_type"] == "direct").sum()
    n_nn = (la["assignment_type"] == "nearest_neighbor").sum()
    legend_elements = [
        mpatches.Patch(facecolor=C["direct_station"], edgecolor="gray", label=f"Direct station (n={n_direct})"),
        mpatches.Patch(facecolor=C["nn_station"], edgecolor="gray", label=f"Nearest neighbor (n={n_nn})"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=FONT_SIZE - 1)

    fig.suptitle("Louisiana Parish Coverage and Station Assignment",
                 fontsize=TITLE_SIZE + 2, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_parish_map.png"))
    plt.close(fig)
    print("    -> fig2_parish_map.png")


# ============================================================================
# FIGURE 3: HW VS NON-HW COMPARISON
# ============================================================================
def fig3_hw_comparison():
    print("  Creating Figure 3: HW Comparison...")

    hw = pd.read_csv(os.path.join(TABLE_DIR, "hw_vs_nonhw_comparison.csv"))

    hw_row = hw[hw["group"] == "Heat wave"].iloc[0]
    non_row = hw[hw["group"] == "Non-heat wave"].iloc[0]

    # Compute Poisson CIs for rates
    def poisson_ci(k, n, z=1.96):
        rate = k / n
        se = np.sqrt(k) / n
        return rate, max(0, rate - z * se), rate + z * se

    hw_rate, hw_lo, hw_hi = poisson_ci(hw_row["total_incidents"], hw_row["parish_days"])
    non_rate, non_lo, non_hi = poisson_ci(non_row["total_incidents"], non_row["parish_days"])

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar([0, 1], [hw_rate, non_rate],
                  color=[C["hw"], C["non_hw"]],
                  width=0.5, edgecolor="white", linewidth=1.5)
    ax.errorbar([0, 1], [hw_rate, non_rate],
                yerr=[[hw_rate - hw_lo, non_rate - non_lo],
                      [hw_hi - hw_rate, non_hi - non_rate]],
                fmt="none", color="black", capsize=8, capthick=1.5, linewidth=1.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Heat Wave\nDays", "Non-Heat Wave\nDays"])
    ax.set_ylabel("Incidents per Parish-Day")
    ax.set_title("Crude Incident Rates: Heat Wave vs. Non-Heat Wave Days",
                 fontweight="bold", fontsize=TITLE_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotations
    for i, (rate, lo, hi, n, inc) in enumerate([
        (hw_rate, hw_lo, hw_hi, int(hw_row["parish_days"]), int(hw_row["total_incidents"])),
        (non_rate, non_lo, non_hi, int(non_row["parish_days"]), int(non_row["total_incidents"]))
    ]):
        ax.text(i, hi + 0.0005,
                f"n = {n:,} parish-days\n{inc} incidents",
                ha="center", fontsize=FONT_SIZE - 1, color=C["primary"])

    # Rate ratio annotation
    rr = hw_rate / non_rate
    ax.text(0.5, max(hw_hi, non_hi) + 0.003,
            f"Crude rate ratio: {rr:.2f}",
            ha="center", fontsize=FONT_SIZE, fontweight="bold", color=C["primary"],
            transform=ax.get_xaxis_transform())

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_hw_comparison.png"))
    plt.close(fig)
    print("    -> fig3_hw_comparison.png")


# ============================================================================
# FIGURE 4: FOREST PLOT (all models)
# ============================================================================
def fig4_forest_plot():
    print("  Creating Figure 4: Forest Plot...")

    m1 = load_model(os.path.join(MODEL_DIR, "primary_model_coefficients.csv"))
    m2 = load_model(os.path.join(MODEL_DIR, "quintile_model_results.csv"))
    m3 = load_model(os.path.join(MODEL_DIR, "exploratory_hw_model_coefficients.csv"))

    with open(os.path.join(MODEL_DIR, "robustness_results.json")) as f:
        robust = json.load(f)

    rows = []

    # Model 1: continuous tmax
    r = get_var(m1, "tmax_std")
    if r is not None:
        rows.append({"label": "Tmax (per SD) — Model 1", "irr": r["IRR"],
                      "lo": r["IRR_CI_Lower"], "hi": r["IRR_CI_Upper"],
                      "p": r["P_value"], "group": "Primary"})

    # Model 2: quintiles
    for qname in ["Q2", "Q3", "Q4", "Q5"]:
        r = get_var(m2, qname)
        if r is not None:
            rows.append({"label": f"{qname} vs Q1 — Model 2", "irr": r["IRR"],
                          "lo": r["IRR_CI_Lower"], "hi": r["IRR_CI_Upper"],
                          "p": r["P_value"], "group": "Quintiles"})

    # Model 3: heat wave flag
    r = get_var(m3, "heat_wave_flag")
    if r is not None:
        rows.append({"label": "Heat Wave — Model 3", "irr": r["IRR"],
                      "lo": r["IRR_CI_Lower"], "hi": r["IRR_CI_Upper"],
                      "p": r["P_value"], "group": "Heat Wave"})

    # Robustness: Model 4a (parish FE)
    if "model_4a_parish_fe" in robust:
        rb = robust["model_4a_parish_fe"]
        rows.append({"label": "HW + Parish FE — Model 4a", "irr": rb["hw_irr"],
                      "lo": rb["hw_ci_lower"], "hi": rb["hw_ci_upper"],
                      "p": rb["hw_pvalue"], "group": "Robustness"})

    # Robustness: Model 4b (labor force offset)
    if "model_4b_labor_offset" in robust:
        rb = robust["model_4b_labor_offset"]
        rows.append({"label": "HW + Labor Offset — Model 4b", "irr": rb["hw_irr"],
                      "lo": rb["hw_ci_lower"], "hi": rb["hw_ci_upper"],
                      "p": rb["hw_pvalue"], "group": "Robustness"})

    # Robustness: Model 4c (tmax only, no humidity)
    if "model_4c_tmax_only" in robust:
        rb = robust["model_4c_tmax_only"]
        rows.append({"label": "Tmax Only — Model 4c", "irr": rb["tmax_irr"],
                      "lo": rb["tmax_ci_lower"], "hi": rb["tmax_ci_upper"],
                      "p": rb["tmax_pvalue"], "group": "Robustness"})

    df = pd.DataFrame(rows)

    group_colors = {
        "Primary": C["secondary"],
        "Quintiles": C["tertiary"],
        "Heat Wave": C["accent"],
        "Lag": "#8E44AD",
        "Robustness": C["dark_gray"],
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = list(range(len(df)))
    for i, row in df.iterrows():
        color = group_colors.get(row["group"], C["primary"])
        ax.errorbar(row["irr"], i, xerr=[[row["irr"] - row["lo"]], [row["hi"] - row["irr"]]],
                    fmt="o", color=color, markersize=7, capsize=5, capthick=1.5,
                    linewidth=1.5, markeredgecolor="white", markeredgewidth=0.5)

        # P-value annotation
        p_str = f"p={row['p']:.3f}" if row["p"] >= 0.001 else "p<0.001"
        if row["p"] < 0.10:
            p_str += " *" if row["p"] < 0.10 else ""
        ax.text(max(row["hi"] + 0.02, 1.55), i, p_str, va="center",
                fontsize=FONT_SIZE - 2, color=C["dark_gray"])

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"], fontsize=FONT_SIZE - 1)
    ax.set_xlabel("Incidence Rate Ratio (IRR)")
    ax.set_title("Poisson Regression — Incidence Rate Ratios\n"
                 "(All models: year/month/DOW FE, log(pop) offset, parish-clustered SEs)",
                 fontweight="bold", fontsize=TITLE_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set reasonable x limits
    ax.set_xlim(0.5, 2.0)

    # Legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                           markersize=8, label=g)
               for g, c in group_colors.items() if g in df["group"].values]
    ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE - 1)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_forest_plot.png"))
    plt.close(fig)
    print("    -> fig4_forest_plot.png")


# ============================================================================
# FIGURE 5: INDUSTRY BREAKDOWN
# ============================================================================
def fig5_industry():
    print("  Creating Figure 5: Industry Breakdown...")

    ind = pd.read_csv(os.path.join(TABLE_DIR, "industry_breakdown.csv"))
    ind = ind.sort_values("total_incidents", ascending=True)

    # Get heat-related by industry from industry_hw_comparison
    ind_hw = pd.read_csv(os.path.join(TABLE_DIR, "industry_hw_comparison.csv"))

    fig, ax = plt.subplots(figsize=(10, 6))

    y = range(len(ind))
    ax.barh(y, ind["total_incidents"].values, color=C["secondary"],
            label="Total Incidents", height=0.6, alpha=0.8)

    # Overlay HW incidents if available
    if "hw_incidents" in ind_hw.columns:
        hw_by_sector = ind_hw.set_index("industry_sector")["hw_incidents"].reindex(ind["industry_sector"].values).fillna(0).values
        ax.barh(y, hw_by_sector, color=C["accent"],
                label="During Heat Waves", height=0.6, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(ind["industry_sector"].values, fontsize=FONT_SIZE - 1)
    ax.set_xlabel("Number of Severe Injuries (2015–2025)")
    ax.set_title("Severe OSHA-Reportable Injuries by Industry Sector",
                 fontweight="bold", fontsize=TITLE_SIZE + 1)
    ax.legend(loc="lower right", fontsize=FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate totals
    for i, (_, row) in enumerate(ind.iterrows()):
        ax.text(row["total_incidents"] + 5, i, str(row["total_incidents"]),
                va="center", fontsize=FONT_SIZE - 2, color=C["primary"])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_industry_breakdown.png"))
    plt.close(fig)
    print("    -> fig5_industry_breakdown.png")


# ============================================================================
# FIGURE 6: TEMPORAL PATTERNS (Monthly + DOW)
# ============================================================================
def fig6_temporal():
    print("  Creating Figure 6: Temporal Patterns...")

    monthly = pd.read_csv(os.path.join(TABLE_DIR, "monthly_pattern.csv"))
    dow = pd.read_csv(os.path.join(TABLE_DIR, "dow_pattern.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Monthly
    ax = axes[0]
    ax2 = ax.twinx()
    months = monthly["month_name"]
    x = range(len(months))

    ax.bar(x, monthly["total_incidents"], color=C["secondary"],
           alpha=0.7, label="Total Incidents", width=0.6)
    ax2.plot(x, monthly["mean_tmax"], color=C["accent"],
             linewidth=2, marker="o", markersize=5, label="Mean Tmax (°F)")

    # Shade HW months
    hw_months = monthly["hw_days"].values
    for i, hw_d in enumerate(hw_months):
        if hw_d > 0:
            ax.axvspan(i - 0.3, i + 0.3, alpha=0.08, color=C["hw"])

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("Number of Incidents")
    ax2.set_ylabel("Mean Tmax (°F)", color=C["accent"])
    ax.set_title("A. Monthly Pattern", fontweight="bold")
    ax.spines["top"].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=FONT_SIZE - 2)

    # Panel B: Day of Week
    ax = axes[1]
    days = dow["day_name"]
    x = range(len(days))

    ax.bar(x, dow["incidents_per_parish_day"], color=C["secondary"], alpha=0.8, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, ha="right")
    ax.set_ylabel("Incidents per Parish-Day")
    ax.set_title("B. Day-of-Week Pattern", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add weekday vs weekend reference
    weekday_mean = dow.iloc[:5]["incidents_per_parish_day"].mean()
    ax.axhline(y=weekday_mean, color=C["dark_gray"], linestyle=":", linewidth=1, alpha=0.7)
    ax.text(6.5, weekday_mean, f"Weekday avg: {weekday_mean:.4f}",
            fontsize=FONT_SIZE - 2, color=C["dark_gray"], ha="right")

    fig.suptitle("Temporal Distribution of Workplace Injuries",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_temporal_patterns.png"))
    plt.close(fig)
    print("    -> fig6_temporal_patterns.png")


# ============================================================================
# FIGURE 7: YEARLY TRENDS
# ============================================================================
def fig7_yearly():
    print("  Creating Figure 7: Yearly Trends...")

    yearly = pd.read_csv(os.path.join(TABLE_DIR, "yearly_summary.csv"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    x = yearly["year"]
    ax.bar(x - 0.15, yearly["total_incidents"], width=0.3,
           color=C["secondary"], alpha=0.8, label="Total Incidents")
    ax2.plot(x, yearly["hw_parish_days"], color=C["hw"], linewidth=2.5,
             marker="s", markersize=6, label="Heat Wave Parish-Days")

    ax.set_xlabel("Year")
    ax.set_ylabel("Total Incidents")
    ax2.set_ylabel("Heat Wave Parish-Days", color=C["hw"])
    ax.set_title("Year-by-Year Incident Counts and Heat Wave Exposure",
                 fontweight="bold", fontsize=TITLE_SIZE)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=FONT_SIZE - 1)

    ax.spines["top"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(x.astype(int), rotation=45)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_yearly_trends.png"))
    plt.close(fig)
    print("    -> fig7_yearly_trends.png")


# ============================================================================
# FIGURE 8: HW BY YEAR AND MONTH (heatmap-style)
# ============================================================================
def fig8_hw_by_year():
    print("  Creating Figure 8: HW by Year...")

    master = pd.read_csv(os.path.join(MODEL_DIR, "master_dataset.csv"),
                         usecols=["date", "heat_wave_flag"])
    master["date"] = pd.to_datetime(master["date"])
    master["year"] = master["date"].dt.year
    master["month"] = master["date"].dt.month

    hw_grid = master.groupby(["year", "month"])["heat_wave_flag"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(hw_grid.values, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(hw_grid.index)))
    ax.set_yticklabels(hw_grid.index.astype(int))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Heat Wave Parish-Days")

    ax.set_title("Heat Wave Parish-Days by Year and Month",
                 fontweight="bold", fontsize=TITLE_SIZE)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig8_hw_by_year.png"))
    plt.close(fig)
    print("    -> fig8_hw_by_year.png")


# ============================================================================
# FIGURE 9: STATION COVERAGE MAP (high-contrast colors)
# ============================================================================
def fig9_station_map():
    print("  Creating Figure 9: Station Coverage Map...")

    try:
        import geopandas as gpd
    except ImportError:
        print("    SKIP: geopandas not available")
        return

    stations = pd.read_csv(os.path.join(MODEL_DIR, "parish_station_assignments.csv"))
    interp = pd.read_csv(os.path.join(TABLE_DIR, "interpolation_summary.csv"))
    interp["parish_fips"] = interp["parish_fips"].astype(str).str.zfill(5)

    gdf = gpd.read_file(SHAPE_PATH)
    la = gdf[gdf["STATEFP"] == "22"].copy()
    la = la.merge(interp[["parish_fips", "assignment_type", "distance_km", "weather_coverage_pct"]],
                  left_on="GEOID", right_on="parish_fips", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Assignment type
    ax = axes[0]
    la["assign_color"] = la["assignment_type"].map({
        "direct": C["direct_station"],
        "nearest_neighbor": C["nn_station"]
    }).fillna(C["light_gray"])
    la.plot(color=la["assign_color"], linewidth=0.5, edgecolor="gray", ax=ax)
    ax.set_title("A. Station Assignment Type", fontweight="bold")
    ax.axis("off")

    n_direct = (la["assignment_type"] == "direct").sum()
    n_nn = (la["assignment_type"] == "nearest_neighbor").sum()
    legend_elements = [
        mpatches.Patch(facecolor=C["direct_station"], edgecolor="gray",
                       label=f"Direct station (n={n_direct})"),
        mpatches.Patch(facecolor=C["nn_station"], edgecolor="gray",
                       label=f"Nearest neighbor (n={n_nn})"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=FONT_SIZE - 1)

    # Panel B: NN distance
    ax = axes[1]
    nn_data = la[la["assignment_type"] == "nearest_neighbor"].copy()
    la_bg = la.copy()
    la_bg.plot(color=C["light_gray"], linewidth=0.5, edgecolor="gray", ax=ax)
    if len(nn_data) > 0:
        nn_data.plot(column="distance_km", cmap="YlOrRd", linewidth=0.5,
                     edgecolor="gray", legend=True, ax=ax,
                     legend_kwds={"label": "Distance to nearest station (km)", "shrink": 0.6})
    ax.set_title("B. Nearest-Neighbor Distance (km)", fontweight="bold")
    ax.axis("off")

    fig.suptitle("Weather Station Coverage and Parish Assignment",
                 fontsize=TITLE_SIZE + 2, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig9_station_coverage.png"))
    plt.close(fig)
    print("    -> fig9_station_coverage.png")


# ============================================================================
# FIGURE 10: INTERPOLATION ACCURACY
# ============================================================================
def fig10_interpolation():
    print("  Creating Figure 10: Interpolation Accuracy...")

    interp = pd.read_csv(os.path.join(TABLE_DIR, "interpolation_summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Coverage % by assignment type
    ax = axes[0]
    direct = interp[interp["assignment_type"] == "direct"]["weather_coverage_pct"]
    nn = interp[interp["assignment_type"] == "nearest_neighbor"]["weather_coverage_pct"]

    bp = ax.boxplot([direct.values, nn.values], labels=["Direct Station", "Nearest Neighbor"],
                    patch_artist=True, widths=0.4,
                    boxprops=dict(facecolor=C["light_gray"]),
                    medianprops=dict(color=C["accent"], linewidth=2))
    bp["boxes"][0].set_facecolor(C["direct_station"])
    bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][1].set_facecolor(C["nn_station"])
    bp["boxes"][1].set_alpha(0.4)

    ax.set_ylabel("Weather Coverage (%)")
    ax.set_title("A. Coverage by Assignment Type", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Distance vs coverage for NN parishes
    ax = axes[1]
    nn_data = interp[interp["assignment_type"] == "nearest_neighbor"]
    ax.scatter(nn_data["distance_km"], nn_data["weather_coverage_pct"],
               color=C["nn_station"], alpha=0.7, edgecolors="gray", s=50)
    ax.set_xlabel("Distance to Nearest Station (km)")
    ax.set_ylabel("Weather Coverage (%)")
    ax.set_title("B. Distance vs. Coverage (NN Parishes)", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Trend line
    if len(nn_data) > 2:
        z = np.polyfit(nn_data["distance_km"], nn_data["weather_coverage_pct"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(nn_data["distance_km"].min(), nn_data["distance_km"].max(), 50)
        ax.plot(x_line, p(x_line), color=C["dark_gray"], linestyle="--", linewidth=1)

    fig.suptitle("Weather Data Quality Assessment",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig10_interpolation_accuracy.png"))
    plt.close(fig)
    print("    -> fig10_interpolation_accuracy.png")


# ============================================================================
# FIGURE 11: TOP 20 PARISHES BY ACCIDENT COUNT
# ============================================================================
def fig11_parishes():
    print("  Creating Figure 11: Accidents by Parish...")

    parish = pd.read_csv(os.path.join(TABLE_DIR, "parish_summary.csv"))
    top20 = parish.nlargest(20, "total_incidents").sort_values("total_incidents", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    y = range(len(top20))
    ax.barh(y, top20["total_incidents"].values, color=C["secondary"], alpha=0.8, height=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(top20["parish_name"].values, fontsize=FONT_SIZE - 1)
    ax.set_xlabel("Total Severe Injuries (2015–2025)")
    ax.set_title("Top 20 Parishes by Severe Injury Count",
                 fontweight="bold", fontsize=TITLE_SIZE + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (_, row) in enumerate(top20.iterrows()):
        pop = row["population"]
        ax.text(row["total_incidents"] + 2, i,
                f"pop: {pop:,.0f}" if pd.notna(pop) else "",
                va="center", fontsize=FONT_SIZE - 2, color=C["dark_gray"])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig11_accidents_by_parish.png"))
    plt.close(fig)
    print("    -> fig11_accidents_by_parish.png")


# ============================================================================
# FIGURE 12: INDUSTRY × PARISH BREAKDOWN
# ============================================================================
def fig12_industry_parish():
    print("  Creating Figure 12: Industry x Parish...")

    # Load OSHA data to get industry x parish cross-tab
    osha = pd.read_csv(os.path.join(MODEL_DIR, "osha_processed.csv"), low_memory=False)

    # Top industries and parishes
    # Determine the industry sector column name
    sector_col = "NAICS_sector" if "NAICS_sector" in osha.columns else "industry_sector"
    top_ind = osha[sector_col].value_counts().head(6).index.tolist()
    top_par = osha["parish_name"].value_counts().head(10).index.tolist()

    subset = osha[osha[sector_col].isin(top_ind) & osha["parish_name"].isin(top_par)]
    ct = pd.crosstab(subset["parish_name"], subset[sector_col])
    ct = ct.reindex(index=top_par, columns=top_ind).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(ct.values, cmap="YlOrBr", aspect="auto")

    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right", fontsize=FONT_SIZE - 1)
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index, fontsize=FONT_SIZE - 1)

    # Annotate cells
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            val = int(ct.iloc[i, j])
            if val > 0:
                color = "white" if val > ct.values.max() * 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=FONT_SIZE - 2, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Number of Incidents")

    ax.set_title("Severe Injuries by Industry and Parish (Top 10 x Top 6)",
                 fontweight="bold", fontsize=TITLE_SIZE)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig12_accidents_by_industry_parish.png"))
    plt.close(fig)
    print("    -> fig12_accidents_by_industry_parish.png")


# ============================================================================
# FIGURE 13: TEMPERATURE TRENDS
# ============================================================================
def fig13_temperature():
    print("  Creating Figure 13: Temperature Trends...")

    master = pd.read_csv(os.path.join(MODEL_DIR, "master_dataset.csv"),
                         usecols=["date", "tmax_f", "heat_index_f", "heat_wave_flag"])
    master["date"] = pd.to_datetime(master["date"])
    master["year_month"] = master["date"].dt.to_period("M")

    monthly = master.groupby("year_month").agg(
        mean_tmax=("tmax_f", "mean"),
        mean_hi=("heat_index_f", "mean"),
        hw_days=("heat_wave_flag", "sum")
    ).reset_index()
    monthly["date_plot"] = monthly["year_month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax2 = ax.twinx()

    ax.plot(monthly["date_plot"], monthly["mean_tmax"], color=C["secondary"],
            linewidth=1.5, alpha=0.8, label="Mean Tmax (°F)")
    ax.plot(monthly["date_plot"], monthly["mean_hi"], color=C["heat"],
            linewidth=1.5, alpha=0.8, label="Mean Heat Index (°F)")

    ax2.fill_between(monthly["date_plot"], 0, monthly["hw_days"],
                     color=C["hw"], alpha=0.2, label="HW Parish-Days")

    ax.set_ylabel("Temperature (°F)")
    ax2.set_ylabel("Heat Wave Parish-Days", color=C["hw"])
    ax.set_xlabel("Date")
    ax.set_title("Monthly Temperature and Heat Wave Trends (2015\u20132025)",
                 fontweight="bold", fontsize=TITLE_SIZE)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=FONT_SIZE - 1)

    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig13_temperature_trends.png"))
    plt.close(fig)
    print("    -> fig13_temperature_trends.png")


# ============================================================================
# FIGURE 14: UNIFIED COMPARISON (All-Cause vs Heat-Related)
# ============================================================================
def fig14_unified_comparison():
    """
    Visual centerpiece: forest plot showing all-cause vs heat-related models
    side by side, plus concentration test bar chart.
    """
    print("  Creating Figure 14: Unified Comparison...")

    # Load heat-related model results
    hr_path = os.path.join(MODEL_DIR, "heat_related_model_results.csv")
    comp_path = os.path.join(MODEL_DIR, "unified_heat_comparison_table.csv")
    if not os.path.exists(hr_path) or not os.path.exists(comp_path):
        print("    SKIP: heat-related model results not found. Run 06_statistical_models.py first.")
        return

    hr_results = pd.read_csv(hr_path)
    comp_table = pd.read_csv(comp_path)

    # Load main model results for the forest plot
    m3 = load_model(os.path.join(MODEL_DIR, "exploratory_hw_model_coefficients.csv"))
    with open(os.path.join(MODEL_DIR, "robustness_results.json")) as f:
        robust = json.load(f)

    # Get HR1 results
    hr1 = hr_results[hr_results["model"] == "HR1_HeatWave_HeatRelated"].iloc[0]
    hr2 = hr_results[hr_results["model"] == "HR2_TMAX_HeatRelated"].iloc[0]

    # Get Model 3 HW flag
    m3_hw = get_var(m3, "heat_wave_flag")
    m4a = robust.get("model_4a_parish_fe", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        'Heat Wave Association with Severe Workplace Injuries in Louisiana (2015\u20132025)\n'
        'All-Cause vs. Heat-Related Outcome',
        fontsize=13, fontweight='bold', y=1.01
    )

    # --- Left panel: Forest plot ---
    ax = axes[0]

    models = []
    if m3_hw is not None:
        models.append(('Model 3\nAll-cause (HW flag)',
                        m3_hw["IRR"], m3_hw["IRR_CI_Lower"], m3_hw["IRR_CI_Upper"],
                        '#4C72B0', 'all'))
    if m4a:
        models.append(('Model 4a\nAll-cause (+Parish FE)',
                        m4a["hw_irr"], m4a["hw_ci_lower"], m4a["hw_ci_upper"],
                        '#4C72B0', 'all'))
    models.append(('Model HR1\nHeat-related (HW flag)',
                    hr1["IRR"], hr1["CI_Lower"], hr1["CI_Upper"],
                    '#C44E52', 'hr'))
    models.append(('Model HR2\nHeat-related (TMAX)',
                    hr2["IRR"], hr2["CI_Lower"], hr2["CI_Upper"],
                    '#C44E52', 'hr'))

    y_positions = list(range(len(models)))[::-1]

    for y, (label, irr, lo, hi, color, _) in zip(y_positions, models):
        ax.plot([lo, hi], [y, y], '-', color=color, linewidth=2, alpha=0.7)
        ax.plot(irr, y, 'o', color=color, markersize=8, zorder=5)
        # Annotate IRR
        text_x = max(hi + 0.05, irr + 0.15)
        ax.text(text_x, y, f'IRR={irr:.3f}', va='center', fontsize=9, color=color)

    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m[0] for m in models], fontsize=9)
    ax.set_xlabel('Incidence Rate Ratio (IRR)', fontsize=10)
    ax.set_title('Forest Plot: All-Cause vs. Heat-Related', fontsize=10, fontweight='bold')

    # Set reasonable x limits based on data
    all_his = [m[3] for m in models]
    ax.set_xlim(0.4, max(all_his) + 1.0)

    all_patch = mpatches.Patch(color='#4C72B0', label='All-cause outcome (n=1,848)')
    hr_patch = mpatches.Patch(color='#C44E52', label='Heat-related outcome (n=81)')
    ax.legend(handles=[all_patch, hr_patch], loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # --- Right panel: Concentration test bar chart ---
    ax2 = axes[1]

    # Get actual values from comparison table
    conc_row = comp_table[comp_table["analysis"].str.contains("Concentration")].iloc[0]
    fisher_p = conc_row["P_value"]

    # Compute concentration values from master data
    master = pd.read_csv(os.path.join(MODEL_DIR, "master_dataset.csv"),
                         usecols=["tmax_f", "heat_wave_flag", "heat_related"])
    mc = master.dropna(subset=["tmax_f"])
    total_pd = len(mc)
    hw_pd = int(mc["heat_wave_flag"].sum())
    total_hr = int(mc["heat_related"].sum())
    hr_hw = int(mc.loc[mc["heat_wave_flag"] == 1, "heat_related"].sum())

    hw_share_pct = hw_pd / total_pd * 100
    hr_hw_share_pct = hr_hw / total_hr * 100 if total_hr > 0 else 0
    overrep = hr_hw_share_pct / hw_share_pct if hw_share_pct > 0 else 0

    categories = ['Share of\nParish-Days', 'Share of\nHeat-Related\nIncidents']
    hw_values = [hw_share_pct, hr_hw_share_pct]
    nonhw_values = [100 - hw_share_pct, 100 - hr_hw_share_pct]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width/2, hw_values, width, label='Heat Wave Days',
                    color='#C44E52', alpha=0.85, edgecolor='white')
    bars2 = ax2.bar(x + width/2, nonhw_values, width, label='Non-Heat Wave Days',
                    color='#4C72B0', alpha=0.85, edgecolor='white')

    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color='#C44E52')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom',
                 fontsize=10, color='#4C72B0')

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel('Percentage (%)', fontsize=10)
    ax2.set_title('Concentration Test\n(Model-Free Evidence)',
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', alpha=0.3)

    p_label = f"p < 0.0001" if fisher_p < 0.0001 else f"p = {fisher_p:.4f}"
    ax2.annotate(f'{overrep:.1f}\u00d7 overrepresentation\n(Fisher {p_label})',
                 xy=(0 - width/2, hr_hw_share_pct), xytext=(0.3, 75),
                 fontsize=9, color='#C44E52', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5))

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig14_unified_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("    -> fig14_unified_comparison.png")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("08_create_figures.py — Generating all 14 figures")
    print("=" * 60)

    fig1_sample_flow()
    fig2_parish_map()
    fig3_hw_comparison()
    fig4_forest_plot()
    fig5_industry()
    fig6_temporal()
    fig7_yearly()
    fig8_hw_by_year()
    fig9_station_map()
    fig10_interpolation()
    fig11_parishes()
    fig12_industry_parish()
    fig13_temperature()
    fig14_unified_comparison()

    print("\n" + "=" * 60)
    print("All 14 figures saved to:", FIG_DIR)
    print("=" * 60)
