import numpy as np
import pandas as pd

PEAK_HOURS = set([18,19,20,21,22])
OFFPEAK_HOURS = set([0,1,2,3,4,5,6])

def building_recommendations(building_row: dict):
    recs = []

    # Pattern-based recs
    if building_row["peak_to_offpeak_ratio"] > 1.7 and building_row["shiftable_share"] > 0.18:
        recs.append("High peak usage: shift flexible loads (EV charging, dishwasher, laundry, water heating) to off-peak hours; use timers/automation.")

    if building_row["base_kw"] > 0.55 and building_row["standby_share"] > 0.25:
        recs.append("High base load: investigate always-on/standby devices (electronics, plug loads); use smart power strips and shutdown policies.")

    if building_row["hvac_share"] > 0.40:
        recs.append("HVAC dominates consumption: tune setpoints, clean filters, check refrigerant/maintenance, and improve envelope (sealing, shading).")

    if building_row["variability_cv"] > 0.9:
        recs.append("Highly variable demand: stagger high-power appliances to avoid coincident peaks; consider demand-limiting controls.")

    # Peak hour-specific
    if int(building_row["peak_hour_mode"]) in PEAK_HOURS:
        recs.append("Peak demand occurs in evening peak window: pre-cool/pre-heat earlier, reduce simultaneous cooking + laundry + EV charging.")

    if not recs:
        recs.append("Usage looks balanced: focus on incremental gains—LED upgrades, HVAC scheduling, and checking standby loads.")

    return recs

def appliance_diagnostics(meter_df: pd.DataFrame, appliances_df: pd.DataFrame, building_id: str):
    """
    meter_df columns: timestamp, building_id, apartment_id, appliance, power_kw
    appliances_df columns: appliance, standby_w, typical_cycle_kwh, startup_overhead_kwh, shiftable
    """
    m = meter_df[meter_df["building_id"] == building_id].copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"])
    m["kwh"] = m["power_kw"] * 0.25
    m["date"] = m["timestamp"].dt.date
    m["hour"] = m["timestamp"].dt.hour

    meta = appliances_df.set_index("appliance")

    recs = []
    # Evaluate each appliance
    for app, g in m.groupby("appliance"):
        if app not in meta.index:
            continue

        standby_kw = float(meta.loc[app, "standby_w"]) / 1000.0
        startup_overhead_kwh = float(meta.loc[app, "startup_overhead_kwh"])
        shiftable = bool(meta.loc[app, "shiftable"])

        # Idle detection: intervals at ~standby (between 50% and 200% standby)
        if standby_kw > 0:
            idle = g[(g["power_kw"] >= 0.5*standby_kw) & (g["power_kw"] <= 2.0*standby_kw)]
            idle_hours_per_day = (idle.groupby("date").size().mean() or 0) * 0.25
        else:
            idle_hours_per_day = 0.0

        total_kwh = g["kwh"].sum()

        # Cycle detection heuristic: count "active" bursts above threshold
        thr = max(0.15, standby_kw + 0.2)
        g2 = g.sort_values("timestamp")
        active = (g2["power_kw"] > thr).astype(int)
        # count rising edges as "cycles"
        cycles = int(((active.diff().fillna(0) == 1).sum()))
        days = max(1, g2["date"].nunique())
        cycles_per_day = cycles / days

        # Frequent short-cycle consolidation only if startup overhead is meaningful
        if startup_overhead_kwh >= 0.20 and cycles_per_day >= 3.5:
            recs.append(f"{app}: frequent cycles (~{cycles_per_day:.1f}/day) with startup overhead—try consolidating runs (e.g., fewer, fuller loads).")

        if idle_hours_per_day >= 10 and standby_kw >= 0.008:
            recs.append(f"{app}: high idle/standby (~{idle_hours_per_day:.1f} h/day). Use a smart plug or switch off at the wall when not needed.")

        # Peak shifting suggestion
        if shiftable:
            peak_kwh = g[g["hour"].isin(list(PEAK_HOURS))]["kwh"].sum()
            off_kwh = g[g["hour"].isin(list(OFFPEAK_HOURS))]["kwh"].sum()
            if peak_kwh > 1.3 * (off_kwh + 1e-9):
                recs.append(f"{app}: runs concentrate in peak hours—schedule to off-peak where possible.")

    # Deduplicate and cap
    recs = list(dict.fromkeys(recs))
    return recs[:12]

def generate_all_recommendations(building_features_row, meter_df, appliances_df):
    recs = []
    recs.extend(building_recommendations(building_features_row))
    recs.extend(appliance_diagnostics(meter_df, appliances_df, building_features_row["building_id"]))
    return list(dict.fromkeys(recs))
