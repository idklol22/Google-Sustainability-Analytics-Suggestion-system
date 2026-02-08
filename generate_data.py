# generate_data.py
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

APPLIANCES = [
    # appliance, category, standby_w, typical_cycle_kwh, startup_overhead_kwh, shiftable
    ("HVAC", "climate", 10, 2.5, 0.2, False),
    ("WaterHeater", "hot_water", 5, 2.0, 0.1, True),
    ("Fridge", "always_on", 30, 1.2, 0.0, False),
    ("Lighting", "lighting", 2, 0.4, 0.0, True),
    ("Washer", "laundry", 1, 0.8, 0.25, True),
    ("Dryer", "laundry", 1, 2.2, 0.30, True),
    ("Dishwasher", "kitchen", 1, 1.1, 0.25, True),
    ("Oven", "kitchen", 1, 1.8, 0.35, True),
    ("TV", "electronics", 8, 0.25, 0.0, True),
    ("Computers", "electronics", 5, 0.35, 0.0, True),
    ("EVCharger", "ev", 2, 7.0, 0.0, True),
]

BUILDING_TYPES = ["Residential", "Office", "MixedUse"]

# ---------- knobs (safe defaults) ----------
DEFAULT_DAYS = 30          # increase to 60 if you want
DEFAULT_FREQ = "15min"
DEFAULT_N_BUILDINGS = 25   # increase to 40 if you want
RES_APT_RANGE = (4, 18)    # was (6,40); increase if your machine can handle it
OFFICE_ZONE_RANGE = (1, 6)
# ------------------------------------------


def make_time_index(days=DEFAULT_DAYS, freq=DEFAULT_FREQ, start="2025-11-01"):
    # 15-min data: 96 samples/day
    per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    return pd.date_range(start=start, periods=per_day * days, freq=freq)


def daily_profile(hour, building_type):
    # Shape a base daily profile by type
    if building_type == "Residential":
        morning = np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
        evening = np.exp(-0.5 * ((hour - 19) / 3.0) ** 2)
        return 0.35 + 0.85 * (0.8 * morning + 1.2 * evening)

    if building_type == "Office":
        work = np.exp(-0.5 * ((hour - 13) / 3.0) ** 2)
        return 0.20 + 1.10 * work

    # MixedUse
    morning = np.exp(-0.5 * ((hour - 8) / 2.2) ** 2)
    midday = np.exp(-0.5 * ((hour - 13) / 3.2) ** 2)
    evening = np.exp(-0.5 * ((hour - 19) / 3.0) ** 2)
    return 0.30 + 0.95 * (0.6 * morning + 0.8 * midday + 1.0 * evening)


def seasonal_temp_factor(day_index, period_days=60):
    # mild drift across the window to induce HVAC variation
    return 0.9 + 0.2 * np.sin(2 * np.pi * day_index / float(period_days))


def make_buildings(n_buildings=DEFAULT_N_BUILDINGS):
    rows = []
    for b in range(n_buildings):
        btype = RNG.choice(BUILDING_TYPES, p=[0.55, 0.30, 0.15])

        if btype != "Office":
            apartments = int(RNG.integers(RES_APT_RANGE[0], RES_APT_RANGE[1]))
        else:
            apartments = int(RNG.integers(OFFICE_ZONE_RANGE[0], OFFICE_ZONE_RANGE[1]))

        # FIX: use np.clip for scalar draws
        floor_area = float(np.clip(RNG.normal(3500, 1400), 800, 10000))

        occupants = int(
            apartments * int(RNG.integers(2, 5)) if btype != "Office" else int(RNG.integers(30, 200))
        )

        eff_grade = RNG.choice(["A", "B", "C", "D"], p=[0.15, 0.35, 0.35, 0.15])
        hvac = RNG.choice(["Split", "Central", "VRF"], p=[0.45, 0.35, 0.20])

        has_solar = bool(RNG.random() < (0.25 if btype != "Office" else 0.35))
        solar_kw = float(np.clip(RNG.normal(30, 15), 0, 80)) if has_solar else 0.0

        has_battery = bool(RNG.random() < 0.18)
        battery_kwh = float(np.clip(RNG.normal(80, 40), 0, 200)) if has_battery else 0.0

        rows.append(
            {
                "building_id": f"B{b:03d}",
                "building_type": btype,
                "floor_area_m2": round(floor_area, 1),
                "efficiency_grade": eff_grade,
                "apartments": apartments,
                "occupants_total": occupants,
                "hvac_type": hvac,
                "has_solar": has_solar,
                "solar_kw": round(solar_kw, 2),
                "has_battery": has_battery,
                "battery_kwh": round(battery_kwh, 1),
            }
        )
    return pd.DataFrame(rows)


def appliances_df():
    return pd.DataFrame(
        APPLIANCES,
        columns=["appliance", "category", "standby_w", "typical_cycle_kwh", "startup_overhead_kwh", "shiftable"],
    )


def simulate(buildings: pd.DataFrame, idx: pd.DatetimeIndex):
    appl = appliances_df().set_index("appliance")
    readings = []

    steps_per_day = int(pd.Timedelta("1D") / (idx[1] - idx[0]))
    period_days = max(1, int(len(idx) / steps_per_day))

    for _, b in buildings.iterrows():
        b_id = b["building_id"]
        btype = b["building_type"]
        eff = b["efficiency_grade"]

        # Efficiency affects baseline and HVAC intensity
        eff_mult = {"A": 0.85, "B": 0.95, "C": 1.05, "D": 1.18}[eff]

        # Segment “behavior archetypes” to create clear clustering structure
        archetype = RNG.choice(
            ["Peaky", "FlatHighBase", "OfficeStrict", "NightOwl", "Efficient"],
            p=[0.25, 0.20, 0.20, 0.20, 0.15],
        )

        apartments = int(b["apartments"])
        if btype == "Office":
            apartments = max(1, apartments)  # treat as zones

        # Allocate per-apartment occupants
        occ = int(b["occupants_total"])
        occ_per_apt = np.maximum(1, RNG.poisson(lam=max(1, occ / max(1, apartments)), size=apartments))

        for a in range(apartments):
            apt_id = f"{b_id}-A{a:03d}"

            # FIX: use np.clip for scalar draws
            apt_scale = float(np.clip(RNG.normal(1.0, 0.12), 0.7, 1.4))

            for ts_i, ts in enumerate(idx):
                hour = ts.hour + ts.minute / 60.0
                dow = ts.dayofweek
                is_weekend = dow >= 5
                day_i = ts_i // steps_per_day

                base = daily_profile(hour, btype) * eff_mult * apt_scale
                temp = seasonal_temp_factor(day_i, period_days=period_days)

                # Archetype shaping
                if archetype == "Peaky":
                    base *= 1.0 + 0.35 * np.exp(-0.5 * ((hour - 19) / 1.6) ** 2)
                elif archetype == "FlatHighBase":
                    base = 0.75 * eff_mult * apt_scale + 0.25 * base
                elif archetype == "OfficeStrict":
                    if btype != "Office":
                        base *= 0.9
                    base *= 0.25 if (hour < 7 or hour > 20) else 1.2
                    if is_weekend:
                        base *= 0.35
                elif archetype == "NightOwl":
                    base *= 1.0 + 0.25 * np.exp(-0.5 * ((hour - 1) / 2.5) ** 2)
                elif archetype == "Efficient":
                    base *= 0.88

                noise = RNG.normal(0, 0.05)
                base = max(0.05, base + noise)

                # Appliance contributions (kW) at this interval
                p = {}

                # Always-on
                p["Fridge"] = (float(appl.loc["Fridge", "standby_w"]) / 1000.0) + 0.06 * base

                # Lighting: more at evening/night in residential
                if btype != "Office":
                    light_factor = 0.2 + 0.9 * np.exp(-0.5 * ((hour - 20) / 3.0) ** 2)
                else:
                    light_factor = 0.15 + 0.7 * np.exp(-0.5 * ((hour - 14) / 4.0) ** 2)
                p["Lighting"] = 0.02 + 0.20 * light_factor * base

                # HVAC: tied to temp factor and occupancy window
                if btype != "Office":
                    occ_window = 0.3 if (hour < 6 or hour > 23) else 1.0
                else:
                    occ_window = 0.25 if (hour < 7 or hour > 19) else 1.0
                hvac_intensity = (0.7 + 0.9 * (temp - 0.9)) * occ_window
                p["HVAC"] = 0.15 + 0.75 * hvac_intensity * base

                # Water heater: morning/evening spikes residential; flatter office
                wh = 0.05
                if btype != "Office":
                    wh += 0.55 * np.exp(-0.5 * ((hour - 7.5) / 1.5) ** 2) + 0.45 * np.exp(
                        -0.5 * ((hour - 20.5) / 1.8) ** 2
                    )
                else:
                    wh += 0.35 * np.exp(-0.5 * ((hour - 12.5) / 2.8) ** 2)
                p["WaterHeater"] = 0.05 + 0.35 * wh * eff_mult * apt_scale

                # Electronics
                if btype != "Office":
                    screen = 0.25 + 0.8 * np.exp(-0.5 * ((hour - 21) / 3.0) ** 2)
                else:
                    screen = 0.2 + 0.8 * np.exp(-0.5 * ((hour - 14) / 3.5) ** 2)

                p["TV"] = 0.01 + 0.18 * screen * apt_scale
                p["Computers"] = 0.02 + 0.22 * (0.5 * screen + (0.7 if btype == "Office" else 0.2)) * apt_scale

                # Shiftable cycle appliances (probabilistic “runs”)
                def cycle_power(appliance, run_prob, mean_kw):
                    if RNG.random() < run_prob:
                        return mean_kw
                    return float(appl.loc[appliance, "standby_w"]) / 1000.0

                peak_bias = 1.35 if archetype == "Peaky" else 1.0
                offpeak_bias = 1.35 if archetype == "OfficeStrict" else 1.0

                evening = np.exp(-0.5 * ((hour - 20) / 2.2) ** 2)
                weekend_boost = 1.25 if is_weekend else 1.0
                p["Washer"] = cycle_power("Washer", run_prob=0.02 * evening * weekend_boost * peak_bias, mean_kw=0.6)
                p["Dryer"] = cycle_power("Dryer", run_prob=0.015 * evening * weekend_boost * peak_bias, mean_kw=1.8)
                p["Dishwasher"] = cycle_power("Dishwasher", run_prob=0.018 * evening * weekend_boost * peak_bias, mean_kw=1.2)

                dinner = np.exp(-0.5 * ((hour - 19) / 1.4) ** 2)
                lunch = np.exp(-0.5 * ((hour - 13) / 1.6) ** 2)
                p["Oven"] = cycle_power("Oven", run_prob=0.02 * (0.7 * lunch + 1.1 * dinner) * peak_bias, mean_kw=2.2)

                night = np.exp(-0.5 * ((hour - 1) / 2.5) ** 2) + np.exp(-0.5 * ((hour - 23) / 2.5) ** 2)
                ev_prob = 0.008 * (night * offpeak_bias + 0.2 * (1.0 - night)) * (
                    1.2 if archetype in ["FlatHighBase", "NightOwl"] else 1.0
                )
                p["EVCharger"] = cycle_power("EVCharger", run_prob=ev_prob, mean_kw=4.8)

                # Convert to long rows
                for appliance, kw in p.items():
                    readings.append(
                        {
                            "timestamp": ts,
                            "building_id": b_id,
                            "apartment_id": apt_id,
                            "appliance": appliance,
                            "power_kw": float(max(0.0, kw)),
                        }
                    )

    return pd.DataFrame(readings)


def main():
    os.makedirs("data", exist_ok=True)
    idx = make_time_index(days=DEFAULT_DAYS, freq=DEFAULT_FREQ, start="2025-11-01")

    b = make_buildings(n_buildings=DEFAULT_N_BUILDINGS)
    a = appliances_df()
    m = simulate(b, idx)

    b.to_csv("data/buildings.csv", index=False)
    a.to_csv("data/appliances.csv", index=False)
    m.to_csv("data/meter_readings.csv", index=False)

    print("Wrote:")
    print("  data/buildings.csv")
    print("  data/appliances.csv")
    print("  data/meter_readings.csv")
    print(f"Rows in meter_readings: {len(m):,}")


if __name__ == "__main__":
    main()
