# make_user_files.py
import argparse
import os
import sys
import pandas as pd


REQUIRED_METER_COLS = ["timestamp", "building_id", "apartment_id", "appliance", "power_kw"]
REQUIRED_APPLIANCE_COLS = ["appliance", "category", "standby_w", "typical_cycle_kwh", "startup_overhead_kwh", "shiftable"]


def ensure_dir(outdir: str):
    os.makedirs(outdir, exist_ok=True)


def write_templates(outdir: str):
    ensure_dir(outdir)

    # Meter readings template (LONG format)
    meter_template = pd.DataFrame(
        [
            # timestamp, building_id, apartment_id, appliance, power_kw
            ["2026-01-01 00:00:00", "B000", "B000-A000", "Fridge", 0.08],
            ["2026-01-01 00:00:00", "B000", "B000-A000", "HVAC", 0.30],
            ["2026-01-01 00:15:00", "B000", "B000-A000", "Fridge", 0.08],
            ["2026-01-01 00:15:00", "B000", "B000-A000", "HVAC", 0.28],
        ],
        columns=REQUIRED_METER_COLS,
    )

    # Appliances template (used by recommendation rules)
    appliances = pd.DataFrame(
        [
            ["HVAC", "climate", 10, 2.5, 0.2, False],
            ["WaterHeater", "hot_water", 5, 2.0, 0.1, True],
            ["Fridge", "always_on", 30, 1.2, 0.0, False],
            ["Lighting", "lighting", 2, 0.4, 0.0, True],
            ["Washer", "laundry", 1, 0.8, 0.25, True],
            ["Dryer", "laundry", 1, 2.2, 0.30, True],
            ["Dishwasher", "kitchen", 1, 1.1, 0.25, True],
            ["Oven", "kitchen", 1, 1.8, 0.35, True],
            ["TV", "electronics", 8, 0.25, 0.0, True],
            ["Computers", "electronics", 5, 0.35, 0.0, True],
            ["EVCharger", "ev", 2, 7.0, 0.0, True],
        ],
        columns=REQUIRED_APPLIANCE_COLS,
    )

    # Optional buildings template (not required by Streamlit upload path, but handy if you extend UI)
    buildings_template = pd.DataFrame(
        [
            ["B000", "Residential", 2200, "B", 12, 36, "Split", True, 20, False, 0],
        ],
        columns=[
            "building_id", "building_type", "floor_area_m2", "efficiency_grade",
            "apartments", "occupants_total", "hvac_type",
            "has_solar", "solar_kw", "has_battery", "battery_kwh"
        ],
    )

    meter_template.to_csv(os.path.join(outdir, "meter_readings_template.csv"), index=False)
    appliances.to_csv(os.path.join(outdir, "appliances.csv"), index=False)
    buildings_template.to_csv(os.path.join(outdir, "buildings_template.csv"), index=False)

    print(f"Wrote templates to: {outdir}")
    print("  meter_readings_template.csv")
    print("  appliances.csv")
    print("  buildings_template.csv")


def validate_meter(meter_path: str):
    m = pd.read_csv(meter_path)
    missing = [c for c in REQUIRED_METER_COLS if c not in m.columns]
    if missing:
        raise ValueError(f"meter_readings missing columns: {missing}")

    # basic type checks
    m["timestamp"] = pd.to_datetime(m["timestamp"], errors="raise")
    if (m["power_kw"] < 0).any():
        raise ValueError("meter_readings contains negative power_kw values.")

    if m["appliance"].isna().any():
        raise ValueError("meter_readings contains null appliance values.")

    # sanity checks
    if m["building_id"].nunique() < 1:
        raise ValueError("No building_id found.")
    if m["timestamp"].nunique() < 10:
        print("Warning: very few timestamps; clustering may be unstable.")

    return m


def validate_appliances(appliances_path: str):
    a = pd.read_csv(appliances_path)
    missing = [c for c in REQUIRED_APPLIANCE_COLS if c not in a.columns]
    if missing:
        raise ValueError(f"appliances missing columns: {missing}")

    # sanity
    if (a["standby_w"] < 0).any():
        raise ValueError("appliances contains negative standby_w values.")
    if (a["startup_overhead_kwh"] < 0).any():
        raise ValueError("appliances contains negative startup_overhead_kwh values.")
    return a


def do_validate(meter_path: str, appliances_path: str | None):
    print(f"Validating meter: {meter_path}")
    m = validate_meter(meter_path)
    print(f"  OK. Rows={len(m):,}, buildings={m['building_id'].nunique()}, apartments={m['apartment_id'].nunique()}")

    if appliances_path:
        print(f"Validating appliances: {appliances_path}")
        a = validate_appliances(appliances_path)
        print(f"  OK. Appliances={a['appliance'].nunique()}")


def generate_synthetic(outdir: str, days: int, freq: str, buildings: int, start: str):
    """
    Generates:
      - buildings.csv
      - appliances.csv
      - meter_readings.csv
    using your existing generate_data.py in the same project folder.
    """
    ensure_dir(outdir)

    try:
        import generate_data  # expects your fixed generate_data.py in same folder
    except Exception as e:
        raise RuntimeError(
            "Could not import generate_data.py. Put make_user_files.py in the same folder as generate_data.py."
        ) from e

    idx = generate_data.make_time_index(days=days, freq=freq, start=start)
    b = generate_data.make_buildings(n_buildings=buildings)
    a = generate_data.appliances_df()
    m = generate_data.simulate(b, idx)

    b.to_csv(os.path.join(outdir, "buildings.csv"), index=False)
    a.to_csv(os.path.join(outdir, "appliances.csv"), index=False)
    m.to_csv(os.path.join(outdir, "meter_readings.csv"), index=False)

    print(f"Wrote synthetic dataset to: {outdir}")
    print("  buildings.csv")
    print("  appliances.csv")
    print("  meter_readings.csv")
    print(f"Rows in meter_readings: {len(m):,}")


def main():
    p = argparse.ArgumentParser(description="Generate/validate input CSVs for the Streamlit energy app.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("templates", help="Write templates you can fill with your own data.")
    t.add_argument("--outdir", default="user_data", help="Output folder")

    v = sub.add_parser("validate", help="Validate your CSVs (columns, types, basic sanity).")
    v.add_argument("--meter", required=True, help="Path to meter_readings CSV")
    v.add_argument("--appliances", default=None, help="Path to appliances CSV (optional)")

    s = sub.add_parser("synthetic", help="Generate a custom synthetic dataset into a folder.")
    s.add_argument("--outdir", default="user_data", help="Output folder")
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--freq", default="15min")
    s.add_argument("--buildings", type=int, default=25)
    s.add_argument("--start", default="2025-11-01")

    args = p.parse_args()

    if args.cmd == "templates":
        write_templates(args.outdir)
    elif args.cmd == "validate":
        do_validate(args.meter, args.appliances)
    elif args.cmd == "synthetic":
        generate_synthetic(args.outdir, args.days, args.freq, args.buildings, args.start)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
