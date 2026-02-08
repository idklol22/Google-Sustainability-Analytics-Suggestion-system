# app.py
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from recommend import generate_all_recommendations

FEATURES = [
    "avg_kw", "peak_kw", "base_kw", "load_factor",
    "variability_cv", "peak_hour_mode",
    "peak_to_offpeak_ratio", "weekend_weekday_ratio",
    "hvac_share", "standby_share", "shiftable_share"
]

@st.cache_data
def load_artifacts():
    scaler = joblib.load("artifacts/scaler.joblib")
    pca = joblib.load("artifacts/pca.joblib")
    km = joblib.load("artifacts/kmeans.joblib")

    pca_ev = pd.read_csv("artifacts/pca_explained_variance.csv")
    k_sweep = pd.read_csv("artifacts/k_sweep.csv")
    scores2d = pd.read_csv("artifacts/pca_scores_2d.csv")
    sil = pd.read_csv("artifacts/silhouette_samples.csv")
    profiles = pd.read_csv("artifacts/cluster_profiles.csv")

    return scaler, pca, km, pca_ev, k_sweep, scores2d, sil, profiles

def build_building_features_from_meter_df(m):
    m = m.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"])
    m["kwh"] = m["power_kw"] * 0.25

    bt = m.groupby(["building_id", "timestamp"], as_index=False)["power_kw"].sum()
    bt["hour"] = bt["timestamp"].dt.hour
    bt["dow"] = bt["timestamp"].dt.dayofweek
    bt["is_weekend"] = (bt["dow"] >= 5).astype(int)

    agg = bt.groupby("building_id")["power_kw"].agg(["mean", "max", "std"]).rename(
        columns={"mean": "avg_kw", "max": "peak_kw", "std": "std_kw"}
    )
    base = bt.groupby("building_id")["power_kw"].quantile(0.05).rename("base_kw")
    feat = agg.join(base)

    feat["load_factor"] = feat["avg_kw"] / feat["peak_kw"]
    feat["variability_cv"] = feat["std_kw"] / (feat["avg_kw"] + 1e-9)

    bt["is_peak_interval"] = bt.groupby("building_id")["power_kw"].transform(lambda s: s == s.max())
    peak_hours = bt[bt["is_peak_interval"]].groupby("building_id")["hour"].agg(
        lambda s: int(s.mode().iloc[0]) if len(s.mode()) else int(s.iloc[0])
    ).rename("peak_hour_mode")
    feat = feat.join(peak_hours)

    peak_mask = bt["hour"].between(18, 22)
    off_mask = bt["hour"].between(0, 6)
    peak_avg = bt[peak_mask].groupby("building_id")["power_kw"].mean().rename("peak_avg_kw")
    off_avg = bt[off_mask].groupby("building_id")["power_kw"].mean().rename("off_avg_kw")
    feat = feat.join(peak_avg).join(off_avg)
    feat["peak_to_offpeak_ratio"] = feat["peak_avg_kw"] / (feat["off_avg_kw"] + 1e-9)
    feat.drop(columns=["peak_avg_kw", "off_avg_kw"], inplace=True)

    wknd = bt[bt["is_weekend"] == 1].groupby("building_id")["power_kw"].mean().rename("wknd_kw")
    wkdy = bt[bt["is_weekend"] == 0].groupby("building_id")["power_kw"].mean().rename("wkdy_kw")
    feat = feat.join(wknd).join(wkdy)
    feat["weekend_weekday_ratio"] = feat["wknd_kw"] / (feat["wkdy_kw"] + 1e-9)
    feat.drop(columns=["wknd_kw", "wkdy_kw"], inplace=True)

    # shares
    app = m.groupby(["building_id", "appliance"], as_index=False)["kwh"].sum()
    total = app.groupby("building_id")["kwh"].sum().rename("total_kwh")
    app = app.merge(total, on="building_id", how="left")
    app["share"] = app["kwh"] / (app["total_kwh"] + 1e-9)

    hvac = app[app["appliance"] == "HVAC"].set_index("building_id")["share"].rename("hvac_share")
    standby_like = app[app["appliance"].isin(["Fridge", "TV", "Computers"])].groupby("building_id")["share"].sum().rename("standby_share")
    shiftable = app[app["appliance"].isin(["Washer", "Dryer", "Dishwasher", "Oven", "EVCharger", "WaterHeater", "Lighting"])].groupby("building_id")["share"].sum().rename("shiftable_share")

    feat = feat.join(hvac).join(standby_like).join(shiftable)
    feat[["hvac_share", "standby_share", "shiftable_share"]] = feat[["hvac_share", "standby_share", "shiftable_share"]].fillna(0.0)

    return feat.reset_index()

def main():
    st.set_page_config(page_title="Energy PCA + KMeans Dashboard", layout="wide")
    st.title("Smart Building Segmentation (PCA + K-Means) + Recommendations")

    scaler, pca, km, pca_ev, k_sweep, scores2d_pre, sil_pre, profiles_pre = load_artifacts()

    st.sidebar.header("Data input")
    mode = st.sidebar.radio("Choose input", ["Use generated ./data", "Upload my CSV"], index=0)

    if mode == "Use generated ./data":
        meter = pd.read_csv("data/meter_readings.csv")
        appliances = pd.read_csv("data/appliances.csv")
        # Use precomputed 2D PCA and silhouette for nicer “model explanation”
        scores2d = scores2d_pre.copy()
        sil = sil_pre.copy()
        profiles = profiles_pre.copy()
        using_precomputed = True
    else:
        up = st.sidebar.file_uploader("Upload meter_readings CSV", type=["csv"])
        up2 = st.sidebar.file_uploader("Upload appliances CSV (optional)", type=["csv"])
        if up is None:
            st.info("Upload a meter_readings CSV with columns: timestamp, building_id, apartment_id, appliance, power_kw")
            return
        meter = pd.read_csv(up)
        appliances = pd.read_csv(up2) if up2 is not None else pd.read_csv("data/appliances.csv")

        # Compute clusters on-the-fly with the saved model
        feat = build_building_features_from_meter_df(meter)
        X = feat[FEATURES]
        Xs = scaler.transform(X)
        Xp = pca.transform(Xs)
        labels = km.predict(Xp)

        pc1 = Xp[:, 0]
        pc2 = Xp[:, 1] if Xp.shape[1] > 1 else 0 * pc1
        scores2d = pd.DataFrame({"building_id": feat["building_id"], "pc1": pc1, "pc2": pc2, "cluster": labels})

        # Silhouette samples require distances in the projected space
        from sklearn.metrics import silhouette_samples
        sil_vals = silhouette_samples(Xp, labels)
        sil = pd.DataFrame({"building_id": feat["building_id"], "cluster": labels, "silhouette": sil_vals})

        # Profiles on-the-fly
        feat["cluster"] = labels
        profiles = feat.groupby("cluster")[FEATURES].mean()
        profiles["count"] = feat.groupby("cluster")["building_id"].size()
        profiles = profiles.reset_index()

        using_precomputed = False

    tabs = st.tabs(["Summary", "PCA", "K-Means quality", "Building deep-dive"])

    # ---------- Summary ----------
    with tabs[0]:
        st.subheader("Cluster summary")
        counts = profiles[["cluster", "count"]].sort_values("cluster")
        st.plotly_chart(px.bar(counts, x="cluster", y="count", title="Buildings per cluster"), use_container_width=True)

        st.subheader("Cluster profiles (mean features)")
        st.dataframe(profiles.sort_values("cluster"), use_container_width=True)

        st.caption("Tip: clusters become explainable when you look at which features are high/low in each cluster.")

    # ---------- PCA ----------
    with tabs[1]:
        st.subheader("Explained variance (PCA)")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pca_ev["component"],
            y=pca_ev["explained_variance_ratio"],
            name="Explained variance ratio"
        ))
        fig.add_trace(go.Scatter(
            x=pca_ev["component"],
            y=pca_ev["cumulative_explained_variance"],
            mode="lines+markers",
            name="Cumulative"
        ))
        fig.update_layout(
            title="How many PCs are needed?",
            xaxis_title="Principal component",
            yaxis_title="Variance explained"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("2D PCA map (points = buildings)")
        st.plotly_chart(
            px.scatter(
                scores2d, x="pc1", y="pc2", color="cluster",
                hover_data=["building_id"],
                title="Buildings projected onto PC1 vs PC2 (colored by cluster)"
            ),
            use_container_width=True
        )

    # ---------- KMeans quality ----------
    with tabs[2]:
        st.subheader("Choosing k (only for the training dataset)")
        if using_precomputed:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=k_sweep["k"], y=k_sweep["silhouette"], mode="lines+markers", name="Silhouette"))
            fig.add_trace(go.Scatter(x=k_sweep["k"], y=k_sweep["inertia"], mode="lines+markers", name="Inertia", yaxis="y2"))
            fig.update_layout(
                title="K sweep: silhouette (higher is better) and inertia (lower is better)",
                xaxis_title="k",
                yaxis=dict(title="Silhouette"),
                yaxis2=dict(title="Inertia", overlaying="y", side="right"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("k-sweep is not shown for uploads because the app uses the pre-trained k from artifacts.")

        st.subheader("Silhouette distribution by cluster")
        st.plotly_chart(
            px.box(
                sil, x="cluster", y="silhouette",
                points="all",
                title="Silhouette samples (per building) grouped by cluster"
            ),
            use_container_width=True
        )

    # ---------- Building deep-dive ----------
    with tabs[3]:
        st.subheader("Select a building")
        b_id = st.selectbox("building_id", sorted(scores2d["building_id"].unique().tolist()))

        # For recommendations we need the engineered feature row
        feat_now = build_building_features_from_meter_df(meter)
        X = feat_now[FEATURES]
        Xs = scaler.transform(X)
        Xp = pca.transform(Xs)
        labels = km.predict(Xp)
        feat_now["cluster"] = labels

        row = feat_now[feat_now["building_id"] == b_id].iloc[0].to_dict()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Recommendations")
            recs = generate_all_recommendations(row, meter, appliances)
            for r in recs:
                st.write("- " + r)

            st.subheader("Feature snapshot")
            st.write({k: row[k] for k in ["cluster"] + FEATURES})

        with col2:
            st.subheader("Load shape + appliance mix")
            m = meter[meter["building_id"] == b_id].copy()
            m["timestamp"] = pd.to_datetime(m["timestamp"])
            m["kwh"] = m["power_kw"] * 0.25

            ts = m.groupby(["timestamp"], as_index=False)["power_kw"].sum()
            st.plotly_chart(px.line(ts, x="timestamp", y="power_kw", title=f"{b_id} total load (kW)"), use_container_width=True)

            app = m.groupby("appliance", as_index=False)["kwh"].sum().sort_values("kwh", ascending=False)
            st.plotly_chart(px.bar(app, x="appliance", y="kwh", title=f"{b_id} energy by appliance (kWh)"), use_container_width=True)

if __name__ == "__main__":
    main()
