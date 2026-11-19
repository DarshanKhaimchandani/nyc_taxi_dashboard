"""NYC Taxi Fleet Intelligence | dashboard based on corrected Jupyter outputs.

Run from Taxi_Fleet_Project folder: streamlit run dashboard.py
No fabricated data is used. Missing exports are clearly identified.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NYC Taxi | Fleet Intelligence", page_icon="🚕", layout="wide", initial_sidebar_state="expanded")
NAVY = "#101a30"
GOLD = "#efb94c"
CYAN = "#5ad5d1"
GRAY = "#9aa9be"
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#d9e5f4"), margin=dict(l=15,r=15,t=35,b=15), hovermode="x unified")

st.markdown("""<style>
:root{--taxi-gold:#efb94c;} .stApp{background:linear-gradient(145deg,#0b1426,#0c182b 65%,#0e1b2f);color:#eff5fd}
[data-testid="stSidebar"]{background:#101d33;border-right:1px solid #243249}
.block-container{max-width:1450px;padding-top:1.9rem;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.025em} h1{font-size:2.4rem!important} h2{font-size:1.4rem!important}
[data-testid="stMetric"]{background:#14233c;padding:1rem 1.1rem;border:1px solid #273854;border-radius:14px}
[data-testid="stMetricLabel"]{color:#afbfce!important}
[data-testid="stMetricValue"]{color:#f4f8ff!important}
.hero{background:linear-gradient(110deg,#1c2b42,#101e33);padding:1.45rem 1.6rem;border:1px solid #33435a;border-radius:18px;margin-bottom:1.35rem}
.hero small{color:#efb94c;letter-spacing:.15em;font-weight:700}.hero p{color:#b7c6d9;margin:.5rem 0 0}
.info{background:#14243b;border:1px solid #34475d;padding:1rem 1.2rem;border-radius:12px;color:#c9d8ea}
.note{color:#9eafc3;font-size:.88rem}
[data-testid="stDataFrame"]{border:1px solid #2d4059;border-radius:10px}
</style>""", unsafe_allow_html=True)

# Supported layouts: dashboard.py in Taxi_Fleet_Project root; or inside a folder next to outputs.
HERE = Path(__file__).resolve().parent
PROJECT = HERE if (HERE / "outputs").is_dir() else HERE.parent if (HERE.parent / "outputs").is_dir() else HERE
OUTPUT = PROJECT / "outputs"
FIGURES = OUTPUT / "figures"

@st.cache_data(show_spinner=False)
def csv_file(path: str, dates: bool=False):
    kwargs = {"parse_dates": ["hour"]} if dates else {}
    return pd.read_csv(path, **kwargs)

@st.cache_data(show_spinner=False)
def json_file(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_optional(filename: str, dates=False):
    p = OUTPUT / filename
    if not p.exists(): return None
    try: return csv_file(str(p), dates=dates)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None

def read_metrics(filename: str):
    df = read_optional(filename)
    if df is not None:
        col = df.columns[0]
        if col.startswith("Unnamed") or col.lower() in ("model", "index"):
            df = df.set_index(col)
    return df

def need_file(name: str):
    st.warning(f"Missing: outputs/{name}. Run the corresponding notebook cells and refresh this page.")

def theme(fig, title=None, height=395):
    fig.update_layout(**PLOT, height=height, title=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#263852", zeroline=False)
    return fig

def download(df, filename: str):
    st.download_button("Download data CSV", df.to_csv(index=False).encode("utf-8"), filename, "text/csv", use_container_width=False)

def safe_pct(a, b):
    return 100 * a / b if b > 0 else 100.0

def allocate_taxis(predictions, fleet_size):
    """Largest-remainder proportional fleet allocator, identical to notebook approach."""
    arr = np.maximum(np.nan_to_num(np.asarray(predictions, dtype=float), nan=0.0), 0)
    if len(arr) == 0: return np.array([], dtype=int)
    probs = arr / arr.sum() if arr.sum() > 0 else np.full(len(arr), 1.0 / len(arr))
    exact = probs * int(fleet_size)
    floor = np.floor(exact).astype(int)
    extra = int(fleet_size) - int(floor.sum())
    if extra: floor[np.argsort(-(exact - floor), kind="stable")[:extra]] += 1
    return floor

@st.cache_data(show_spinner=False)
def compute_sensitivity(_predictions, sizes):
    vals = []
    groups = [(g.pickup_count.to_numpy(), g.baseline_prediction.to_numpy(), g.predicted_pickups.to_numpy())
              for _, g in _predictions.groupby("hour", sort=True)]
    for fleet in sizes:
        total = 0; baseline = 0; ai = 0
        for actual, prev, pred in groups:
            a1 = allocate_taxis(prev, fleet); a2 = allocate_taxis(pred, fleet)
            total += actual.sum(); baseline += np.minimum(actual,a1).sum(); ai += np.minimum(actual,a2).sum()
        vals.append({"Fleet Size":fleet, "Baseline Coverage (%)":safe_pct(baseline,total),
                     "AI Coverage (%)":safe_pct(ai,total),"Gain (pp)":safe_pct(ai,total)-safe_pct(baseline,total)})
    return pd.DataFrame(vals)

predictions = read_optional("test_predictions_corrected.csv", dates=True)
metrics = read_metrics("test_metrics_corrected.csv")
validation = read_metrics("validation_metrics_corrected.csv")
three_models = read_metrics("three_model_validation_comparison.csv")
simulation = read_optional("fleet_simulation_corrected.csv", dates=True)
importance = read_optional("feature_importance_corrected.csv")
sensitivity = read_optional("fleet_sensitivity_corrected.csv")
quality = read_optional("data_quality_summary.csv")
raw_business_path = OUTPUT / "business_results_corrected.json"
business = json_file(str(raw_business_path)) if raw_business_path.exists() else None

st.sidebar.markdown("## 🚕 TAXI INTELLIGENCE")
st.sidebar.caption("NYC Yellow Taxi  ·  January 2024")
page = st.sidebar.radio("Explore analysis", ["Overview", "Demand explorer", "Model performance", "Fleet decision lab", "Data & methodology"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**Dataset**")
st.sidebar.caption("NYC TLC official real-world trip records. January 2024; 25 selected pickup zones.")
st.sidebar.link_button("Official NYC TLC source", "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page", use_container_width=True)
st.sidebar.caption("Academic prototype • Not a production dispatch system")

st.markdown('<div class="hero"><small>DOMAIN APPLICATIONS  /  MACHINE LEARNING</small><h1>NYC Taxi Fleet Intelligence</h1><p>Rolling one-hour-ahead pickup forecasts & cost-aware fleet-planning exploration using real taxi trip records.</p></div>', unsafe_allow_html=True)

if predictions is not None:
    required_cols = {"hour","PULocationID","pickup_count","predicted_pickups","baseline_prediction"}
    absent = sorted(required_cols.difference(predictions.columns))
    if absent:
        st.error("Predictions file lacks columns: " + ", ".join(absent))
        st.stop()
    predictions = predictions.sort_values(["hour","PULocationID"])

if page == "Overview":
    st.subheader("Decision intelligence at a glance")
    if business:
        b = business.get("baseline_coverage_pct")
        a = business.get("ai_coverage_pct")
    else:
        b = a = None
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Analysed trips", "2.96M", help="January 2024 raw records from the user's notebook; 2,964,624 before cleaning")
    c2.metric("Selected zones", f"{predictions.PULocationID.nunique():,}" if predictions is not None else "25")
    c3.metric("Baseline coverage*", f"{b:.2f}%" if b is not None else "Not exported")
    c4.metric("AI coverage*", f"{a:.2f}%" if a is not None else "Not exported", delta=f"{a-b:+.2f} pp" if a is not None and b is not None else None)
    st.caption("*Average of hourly simulated coverage percentages with the notebook's assumed 1,000-vehicle fleet. Not real-world service coverage or verified idle-time savings.")
    left,right = st.columns([1.65,1], gap="large")
    with left:
        st.subheader("Observed demand vs forecast")
        if predictions is None: need_file("test_predictions_corrected.csv")
        else:
            hourly_plot = predictions.groupby("hour",as_index=False)[["pickup_count","predicted_pickups"]].sum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hourly_plot.hour,y=hourly_plot.pickup_count,name="Actual demand",line=dict(color=GOLD,width=2)))
            fig.add_trace(go.Scatter(x=hourly_plot.hour,y=hourly_plot.predicted_pickups,name="AI forecast",line=dict(color=CYAN,width=2,dash="dash")))
            st.plotly_chart(theme(fig,height=385), use_container_width=True)
    with right:
        st.subheader("What the solution does")
        st.markdown("**01 — Forecast:** Predict taxi pickups for each zone and hour using lagged demand and time features.")
        st.markdown("**02 — Allocate:** Divide an assumed fixed fleet between zones proportional to predicted demand.")
        st.markdown("**03 — Evaluate:** Compare historical and AI strategies on the held-out test period.")
        st.info("The reported improvement is a simplified **simulation**, not a measured reduction in idle time, fuel use or cost.")
    if metrics is not None:
        st.subheader("Final held-out test performance")
        st.dataframe(metrics.round(3), use_container_width=True)
    else: need_file("test_metrics_corrected.csv")

elif page == "Demand explorer":
    st.subheader("Zone-level pickup forecasting")
    if predictions is None: need_file("test_predictions_corrected.csv")
    else:
        zones = sorted(predictions.PULocationID.unique().tolist())
        selected = st.selectbox("Taxi pickup zone ID", zones, index=0)
        zone_df = predictions.loc[predictions.PULocationID == selected].sort_values("hour")
        col1,col2,col3 = st.columns(3)
        col1.metric("Actual pickups", f"{zone_df.pickup_count.sum():,.0f}")
        col2.metric("AI estimated pickups", f"{zone_df.predicted_pickups.sum():,.0f}")
        col3.metric("Mean absolute error", f"{(zone_df.pickup_count-zone_df.predicted_pickups).abs().mean():.2f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=zone_df.hour,y=zone_df.pickup_count,name="Actual",line=dict(color=GOLD,width=2)))
        fig.add_trace(go.Scatter(x=zone_df.hour,y=zone_df.predicted_pickups,name="Gradient Boosting",line=dict(color=CYAN,width=2,dash="dash")))
        fig.add_trace(go.Scatter(x=zone_df.hour,y=zone_df.baseline_prediction,name="Yesterday baseline",line=dict(color="#8494b8",width=1,dash="dot")))
        st.plotly_chart(theme(fig, "Hourly demand — selected zone", 450),use_container_width=True)
        cols = st.columns(2)
        with cols[0]:
            dd = predictions.copy(); dd["Hour of day"] = dd.hour.dt.hour
            d = dd.groupby("Hour of day",as_index=False).pickup_count.mean()
            fig2 = px.bar(d,x="Hour of day",y="pickup_count",color_discrete_sequence=[GOLD])
            st.plotly_chart(theme(fig2,"Average zone-hour demand by clock hour",320),use_container_width=True)
        with cols[1]:
            z = predictions.groupby("PULocationID",as_index=False).pickup_count.sum().nlargest(10,"pickup_count")
            z["Zone"] = z.PULocationID.astype(str)
            fig3 = px.bar(z,x="Zone",y="pickup_count",color_discrete_sequence=[CYAN])
            st.plotly_chart(theme(fig3,"Top 10 test-period zones",320),use_container_width=True)
        download(zone_df,"selected_zone_predictions.csv")
        st.caption("Prediction uses observed previous hourly pickup counts. This is a rolling one-hour-ahead study, not an advance monthly forecast.")

elif page == "Model performance":
    st.subheader("Model accuracy & explainability")
    if metrics is not None:
        st.markdown("**Final chronological hold-out test**")
        display_cols = [col for col in ["MAE","RMSE","WAPE (%)"] if col in metrics.columns]
        if display_cols:
            shown = metrics[display_cols]
            st.dataframe(shown.round(3),use_container_width=True)
            melted = shown.reset_index().melt(id_vars=shown.index.name or "index",value_vars=display_cols, var_name="Metric",value_name="Error")
            namecol = shown.index.name or "index"
            fig = px.bar(melted,x="Metric",y="Error",color=namecol,barmode="group",color_discrete_sequence=["#9aa9be",GOLD,CYAN])
            st.plotly_chart(theme(fig,"Lower is better",365),use_container_width=True)
    else: need_file("test_metrics_corrected.csv")
    if three_models is not None:
        st.markdown("**Validation-only comparison — three approaches**")
        st.dataframe(three_models.round(3),use_container_width=True)
        st.caption("Ridge Regression is validated, not selected using the final test set.")
    elif validation is not None:
        st.markdown("**Chronological validation**")
        st.dataframe(validation.round(3),use_container_width=True)
    if importance is not None and {"Feature","Importance"}.issubset(importance.columns):
        chart = importance.sort_values("Importance")
        fig = px.bar(chart,x="Importance",y="Feature",orientation="h",color_discrete_sequence=[CYAN],error_x="Std" if "Std" in chart else None)
        st.plotly_chart(theme(fig,"Permutation feature importance • validation",385),use_container_width=True)
        st.caption("Values show the validation MAE increase when each feature is shuffled. Correlated inputs can share importance; this is not a causal explanation.")
    else: need_file("feature_importance_corrected.csv")
    if predictions is not None:
        p = predictions.copy()
        p["Residual"] = p.pickup_count - p.predicted_pickups
        p["Clock hour"] = p.hour.dt.hour
        left,right = st.columns(2)
        with left:
            hist = px.histogram(p,x="Residual",nbins=65,color_discrete_sequence=[GOLD]); st.plotly_chart(theme(hist,"Error distribution",325), use_container_width=True)
        with right:
            grouped = p.groupby("Clock hour").Residual.apply(lambda s:s.abs().mean()).reset_index(name="MAE")
            fig2 = px.bar(grouped,x="Clock hour",y="MAE",color_discrete_sequence=[CYAN]); st.plotly_chart(theme(fig2,"Mean absolute error by hour",325), use_container_width=True)

elif page == "Fleet decision lab":
    st.subheader("Interactive fleet positioning")
    st.markdown("Compare two proportional allocation strategies for a **hypothetical fixed fleet**. Adjust fleet size and choose a test hour to inspect allocations.")
    if predictions is None: need_file("test_predictions_corrected.csv")
    else:
        size = st.slider("Assumed available taxis", min_value=100,max_value=3000,value=1000,step=50)
        hours = sorted(predictions.hour.unique().tolist())
        selected_hour = st.select_slider("Test-hour selection",options=hours,value=hours[len(hours)//2],format_func=lambda x:pd.Timestamp(x).strftime("%d %b %Y, %H:%M"))
        subset = predictions.loc[predictions.hour == selected_hour].copy().sort_values("PULocationID")
        subset["Baseline taxis"] = allocate_taxis(subset.baseline_prediction,size)
        subset["AI taxis"] = allocate_taxis(subset.predicted_pickups,size)
        subtotal = subset.pickup_count.sum()
        base_served = np.minimum(subset.pickup_count,subset["Baseline taxis"]).sum()
        ai_served = np.minimum(subset.pickup_count,subset["AI taxis"]).sum()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Observed pickups", f"{subtotal:,.0f}")
        c2.metric("Baseline covered", f"{safe_pct(base_served,subtotal):.2f}%")
        c3.metric("AI covered", f"{safe_pct(ai_served,subtotal):.2f}%")
        c4.metric("Coverage difference", f"{safe_pct(ai_served,subtotal)-safe_pct(base_served,subtotal):+.2f} pp")
        subset["Zone"] = subset.PULocationID.astype(str)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=subset.Zone,y=subset["Baseline taxis"],name="Historical allocation",marker_color="#8595b0"))
        fig.add_trace(go.Bar(x=subset.Zone,y=subset["AI taxis"],name="AI allocation",marker_color=GOLD))
        fig.update_layout(barmode="group")
        st.plotly_chart(theme(fig,"Recommended taxis by pickup zone",405),use_container_width=True)
        with st.expander("View allocation table and download"):
            st.dataframe(subset[["PULocationID","pickup_count","baseline_prediction","predicted_pickups","Baseline taxis","AI taxis"]].round(2),use_container_width=True)
            download(subset,"fleet_allocation_selected_hour.csv")
        st.divider()
        st.subheader("What if the fleet size changes?")
        if sensitivity is not None:
            sens = sensitivity.copy()
        else:
            sens = compute_sensitivity(predictions,tuple([250,500,750,1000,1250,1500,2000,2500,3000]))
        expected = ["Fleet Size","Baseline Coverage (%)","AI Coverage (%)"]
        if set(expected).issubset(sens.columns):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=sens["Fleet Size"],y=sens["Baseline Coverage (%)"],mode="lines+markers",name="Historical allocation",line=dict(color="#95a4b9")))
            fig2.add_trace(go.Scatter(x=sens["Fleet Size"],y=sens["AI Coverage (%)"],mode="lines+markers",name="AI allocation",line=dict(color=GOLD)))
            st.plotly_chart(theme(fig2,"Pickup-weighted simulated coverage across the full test period",380),use_container_width=True)
            st.dataframe(sens.round(2),use_container_width=True)
        st.warning("**Simulation assumptions:** every taxi is instantly located in its allocated zone, serves at most one pickup per hour, and incurs no travel/repositioning cost. Coverage is a proxy, not actual reductions in waiting time, operating cost or driver idle time.")

else:
    st.subheader("Data provenance & analytical design")
    st.markdown("**Official dataset:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) · January 2024 Yellow Taxi trips")
    a,b,c = st.columns(3)
    a.metric("Raw records", "2,964,624")
    b.metric("Raw columns", "19")
    c.metric("Scope", "January 2024")
    st.markdown("""**Pipeline**: validate pickup timestamps and zones → aggregate pickups into hour × zone counts → fill unobserved zero-pickup hours → engineer calendar and lag features → chronological train/validation/test split → fit and evaluate models → simulate fleet allocations.

**Forecast inputs**: pickup zone ID, hour of day, day of week, weekend flag, observed pickup counts at 1/24/168 hours prior and trailing 24-hour average. No demand from the predicted hour is used as input.

**Look-ahead correction**: Top-25 zones were reselected from training-period data only after discovering one mismatched zone in the original selection. Earlier notebook results should not be used for final reporting.

**Split**: 70% training, 15% validation and 15% testing, ordered by time rather than randomly shuffled. Predictions are evaluated as rolling one-hour-ahead forecasts, using historical observations available by each prediction time.

**Missing values and outliers**: Date/zone validity is critical for pickup-demand estimation. Missing passenger counts and negative fares are not automatically reasons to exclude pickup events. Genuine peak demand should be preserved.

**Ethics and limitations**: Trip records may reveal aggregate mobility patterns. Avoid identifying riders or claiming individual-level behaviour. Only one month is studied; there is no evidence of full-year seasonality, no measured driver supply, no travel-time constraints, and no real-world cost savings established.
""")
    if quality is not None:
        st.subheader("Notebook data-quality audit")
        st.dataframe(quality,use_container_width=True,hide_index=True)
    else:
        st.caption("Optional: export outputs/data_quality_summary.csv from your notebook to show your recorded quality checks here.")
    st.divider()
    st.subheader("Available output files")
    files = ["test_predictions_corrected.csv","test_metrics_corrected.csv","validation_metrics_corrected.csv","three_model_validation_comparison.csv","fleet_simulation_corrected.csv","business_results_corrected.json","feature_importance_corrected.csv","fleet_sensitivity_corrected.csv","data_quality_summary.csv"]
    st.dataframe(pd.DataFrame([{"File":f,"Status":"Ready" if (OUTPUT / f).is_file() else "Missing"} for f in files]),hide_index=True,use_container_width=True)

st.divider()
st.caption("NC Ireland MSc Domain Applications · reproducible dashboard from your corrected January notebook exports · no made-up model results")
