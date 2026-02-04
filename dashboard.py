# dashboard_test6.py
# ----------------------------------------------------------
# Real-time Dashboard (Home, Feeder A, Feeder B, Feeder C)
# Smooth Live Updates (no blinking/full page rerun)
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import time
import datetime
import plotly.graph_objs as go

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="Power Distribution DT - Home", layout="wide")
CSV_PATH = "Total_P&Q.csv"
FEEDER_A_P = "FeederA_P.csv"
FEEDER_A_Q = "FeederA_Q.csv"
FEEDER_A_V = "FeederA_Bus_pu_voltages.csv"
FEEDER_B_P = "FeederB_P.csv"
FEEDER_B_Q = "FeederB_Q.csv"
FEEDER_B_V = "FeederB_Bus_pu_voltages.csv"
FEEDER_C_P = "FeederC_P.csv"
FEEDER_C_Q = "FeederC_Q.csv"
FEEDER_C_V = "FeederC_Bus_pu_voltages.csv"

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def load_feeder_a_p():
    return pd.read_csv(FEEDER_A_P)

@st.cache_data
def load_feeder_a_q():
    return pd.read_csv(FEEDER_A_Q)

@st.cache_data
def load_feeder_a_v():
    return pd.read_csv(FEEDER_A_V)

@st.cache_data
def load_feeder_b_p():
    return pd.read_csv(FEEDER_B_P)

@st.cache_data
def load_feeder_b_q():
    return pd.read_csv(FEEDER_B_Q)

@st.cache_data
def load_feeder_b_v():
    return pd.read_csv(FEEDER_B_V)

@st.cache_data
def load_feeder_c_p():
    return pd.read_csv(FEEDER_C_P)

@st.cache_data
def load_feeder_c_q():
    return pd.read_csv(FEEDER_C_Q)

@st.cache_data
def load_feeder_c_v():
    return pd.read_csv(FEEDER_C_V)


def make_line_fig(x, y, title, yaxis_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=title,
                             hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"))
    fig.update_layout(
        margin=dict(l=20, r=20, t=32, b=20),
        title=dict(text=title, x=0.01, xanchor='left'),
        yaxis=dict(title=yaxis_title),
        xaxis=dict(title="Time")
    )
    return fig


def make_gauge(value, title, unit, color):
    # handle zero or very small values for axis
    axis_max = max(1.0, abs(value) * 1.5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': f"<b>{title}</b>", 'font': {'size': 18}},
        number={'suffix': f" {unit}", 'font': {'size': 26}},
        delta={'reference': 0, 'relative': False},
        gauge={
            'axis': {'range': [0, axis_max], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
        }
    ))
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def make_dual_line_fig(x, y1, y2, title, y1_title, y2_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y1, mode="lines", name=y1_title,
        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        line=dict(color="#4CAF50")
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y2, mode="lines", name=y2_title,
        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        line=dict(color="#2196F3"),
        yaxis="y2"
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=32, b=20),
        title=dict(text=title, x=0.01, xanchor='left'),
        yaxis=dict(title=y1_title),
        yaxis2=dict(title=y2_title, overlaying="y", side="right"),
        xaxis=dict(title="Time")
    )
    return fig


def make_voltage_fig(x, voltages_df, v_cols, tap_cols, bus_name):
    fig = go.Figure()
    phase_colors = ["red", "green", "blue"]
    for i, col in enumerate(v_cols):
        if col in voltages_df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=voltages_df[col], mode="lines", name=f"Voltage Phase {i+1}",
                hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
                line=dict(color=phase_colors[i])
            ))
    tap_colors = ["orange", "purple", "brown"]
    for i, col in enumerate(tap_cols):
        if col in voltages_df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=voltages_df[col], mode="lines", name=f"Tap Changer {i+1}",
                hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
                line=dict(color=tap_colors[i], dash="dash")
            ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=32, b=20),
        title=dict(text=f"{bus_name} - Bus Voltages (PU) & Tap Changers", x=0.01, xanchor='left'),
        yaxis=dict(title="PU"),
        xaxis=dict(title="Time")
    )
    return fig


# ----------------------------------------------------------
# LOAD PRIMARY DATA
# ----------------------------------------------------------
try:
    df_raw = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Could not load CSV '{CSV_PATH}': {e}")
    st.stop()

required_cols = ["Total_Active_Power", "Total_Reac_Power"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"CSV is missing expected columns: {missing}")
    st.stop()


# ----------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Power Distribution DT Dashboard")
    nav = st.radio("Navigation", ["Home", "Feeder A", "Feeder B", "Feeder C"])

    st.markdown("---")
    st.subheader("Simulation Settings")
    start_date = st.date_input("Start date (if no timestamp)", value=datetime.date(2024, 1, 1))
    update_interval = st.slider("Update interval (seconds)", 1, 10, 2)
    window_size = st.slider("Chart window (points shown)", 20, 2000, 200)
    loop_dataset = st.checkbox("Loop dataset", value=False)

    st.markdown("---")
    start_btn = st.button("▶ Start")
    stop_btn = st.button("⏸ Stop")
    reset_btn = st.button("↺ Reset")

# ----------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "start_date" not in st.session_state:
    st.session_state.start_date = start_date
if "selected_bus_a" not in st.session_state:
    st.session_state.selected_bus_a = None
if "selected_bus_b" not in st.session_state:
    st.session_state.selected_bus_b = None
if "selected_bus_c" not in st.session_state:
    st.session_state.selected_bus_c = None

# control buttons
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False
if reset_btn:
    st.session_state.running = False
    st.session_state.idx = 0

# update start date change
if start_date != st.session_state.start_date:
    st.session_state.start_date = start_date
    st.session_state.idx = 0

# ----------------------------------------------------------
# NAV: HOME
# ----------------------------------------------------------
if nav == "Home":
    st.header("🏠 Home — Real-Time Power Flow Dashboard")
    st.write("Simulating smooth, real-time Total Active (kW) and Reactive (kVAR) power.")

    # prepare dataframe with timestamp index (if data has no timestamp)
    n = len(df_raw)
    idx = pd.date_range(start=st.session_state.start_date, periods=n, freq="h")
    df = df_raw.copy()
    df.index = idx

    # header / top row
    top = st.container()
    with top:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("### ⚡ Real-Time Total Power Monitoring")
            st.caption("Streaming one timestamp at a time for simulated live visualization.")
        with c2:
            st.metric("Total Data Points", f"{n:,}")

    # placeholders for in-place updates
    gauge_col1, gauge_col2 = st.columns(2)
    metric_col1, metric_col2 = st.columns(2)
    chart_col1, chart_col2 = st.columns(2)

    gauge_ph_1 = gauge_col1.empty()
    gauge_ph_2 = gauge_col2.empty()
    metric_ph_1 = metric_col1.empty()
    metric_ph_2 = metric_col2.empty()
    chart_ph_1 = chart_col1.empty()
    chart_ph_2 = chart_col2.empty()
    status_ph = st.empty()

    # initial draw (so layout is stable)
    current_idx = max(0, min(st.session_state.idx, n - 1))
    history = df.iloc[:current_idx + 1] if current_idx >= 0 else pd.DataFrame()
    if history.empty:
        chart_ph_1.plotly_chart(make_line_fig([], [], "Total Active Power (kW)", "kW"), use_container_width=True)
        chart_ph_2.plotly_chart(make_line_fig([], [], "Total Reactive Power (kVAR)", "kVAR"), use_container_width=True)
        gauge_ph_1.plotly_chart(make_gauge(0, "Active Power", "kW", "#4CAF50"), use_container_width=True)
        gauge_ph_2.plotly_chart(make_gauge(0, "Reactive Power", "kVAR", "#2196F3"), use_container_width=True)

    # live update loop for Home (updates only while on Home and running True)
    try:
        while st.session_state.running and nav == "Home":
            # bounds check
            if st.session_state.idx >= n:
                if loop_dataset:
                    st.session_state.idx = 0
                else:
                    st.session_state.running = False
                    status_ph.success("✅ Reached end of dataset. Press Reset or enable Loop to restart.")
                    break

            current_idx = max(0, min(st.session_state.idx, n - 1))
            history = df.iloc[:current_idx + 1]

            # ensure non-empty
            if history.empty:
                active_val = 0.0
                reactive_val = 0.0
                delta_active = 0.0
                delta_reactive = 0.0
            else:
                active_val = history["Total_Active_Power"].iloc[-1]
                reactive_val = history["Total_Reac_Power"].iloc[-1]
                if len(history) >= 2:
                    delta_active = active_val - history["Total_Active_Power"].iloc[-2]
                    delta_reactive = reactive_val - history["Total_Reac_Power"].iloc[-2]

            metric_ph_1.metric("Total Active Power (kW)", f"{active_val:,.2f}", f"{delta_active:+.2f}")
            metric_ph_2.metric("Total Reactive Power (kVAR)", f"{reactive_val:,.2f}", f"{delta_reactive:+.2f}")

            # update gauges
            gauge_ph_1.plotly_chart(make_gauge(active_val, "Active Power", "kW", "#4CAF50"), use_container_width=True)
            gauge_ph_2.plotly_chart(make_gauge(reactive_val, "Reactive Power", "kVAR", "#2196F3"), use_container_width=True)

            # update line charts
            start_window = max(0, len(history) - window_size)
            hist_win = history.iloc[start_window:]

            fig_active = make_line_fig(
                hist_win.index,
                hist_win["Total_Active_Power"] if not hist_win.empty else [],
                "Total Active Power Over Time",
                "Active Power (kW)"
            )
            fig_reactive = make_line_fig(
                hist_win.index,
                hist_win["Total_Reac_Power"] if not hist_win.empty else [],
                "Total Reactive Power Over Time",
                "Reactive Power (kVAR)"
            )
            chart_ph_1.plotly_chart(fig_active, use_container_width=True)
            chart_ph_2.plotly_chart(fig_reactive, use_container_width=True)

            st.session_state.idx += 1
            status_ph.info(f"Streaming Home... Row {min(st.session_state.idx, n)}/{n}  •  Next update in {update_interval}s")
            time.sleep(update_interval)
            # Rerun to update placeholders without full page reload
            st.rerun()
    except Exception as e:
        status_ph.error(f"Home loop stopped: {e}")
    finally:
        if not st.session_state.running:
            status_ph.info("Simulation paused for Home. Press ▶ Start to resume or ↺ Reset.")
        else:
            status_ph.info("Stopped streaming Home (tab changed or error).")

# ----------------------------------------------------------
# NAV: FEEDER A
# ----------------------------------------------------------
elif nav == "Feeder A":
    st.header("🔌 Feeder A — Bus-Level Power & Voltage Monitoring")
    st.write("Smooth real-time Active (kW), Reactive (kVAR), and Voltage (PU) updates for Feeder A.")

    # load feeder A data
    try:
        df_p = load_feeder_a_p()
        df_q = load_feeder_a_q()
        df_v = load_feeder_a_v()
    except Exception as e:
        st.error(f"Could not load Feeder A CSVs: {e}")
        st.stop()

    buses = df_p.columns.tolist() if not df_p.empty else []
    if "selected_bus_a" not in st.session_state or st.session_state.selected_bus_a not in buses:
        st.session_state.selected_bus_a = buses[0] if buses else None

    selected_bus = st.selectbox("Select Bus", options=buses, index=buses.index(st.session_state.selected_bus_a) if st.session_state.selected_bus_a in buses else 0)
    st.session_state.selected_bus_a = selected_bus

    # ensure timestamp index
    n = len(df_p)
    idx = pd.date_range(start=st.session_state.start_date, periods=n, freq="h")
    df_p.index = idx
    df_q.index = idx
    df_v.index = idx

    # header / top row
    top = st.container()
    with top:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### 📊 {selected_bus} Power & Voltage Monitoring")
            st.caption("Streaming one timestamp at a time for simulated live visualization.")
        with c2:
            st.metric("Total Data Points", f"{n:,}")

    # placeholders
    metric_col1, metric_col2 = st.columns(2)
    metric_ph_1 = metric_col1.empty()
    metric_ph_2 = metric_col2.empty()

    power_ph = st.empty()
    voltage_ph = st.empty()
    status_ph = st.empty()

    # initial plots
    power_ph.plotly_chart(make_dual_line_fig([], [], [], f"{selected_bus} Active & Reactive Power", "kW", "kVAR"), use_container_width=True)
    voltage_ph.plotly_chart(make_voltage_fig([], pd.DataFrame(), [], [], selected_bus), use_container_width=True)

    # live update loop for Feeder A while user is on this tab
    try:
        while st.session_state.running and nav == "Feeder A":
            if st.session_state.idx >= n:
                if loop_dataset:
                    st.session_state.idx = 0
                else:
                    st.session_state.running = False
                    status_ph.success("✅ Reached end of Feeder A dataset.")
                    break

            current_idx = max(0, min(st.session_state.idx, n - 1))
            history_p = df_p.iloc[:current_idx + 1]
            history_q = df_q.iloc[:current_idx + 1]
            history_v = df_v.iloc[:current_idx + 1]

            # get active/reactive values for selected bus (guard missing columns)
            if selected_bus not in df_p.columns or selected_bus not in df_q.columns:
                status_ph.error(f"Selected bus '{selected_bus}' not found in Feeder A data.")
                break

            active_val = history_p[selected_bus].iloc[-1] if not history_p.empty else 0.0
            reactive_val = history_q[selected_bus].iloc[-1] if not history_q.empty else 0.0

            if len(history_p) >= 2:
                delta_active = active_val - history_p[selected_bus].iloc[-2]
            else:
                delta_active = 0.0
            if len(history_q) >= 2:
                delta_reactive = reactive_val - history_q[selected_bus].iloc[-2]
            else:
                delta_reactive = 0.0

            metric_ph_1.metric(f"{selected_bus} Active Power (kW)", f"{active_val:,.2f}", f"{delta_active:+.2f}")
            metric_ph_2.metric(f"{selected_bus} Reactive Power (kVAR)", f"{reactive_val:,.2f}", f"{delta_reactive:+.2f}")

            # update power dual-axis chart
            start_window = max(0, len(history_p) - window_size)
            hist_p_win = history_p.iloc[start_window:]
            hist_q_win = history_q.iloc[start_window:]

            fig_power = make_dual_line_fig(
                hist_p_win.index,
                hist_p_win[selected_bus] if not hist_p_win.empty else [],
                hist_q_win[selected_bus] if not hist_q_win.empty else [],
                f"{selected_bus} Active & Reactive Power Over Time",
                "Active Power (kW)",
                "Reactive Power (kVAR)"
            )
            power_ph.plotly_chart(fig_power, use_container_width=True)

            # voltage plot (we expect columns like 'busX.1' etc. adjust if needed)
            bus_num = selected_bus.lstrip("Bus ")
            bus_lower = f"bus{bus_num}"
            v_phases = [f"{bus_lower}.{i}" for i in ['1', '2', '3']]
            t_taps = [f"t_{bus_lower}_l.{i}" for i in ['1', '2', '3']]

            # If voltage columns missing, show warning but continue
            missing_v = [col for col in v_phases + t_taps if col not in df_v.columns]
            if missing_v:
                voltage_ph.warning(f"Voltage/tap columns missing for {selected_bus}: {missing_v}.")
            else:
                hist_v_win = history_v.iloc[start_window:]
                fig_voltage = make_voltage_fig(hist_v_win.index, hist_v_win, v_phases, t_taps, selected_bus)
                voltage_ph.plotly_chart(fig_voltage, use_container_width=True)

            st.session_state.idx += 1
            status_ph.info(f"Streaming Feeder A... Row {min(st.session_state.idx, n)}/{n}  •  Next update in {update_interval}s")
            time.sleep(update_interval)
            st.rerun()
    except Exception as e:
        status_ph.error(f"Feeder A loop stopped: {e}")
    finally:
        if not st.session_state.running:
            status_ph.info("Simulation paused for Feeder A. Press ▶ Start to resume or ↺ Reset.")
        else:
            status_ph.info("Stopped streaming Feeder A (tab changed or error).")

# ----------------------------------------------------------
# NAV: FEEDER B
# ----------------------------------------------------------
elif nav == "Feeder B":
    st.header("🔌 Feeder B — Bus-Level Power & Voltage Monitoring")
    st.write("Smooth real-time Active (kW), Reactive (kVAR), and Voltage (PU) updates for Feeder B.")

    # load feeder B data
    try:
        df_p = load_feeder_b_p()
        df_q = load_feeder_b_q()
        df_v = load_feeder_b_v()
    except Exception as e:
        st.error(f"Could not load Feeder B CSVs: {e}")
        st.stop()

    buses = df_p.columns.tolist() if not df_p.empty else []
    if "selected_bus_b" not in st.session_state or st.session_state.selected_bus_b not in buses:
        st.session_state.selected_bus_b = buses[0] if buses else None

    selected_bus = st.selectbox("Select Bus", options=buses, index=buses.index(st.session_state.selected_bus_b) if st.session_state.selected_bus_b in buses else 0)
    st.session_state.selected_bus_b = selected_bus

    # ensure timestamp index
    n = len(df_p)
    idx = pd.date_range(start=st.session_state.start_date, periods=n, freq="h")
    df_p.index = idx
    df_q.index = idx
    df_v.index = idx

    # placeholders
    top = st.container()
    with top:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### 📊 {selected_bus} Power & Voltage Monitoring")
            st.caption("Streaming one timestamp at a time for simulated live visualization.")
        with c2:
            st.metric("Total Data Points", f"{n:,}")

    metric_col1, metric_col2 = st.columns(2)
    metric_ph_1 = metric_col1.empty()
    metric_ph_2 = metric_col2.empty()

    power_ph = st.empty()
    voltage_ph = st.empty()
    status_ph = st.empty()

    # initial plots
    power_ph.plotly_chart(make_dual_line_fig([], [], [], f"{selected_bus} Active & Reactive Power", "kW", "kVAR"), use_container_width=True)
    voltage_ph.plotly_chart(make_voltage_fig([], pd.DataFrame(), [], [], selected_bus), use_container_width=True)

    # live update loop for Feeder B
    try:
        while st.session_state.running and nav == "Feeder B":
            if st.session_state.idx >= n:
                if loop_dataset:
                    st.session_state.idx = 0
                else:
                    st.session_state.running = False
                    status_ph.success("✅ Reached end of Feeder B dataset.")
                    break

            current_idx = max(0, min(st.session_state.idx, n - 1))
            history_p = df_p.iloc[:current_idx + 1]
            history_q = df_q.iloc[:current_idx + 1]
            history_v = df_v.iloc[:current_idx + 1]

            if selected_bus not in df_p.columns or selected_bus not in df_q.columns:
                status_ph.error(f"Selected bus '{selected_bus}' not found in Feeder B data.")
                break

            active_val = history_p[selected_bus].iloc[-1] if not history_p.empty else 0.0
            reactive_val = history_q[selected_bus].iloc[-1] if not history_q.empty else 0.0

            if len(history_p) >= 2:
                delta_active = active_val - history_p[selected_bus].iloc[-2]
            else:
                delta_active = 0.0
            if len(history_q) >= 2:
                delta_reactive = reactive_val - history_q[selected_bus].iloc[-2]
            else:
                delta_reactive = 0.0

            metric_ph_1.metric(f"{selected_bus} Active Power (kW)", f"{active_val:,.2f}", f"{delta_active:+.2f}")
            metric_ph_2.metric(f"{selected_bus} Reactive Power (kVAR)", f"{reactive_val:,.2f}", f"{delta_reactive:+.2f}")

            # update power plots
            start_window = max(0, len(history_p) - window_size)
            hist_p_win = history_p.iloc[start_window:]
            hist_q_win = history_q.iloc[start_window:]

            fig_power = make_dual_line_fig(
                hist_p_win.index,
                hist_p_win[selected_bus] if not hist_p_win.empty else [],
                hist_q_win[selected_bus] if not hist_q_win.empty else [],
                f"{selected_bus} Active & Reactive Power Over Time",
                "Active Power (kW)",
                "Reactive Power (kVAR)"
            )
            power_ph.plotly_chart(fig_power, use_container_width=True)

            # voltage plot
            bus_num = selected_bus.lstrip("Bus ")
            bus_lower = f"bus{bus_num}"
            v_phases = [f"{bus_lower}.{i}" for i in ['1', '2', '3']]
            t_taps = [f"t_{bus_lower}_l.{i}" for i in ['1', '2', '3']]

            missing_v = [col for col in v_phases + t_taps if col not in df_v.columns]
            if missing_v:
                voltage_ph.warning(f"Voltage/tap columns missing for {selected_bus}: {missing_v}.")
            else:
                hist_v_win = history_v.iloc[start_window:]
                fig_voltage = make_voltage_fig(hist_v_win.index, hist_v_win, v_phases, t_taps, selected_bus)
                voltage_ph.plotly_chart(fig_voltage, use_container_width=True)

            st.session_state.idx += 1
            status_ph.info(f"Streaming Feeder B... Row {min(st.session_state.idx, n)}/{n}  •  Next update in {update_interval}s")
            time.sleep(update_interval)
            st.rerun()
    except Exception as e:
        status_ph.error(f"Feeder B loop stopped: {e}")
    finally:
        if not st.session_state.running:
            status_ph.info("Simulation paused for Feeder B. Press ▶ Start to resume or ↺ Reset.")
        else:
            status_ph.info("Stopped streaming Feeder B (tab changed or error).")

# ----------------------------------------------------------
# NAV: FEEDER C
# ----------------------------------------------------------
elif nav == "Feeder C":
    st.header("🔌 Feeder C — Bus-Level Power & Voltage Monitoring")
    st.write("Smooth real-time Active (kW), Reactive (kVAR), and Voltage (PU) updates for Feeder C.")

    # load feeder C data
    try:
        df_p = load_feeder_c_p()
        df_q = load_feeder_c_q()
        df_v = load_feeder_c_v()
    except Exception as e:
        st.error(f"Could not load Feeder C CSVs: {e}")
        st.stop()

    buses = df_p.columns.tolist() if not df_p.empty else []
    if "selected_bus_c" not in st.session_state or st.session_state.selected_bus_c not in buses:
        st.session_state.selected_bus_c = buses[0] if buses else None

    selected_bus = st.selectbox("Select Bus", options=buses, index=buses.index(st.session_state.selected_bus_c) if st.session_state.selected_bus_c in buses else 0)
    st.session_state.selected_bus_c = selected_bus

    # ensure timestamp index
    n = len(df_p)
    idx = pd.date_range(start=st.session_state.start_date, periods=n, freq="h")
    df_p.index = idx
    df_q.index = idx
    df_v.index = idx

    # header / top row
    top = st.container()
    with top:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### 📊 {selected_bus} Power & Voltage Monitoring")
            st.caption("Streaming one timestamp at a time for simulated live visualization.")
        with c2:
            st.metric("Total Data Points", f"{n:,}")

    # placeholders
    metric_col1, metric_col2 = st.columns(2)
    metric_ph_1 = metric_col1.empty()
    metric_ph_2 = metric_col2.empty()

    power_ph = st.empty()
    voltage_ph = st.empty()
    status_ph = st.empty()

    # initial plots
    power_ph.plotly_chart(make_dual_line_fig([], [], [], f"{selected_bus} Active & Reactive Power", "kW", "kVAR"), use_container_width=True)
    voltage_ph.plotly_chart(make_voltage_fig([], pd.DataFrame(), [], [], selected_bus), use_container_width=True)

    # live update loop for Feeder C
    try:
        while st.session_state.running and nav == "Feeder C":
            if st.session_state.idx >= n:
                if loop_dataset:
                    st.session_state.idx = 0
                else:
                    st.session_state.running = False
                    status_ph.success("✅ Reached end of Feeder C dataset.")
                    break

            current_idx = max(0, min(st.session_state.idx, n - 1))
            history_p = df_p.iloc[:current_idx + 1]
            history_q = df_q.iloc[:current_idx + 1]
            history_v = df_v.iloc[:current_idx + 1]

            if selected_bus not in df_p.columns or selected_bus not in df_q.columns:
                status_ph.error(f"Selected bus '{selected_bus}' not found in Feeder C data.")
                break

            active_val = history_p[selected_bus].iloc[-1] if not history_p.empty else 0.0
            reactive_val = history_q[selected_bus].iloc[-1] if not history_q.empty else 0.0

            if len(history_p) >= 2:
                delta_active = active_val - history_p[selected_bus].iloc[-2]
            else:
                delta_active = 0.0
            if len(history_q) >= 2:
                delta_reactive = reactive_val - history_q[selected_bus].iloc[-2]
            else:
                delta_reactive = 0.0

            metric_ph_1.metric(f"{selected_bus} Active Power (kW)", f"{active_val:,.2f}", f"{delta_active:+.2f}")
            metric_ph_2.metric(f"{selected_bus} Reactive Power (kVAR)", f"{reactive_val:,.2f}", f"{delta_reactive:+.2f}")

            # update power plots
            start_window = max(0, len(history_p) - window_size)
            hist_p_win = history_p.iloc[start_window:]
            hist_q_win = history_q.iloc[start_window:]

            fig_power = make_dual_line_fig(
                hist_p_win.index,
                hist_p_win[selected_bus] if not hist_p_win.empty else [],
                hist_q_win[selected_bus] if not hist_q_win.empty else [],
                f"{selected_bus} Active & Reactive Power Over Time",
                "Active Power (kW)",
                "Reactive Power (kVAR)"
            )
            power_ph.plotly_chart(fig_power, use_container_width=True)

            # voltage plot
            bus_num = selected_bus.lstrip("Bus ")
            bus_lower = f"bus{bus_num}"
            v_phases = [f"{bus_lower}.{i}" for i in ['1', '2', '3']]
            t_taps = [f"t_{bus_lower}_l.{i}" for i in ['1', '2', '3']]

            missing_v = [col for col in v_phases + t_taps if col not in df_v.columns]
            if missing_v:
                voltage_ph.warning(f"Voltage/tap columns missing for {selected_bus}: {missing_v}.")
            else:
                hist_v_win = history_v.iloc[start_window:]
                fig_voltage = make_voltage_fig(hist_v_win.index, hist_v_win, v_phases, t_taps, selected_bus)
                voltage_ph.plotly_chart(fig_voltage, use_container_width=True)

            st.session_state.idx += 1
            status_ph.info(f"Streaming Feeder C... Row {min(st.session_state.idx, n)}/{n}  •  Next update in {update_interval}s")
            time.sleep(update_interval)
            st.rerun()
    except Exception as e:
        status_ph.error(f"Feeder C loop stopped: {e}")
    finally:
        if not st.session_state.running:
            status_ph.info("Simulation paused for Feeder C. Press ▶ Start to resume or ↺ Reset.")
        else:
            status_ph.info("Stopped streaming Feeder C (tab changed or error).")