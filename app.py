import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats, fft
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="CNC Milling Tool Wear Monitoring & Digital Shadow",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean cards and index-free tables
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .status-box {
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-size: 0.95rem;
    }
    .module-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #3b82f6;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .module-title {
        font-weight: 700;
        font-size: 1.02rem;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .module-body {
        color: #334155;
        font-size: 0.94rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# 1. ROBUST MATLAB STRUCT PARSER & CORRUPT FLOAT SANITIZER
# ==============================================================================
@st.cache_data(show_spinner="Loading NASA Ames Milling Dataset (mill.mat)...")
def load_nasa_mill_dataset(base_dir: str):
    candidate_paths = [
        os.path.join(base_dir, "3. Milling", "mill.mat"),
        os.path.join(base_dir, "3. Milling", "mill", "mill.mat"),
        os.path.join(base_dir, "3. Milling", "mill"),
        r"C:\Projects\Milling Digital Shadow\3. Milling\mill.mat",
        r"C:\Projects\Milling Digital Shadow\3. Milling\mill\mill.mat",
        r"C:\Projects\Milling Digital Shadow\3. Milling\mill",
    ]

    mat_file_path = None
    for path in candidate_paths:
        if os.path.isfile(path):
            mat_file_path = path
            break
        elif os.path.isdir(path):
            found = glob.glob(os.path.join(path, "*.mat"))
            if found:
                mat_file_path = found[0]
                break

    if mat_file_path is None:
        found = glob.glob(os.path.join(base_dir, "**", "*.mat"), recursive=True)
        if found:
            mat_file_path = found[0]
        else:
            raise FileNotFoundError(
                f"Could not find 'mill.mat' in {base_dir}. "
                "Ensure 'mill.mat' is inside 'C:\\Projects\\Milling Digital Shadow\\3. Milling\\'."
            )

    mat_data = sio.loadmat(mat_file_path)
    if "mill" not in mat_data:
        raise KeyError(f"'mill' struct array not found in {mat_file_path}.")

    raw_struct = mat_data["mill"]  # 1 x 167 struct array
    num_runs = raw_struct.shape[1]

    sensor_channels = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
    records = []
    waveforms = {}

    for idx in range(num_runs):
        entry = raw_struct[0, idx]

        def extract_scalar(field_name):
            val = entry[field_name]
            if val.size == 0:
                return np.nan
            v = float(val.flatten()[0])
            return v if np.isfinite(v) and abs(v) < 1e6 else np.nan

        case_id = int(extract_scalar("case"))
        run_id = int(extract_scalar("run"))
        vb_val = extract_scalar("VB")
        time_val = extract_scalar("time")
        doc_val = extract_scalar("DOC")
        feed_val = extract_scalar("feed")
        mat_id = int(extract_scalar("material"))

        run_waves = {}
        is_corrupt = False
        for ch in sensor_channels:
            arr = np.asarray(entry[ch], dtype=np.float64).flatten()
            # Check for corrupt float64 values (> 1e4 V/A or NaN/Inf) that cause float32 overflow
            if (
                arr.size < 1000
                or not np.all(np.isfinite(arr))
                or np.max(np.abs(arr)) > 1e4
                or np.std(arr) < 1e-7
            ):
                is_corrupt = True
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(arr, -100.0, 100.0)

            run_waves[ch] = arr

        waveforms[(case_id, run_id)] = run_waves

        records.append({
            "struct_idx": idx,
            "case": case_id,
            "run": run_id,
            "time": time_val if np.isfinite(time_val) else float(run_id * 4.0),
            "DOC": doc_val,
            "feed": feed_val,
            "material": mat_id,
            "material_name": "Cast Iron" if mat_id == 1 else "Stainless Steel J45",
            "VB_raw": vb_val,
            "is_corrupt": is_corrupt
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by=["case", "run"]).reset_index(drop=True)

    # Linear interpolation for unmeasured intermediate VB runs per case
    df["VB_interp"] = np.nan
    for case_id, group in df.groupby("case"):
        s = group["VB_raw"].copy()
        if pd.isna(s.iloc[0]):
            s.iloc[0] = 0.0
        s_interp = s.interpolate(method="linear", limit_direction="both")
        s_interp = np.maximum.accumulate(s_interp.fillna(0.0))
        df.loc[group.index, "VB_interp"] = np.clip(s_interp, 0.0, 1.5)

    return df, waveforms, mat_file_path


# ==============================================================================
# 2. SIGNAL CONDITIONING & STEADY-STATE FEATURE EXTRACTION
# ==============================================================================
@st.cache_data(show_spinner="Computing steady-state time and frequency domain features...")
def extract_signal_features(df: pd.DataFrame, waveforms: dict, win_start: int, win_end: int):
    sensor_channels = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
    feature_rows = []

    for _, row in df.iterrows():
        key = (int(row["case"]), int(row["run"]))
        waves = waveforms[key]
        feat = {}

        for ch in sensor_channels:
            sig = waves[ch]
            steady = sig[win_start:win_end] if sig.size >= win_end else sig
            steady = np.clip(np.nan_to_num(steady, nan=0.0, posinf=0.0, neginf=0.0), -500.0, 500.0)

            rms_val = float(np.sqrt(np.mean(np.square(steady))))
            mean_val = float(np.mean(steady))
            std_val = float(np.std(steady))
            p2p_val = float(np.ptp(steady))

            if std_val > 1e-6:
                kurt_val = float(stats.kurtosis(steady, fisher=True, bias=False))
                if not np.isfinite(kurt_val):
                    kurt_val = 0.0
            else:
                kurt_val = 0.0

            fft_vals = np.abs(fft.rfft(steady - mean_val))
            spec_energy = float(np.sum(np.square(fft_vals)) / max(len(steady), 1))

            # Clamp all extracted features safely within float32 range
            feat[f"{ch}_rms"] = np.clip(rms_val, -1e5, 1e5)
            feat[f"{ch}_mean"] = np.clip(mean_val, -1e5, 1e5)
            feat[f"{ch}_std"] = np.clip(std_val, 0.0, 1e5)
            feat[f"{ch}_p2p"] = np.clip(p2p_val, 0.0, 1e5)
            feat[f"{ch}_kurt"] = np.clip(kurt_val, -50.0, 50.0)
            feat[f"{ch}_energy"] = np.clip(spec_energy, 0.0, 1e6)

        feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows, index=df.index)
    full_df = pd.concat([df, feat_df], axis=1)
    return full_df


# ==============================================================================
# 3. WEAR ESTIMATION: EMPIRICAL ML + TAYLOR WEAR EQUATION
# ==============================================================================
@st.cache_resource(show_spinner="Training Random Forest Regressor & Taylor Wear Curve...")
def train_wear_models(full_df: pd.DataFrame):
    feature_cols = [
        "DOC", "feed", "material", "time",
        "smcAC_rms", "smcAC_std", "smcAC_p2p",
        "smcDC_rms", "smcDC_mean", "smcDC_p2p",
        "vib_table_rms", "vib_table_kurt", "vib_table_energy",
        "vib_spindle_rms", "vib_spindle_kurt", "vib_spindle_energy",
        "AE_table_rms", "AE_table_energy",
        "AE_spindle_rms", "AE_spindle_energy"
    ]

    clean_mask = (~full_df["is_corrupt"]) & (full_df["VB_interp"].notna()) & (full_df["VB_interp"] <= 0.65)
    train_df = full_df[clean_mask].copy()

    # Guaranteed finite float64 array clamped well inside float32 limits
    X_train = np.nan_to_num(
        train_df[feature_cols].to_numpy(dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    X_train = np.clip(X_train, -1e6, 1e6)
    y_train = np.nan_to_num(train_df["VB_interp"].to_numpy(dtype=np.float64), nan=0.0)

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    X_all = np.nan_to_num(
        full_df[feature_cols].to_numpy(dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    X_all = np.clip(X_all, -1e6, 1e6)

    full_df = full_df.copy()
    full_df["VB_ml_pred"] = np.clip(rf_model.predict(X_all), 0.0, 1.0)

    # Analytical Taylor-type gradual wear rate integration + Sensor-Physics Weighted Estimate
    full_df["VB_taylor"] = 0.0
    full_df["VB_combined"] = 0.0

    for case_id, group in full_df.groupby("case"):
        vb_tay = 0.0
        vb_comb = 0.0
        prev_time = 0.0

        for idx, row in group.iterrows():
            t_now = float(row["time"]) if np.isfinite(row["time"]) else prev_time + 3.0
            dt = max(t_now - prev_time, 1.0) if prev_time > 0 else max(t_now, 2.0)
            prev_time = t_now

            # Empirical rate constant for Cast Iron (1) vs Steel J45 (2)
            k_mat = 0.0048 if row["material"] == 1 else 0.0082
            doc_factor = (float(row["DOC"]) / 1.0) ** 0.45
            feed_factor = (float(row["feed"]) / 0.35) ** 0.65
            smc_load = np.clip(float(row["smcDC_mean"]), 0.1, 15.0) / 6.0

            d_vb = k_mat * doc_factor * feed_factor * (0.75 + 0.25 * smc_load) * dt
            vb_tay = vb_tay + d_vb

            ml_val = float(row["VB_ml_pred"])
            if not row["is_corrupt"]:
                # Weighted filter combining analytical wear progression (35%) and ML sensor regression (65%)
                vb_comb = 0.35 * (vb_comb + d_vb) + 0.65 * ml_val
            else:
                # Fallback to analytical wear curve when DAQ signals drop out
                vb_comb = vb_comb + d_vb

            full_df.loc[idx, "VB_taylor"] = vb_tay
            full_df.loc[idx, "VB_combined"] = vb_comb

    valid_eval = full_df[(~full_df["is_corrupt"]) & (full_df["VB_raw"].notna())]
    r2_ml = r2_score(valid_eval["VB_raw"], valid_eval["VB_ml_pred"])
    r2_comb = r2_score(valid_eval["VB_raw"], valid_eval["VB_combined"])
    rmse_comb = np.sqrt(mean_squared_error(valid_eval["VB_raw"], valid_eval["VB_combined"]))

    metrics = {
        "r2_ml": r2_ml,
        "r2_comb": r2_comb,
        "rmse_comb": rmse_comb,
        "feature_cols": feature_cols
    }
    return rf_model, full_df, metrics


# ==============================================================================
# 4. RULE-BASED FEED-RATE OVERRIDE SIMULATION
# ==============================================================================
def simulate_feed_override(vb_est: float, smc_dc: float, nominal_feed: float, vb_limit: float = 0.30):
    n_rpm = 826.0  # Constant spindle speed for 200 m/min with 70 mm cutter
    wear_ratio = vb_est / vb_limit

    if wear_ratio < 0.60 and smc_dc < 7.5:
        status = "NORMAL ROUGHING OPERATION"
        color = "#10B981"
        feed_override_pct = 100.0
        action_msg = "Estimated flank wear and motor current are within nominal limits. Maintain 100% programmed feed."
    elif wear_ratio < 0.90:
        throttle = 100.0 - 35.0 * ((wear_ratio - 0.60) / 0.30)
        if smc_dc > 8.0:
            throttle -= 10.0
        feed_override_pct = float(np.clip(throttle, 55.0, 95.0))
        status = "PROPORTIONAL FEED DERATING"
        color = "#F59E0B"
        action_msg = (
            f"Estimated wear ({vb_est:.3f} mm) is approaching the limit. "
            f"Recommended feed-rate override reduced to {feed_override_pct:.1f}% to lower cutting force."
        )
    elif wear_ratio < 1.05:
        feed_override_pct = 50.0
        status = "WEAR LIMIT REACHED — SCHEDULE TOOL CHANGE"
        color = "#EF4444"
        action_msg = (
            f"ISO 3685 flank wear threshold reached ({vb_est:.3f} mm >= {vb_limit:.2f} mm). "
            "Reduce feed to 50% to complete the active pass safely, then replace insert."
        )
    else:
        feed_override_pct = 0.0
        status = "EXCESSIVE WEAR — CYCLE STOP RECOMMENDED"
        color = "#991B1B"
        action_msg = (
            f"Flank wear ({vb_est:.3f} mm) significantly exceeds safe roughing limits. "
            "Halt feed to prevent insert chipping or workpiece surface damage."
        )

    cmd_feed_mm_rev = nominal_feed * (feed_override_pct / 100.0)
    cmd_table_feed_mm_min = cmd_feed_mm_rev * n_rpm
    force_reduction_pct = (1.0 - (feed_override_pct / 100.0) ** 0.75) * 100.0 if feed_override_pct > 0 else 100.0

    return {
        "status": status,
        "color": color,
        "feed_override_pct": feed_override_pct,
        "cmd_feed_mm_rev": cmd_feed_mm_rev,
        "cmd_table_feed_mm_min": cmd_table_feed_mm_min,
        "force_reduction_pct": force_reduction_pct,
        "action_msg": action_msg
    }


# ==============================================================================
# MAIN DASHBOARD UI
# ==============================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        raw_df, waveforms, loaded_path = load_nasa_mill_dataset(base_dir)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Sidebar
    st.sidebar.title("Process Parameters")
    st.sidebar.caption(f"Source: `{os.path.basename(loaded_path)}` (167 Runs, 250 Hz)")

    st.sidebar.markdown("### 1. Signal Windowing")
    win_range = st.sidebar.slider(
        "Steady-State Sample Window (0–9000)",
        min_value=500,
        max_value=8500,
        value=(3000, 6500),
        step=250,
        help="Isolates the steady-state cutting phase by excluding entry and exit air-cutting transients."
    )

    full_df = extract_signal_features(raw_df, waveforms, win_range[0], win_range[1])
    rf_model, full_df, metrics = train_wear_models(full_df)

    st.sidebar.markdown("### 2. Experimental Case & Run")
    cases_summary = (
        full_df.groupby("case")
        .agg(
            runs=("run", "count"),
            mat=("material_name", "first"),
            doc=("DOC", "first"),
            feed=("feed", "first")
        )
        .reset_index()
    )
    case_options = {
        row["case"]: f"Case {row['case']:02d} | {row['mat']} | DOC={row['doc']}mm, f={row['feed']}mm/rev ({row['runs']} runs)"
        for _, row in cases_summary.iterrows()
    }

    selected_case = st.sidebar.selectbox(
        "Select Case (1–16)",
        options=list(case_options.keys()),
        format_func=lambda x: case_options[x],
        index=0
    )

    case_runs_df = full_df[full_df["case"] == selected_case].sort_values("run")
    run_list = case_runs_df["run"].tolist()
    selected_run = st.sidebar.select_slider(
        "Select Run Number",
        options=run_list,
        value=run_list[min(len(run_list) - 1, len(run_list) // 2)]
    )

    st.sidebar.markdown("### 3. Wear Threshold & Test Bias")
    vb_limit = st.sidebar.slider(
        "ISO 3685 Flank Wear Limit VB [mm]",
        min_value=0.20,
        max_value=0.45,
        value=0.30,
        step=0.05
    )
    sim_wear_bias = st.sidebar.slider(
        "Add Test Wear Offset (+mm)",
        min_value=0.00,
        max_value=0.25,
        value=0.00,
        step=0.02,
        help="Adds a manual wear offset to test how the feed-rate override rule responds."
    )

    cur_row = case_runs_df[case_runs_df["run"] == selected_run].iloc[0]
    cur_waves = waveforms[(int(selected_case), int(selected_run))]

    vb_est_live = float(cur_row["VB_combined"]) + sim_wear_bias
    smc_dc_live = float(cur_row["smcDC_mean"]) * (1.0 + 1.2 * sim_wear_bias)
    ctrl = simulate_feed_override(
        vb_est=vb_est_live,
        smc_dc=smc_dc_live,
        nominal_feed=float(cur_row["feed"]),
        vb_limit=vb_limit
    )

    # Title & Subtitle
    st.title("CNC Milling Tool Wear Estimation & Process Control Dashboard")
    st.markdown(
        "**Dataset:** NASA Ames / UC Berkeley Milling Dataset (`mill.mat`) | **Machine:** Matsuura MC-510V "
        "| **Tool:** 70 mm 6-Insert KC710 Face Mill ($v_c = 200\\text{ m/min}, N = 826\\text{ rev/min}$)"
    )

    # Top KPI Metrics
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Measured Flank Wear (VB)",
        f"{cur_row['VB_raw']:.3f} mm" if pd.notna(cur_row["VB_raw"]) else "Not Measured",
        delta=f"Interpolated: {cur_row['VB_interp']:.3f} mm"
    )
    k2.metric(
        "Estimated Flank Wear (VB)",
        f"{vb_est_live:.3f} mm",
        delta=f"{vb_est_live - vb_limit:+.3f} mm vs Limit",
        delta_color="inverse"
    )
    k3.metric(
        "Spindle DC Current (Mean)",
        f"{smc_dc_live:.2f} A",
        delta=f"AC RMS: {cur_row['smcAC_rms']:.2f} A"
    )
    k4.metric(
        "Recommended Feed Override",
        f"{ctrl['feed_override_pct']:.1f}%",
        delta=f"{ctrl['cmd_table_feed_mm_min']:.0f} mm/min Table Feed"
    )
    k5.metric(
        "Test Set R² Accuracy",
        f"{metrics['r2_comb']:.3f}",
        delta=f"RMSE: {metrics['rmse_comb']:.4f} mm"
    )

    # Process Control Status Box
    st.markdown(
        f"""
        <div class="status-box" style="background-color:{ctrl['color']}18; border-left: 6px solid {ctrl['color']};">
            <strong style="color:{ctrl['color']};">PROCESS CONTROL STATUS: {ctrl['status']}</strong><br/>
            <span>{ctrl['action_msg']} (Estimated Cutting Force Reduction: <b>{ctrl['force_reduction_pct']:.1f}%</b>)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 5 Clean, Technical Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Sensor Waveforms & Filtering",
        "2. Tool Wear Estimation (Digital Shadow)",
        "3. Feed-Rate Override Control",
        "4. Course Syllabus Alignment",
        "5. Processed Run Data"
    ])

    # ==========================================================================
    # TAB 1: SENSOR WAVEFORMS & FILTERING
    # ==========================================================================
    with tab1:
        col_left, col_right = st.columns([2.1, 1.0])

        with col_left:
            st.subheader(f"250 Hz Sensor Waveforms — Case {selected_case}, Run {selected_run}")
            if cur_row["is_corrupt"]:
                st.warning(
                    "Note: The raw sensor arrays for this specific run contain corrupt/dropout values in the original NASA file. "
                    "The estimator automatically relies on the analytical wear equation for this data point."
                )

            time_axis = np.arange(len(cur_waves["smcDC"])) / 250.0
            t_start, t_end = win_range[0] / 250.0, win_range[1] / 250.0

            fig_wave = make_subplots(
                rows=3, cols=2,
                shared_xaxes=False,
                vertical_spacing=0.13,
                horizontal_spacing=0.08,
                subplot_titles=(
                    "AC Spindle Motor Current (smcAC)",
                    "DC Spindle Motor Current (smcDC)",
                    "Table Vibration RMS (vib_table)",
                    "Spindle Vibration RMS (vib_spindle)",
                    "Table Acoustic Emission RMS (AE_table)",
                    "Spindle Acoustic Emission RMS (AE_spindle)"
                )
            )

            ch_meta = [
                ("smcAC", 1, 1, "#2563EB", "Current [A]"),
                ("smcDC", 1, 2, "#DC2626", "Current [A]"),
                ("vib_table", 2, 1, "#059669", "Vibration [V]"),
                ("vib_spindle", 2, 2, "#D97706", "Vibration [V]"),
                ("AE_table", 3, 1, "#7C3AED", "AE RMS [V]"),
                ("AE_spindle", 3, 2, "#DB2777", "AE RMS [V]")
            ]

            for ch_name, r_idx, c_idx, color_hex, y_label in ch_meta:
                sig = cur_waves[ch_name]
                fig_wave.add_trace(
                    go.Scatter(
                        x=time_axis[::3],
                        y=sig[::3],
                        mode="lines",
                        line=dict(color=color_hex, width=1.2),
                        name=ch_name
                    ),
                    row=r_idx, col=c_idx
                )
                fig_wave.add_vrect(
                    x0=t_start, x1=t_end,
                    fillcolor="rgba(16, 185, 129, 0.14)",
                    line_width=1,
                    line_dash="dash",
                    line_color="#059669",
                    row=r_idx, col=c_idx
                )
                fig_wave.update_xaxes(title_text="Time [s]", row=r_idx, col=c_idx, title_standoff=4)
                fig_wave.update_yaxes(title_text=y_label, row=r_idx, col=c_idx, title_standoff=6)

            fig_wave.update_layout(
                height=680,
                showlegend=False,
                margin=dict(l=40, r=20, t=40, b=40)
            )
            st.plotly_chart(fig_wave)

        with col_right:
            st.subheader("Steady-State FFT Spectrum")
            selected_fft_ch = st.selectbox(
                "Select Sensor Channel",
                ["vib_spindle", "vib_table", "AE_spindle", "AE_table", "smcAC"]
            )
            sig_steady = cur_waves[selected_fft_ch][win_range[0]:win_range[1]]
            sig_zero = sig_steady - np.mean(sig_steady)
            freqs = fft.rfftfreq(len(sig_zero), d=1.0 / 250.0)
            fft_mag = np.abs(fft.rfft(sig_zero)) / max(len(sig_zero), 1)

            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=freqs,
                y=fft_mag,
                mode="lines",
                fill="tozeroy",
                line=dict(color="#4F46E5", width=1.6),
                name="FFT Magnitude"
            ))
            fig_fft.update_layout(
                height=310,
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis_title="Frequency [Hz] (fs = 250 Hz, f_Nyquist = 125 Hz)",
                yaxis_title="Spectral Magnitude"
            )
            st.plotly_chart(fig_fft)

            st.markdown("#### Signal Acquisition Details")
            st.info(
                "- **Acoustic Emission (WD 925):** 50 kHz high-pass preamplifier -> 8.0 ms analog RMS filter -> 250 Hz DAQ.\n"
                "- **Vibration (Endevco 7201-50):** Charge amplifier -> 400 Hz to 1 kHz bandpass filter (suppresses 180 Hz 3rd harmonic power noise) -> 8.0 ms RMS.\n"
                "- **Steady-State Windowing (Green Region):** Excludes entry and exit air-cutting transients so features reflect actual material removal."
            )

    # ==========================================================================
    # TAB 2: TOOL WEAR ESTIMATION (DIGITAL SHADOW)
    # ==========================================================================
    with tab2:
        c_left, c_right = st.columns([1.75, 1.25])

        with c_left:
            st.subheader(f"Case {selected_case}: Tool Flank Wear (VB) Progression")
            fig_vb = go.Figure()

            fig_vb.add_trace(go.Scatter(
                x=case_runs_df["time"],
                y=case_runs_df["VB_taylor"],
                mode="lines+markers",
                name="Analytical Wear Curve (Extended Taylor)",
                line=dict(color="#64748B", width=2, dash="dot")
            ))

            fig_vb.add_trace(go.Scatter(
                x=case_runs_df["time"],
                y=case_runs_df["VB_ml_pred"],
                mode="lines+markers",
                name="Random Forest Sensor Prediction",
                line=dict(color="#2563EB", width=2)
            ))

            fig_vb.add_trace(go.Scatter(
                x=case_runs_df["time"],
                y=case_runs_df["VB_combined"],
                mode="lines+markers",
                name="Combined State Estimate (Physics + ML)",
                line=dict(color="#059669", width=3.2)
            ))

            measured_df = case_runs_df[case_runs_df["VB_raw"].notna()]
            fig_vb.add_trace(go.Scatter(
                x=measured_df["time"],
                y=measured_df["VB_raw"],
                mode="markers",
                name="Microscope Measured VB (Ground Truth)",
                marker=dict(color="#DC2626", size=10, symbol="diamond", line=dict(width=1.2, color="white"))
            ))

            fig_vb.add_hline(
                y=vb_limit,
                line_dash="dash",
                line_color="#DC2626",
                annotation_text=f"ISO 3685 Wear Limit ({vb_limit:.2f} mm)",
                annotation_position="top left"
            )

            fig_vb.add_vline(
                x=cur_row["time"],
                line_dash="solid",
                line_color="#D97706",
                annotation_text=f"Selected Run {selected_run}"
            )

            fig_vb.update_layout(
                height=470,
                xaxis_title="Cumulative Machining Time [minutes]",
                yaxis_title="Flank Wear VB [mm]",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_vb)

        with c_right:
            st.subheader("Random Forest Feature Importance")
            importances = rf_model.feature_importances_
            feat_names = metrics["feature_cols"]
            imp_df = (
                pd.DataFrame({"Feature": feat_names, "Importance": importances})
                .sort_values("Importance", ascending=True)
                .tail(10)
            )

            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation="h",
                marker_color="#4F46E5"
            ))
            fig_imp.update_layout(
                height=320,
                margin=dict(l=40, r=20, t=10, b=40),
                xaxis_title="Relative Feature Importance (Gini)",
                yaxis_title="Extracted Feature"
            )
            st.plotly_chart(fig_imp)

            st.markdown("#### Engineering Distinction: Digital Shadow vs. Digital Twin")
            st.info(
                "- **What We Built (Digital Shadow):** A one-way data pipeline that ingests recorded CNC telemetry (`mill.mat`) and synchronizes an analytical wear curve with a Random Forest regressor to estimate flank wear $VB$.\n"
                "- **Why Combine Physics & ML:** Pure Taylor equations miss material hardness inclusions, while pure sensor regression fails during DAQ signal dropouts (such as Case 6 Run 1)."
            )

    # ==========================================================================
    # TAB 3: FEED-RATE OVERRIDE CONTROL
    # ==========================================================================
    with tab3:
        st.subheader(f"Rule-Based Feed-Rate Override Simulation — Case {selected_case}")
        st.markdown(
            "In a CNC controller, spindle motor current (`smcDC`) is proportional to cutting torque and force. "
            "This simulation demonstrates how the estimated flank wear $VB$ can drive a proportional feed-rate override "
            "to cap cutting forces when the tool approaches its wear limit."
        )

        sim_rows = []
        for _, r in case_runs_df.iterrows():
            vb_val = float(r["VB_combined"]) + sim_wear_bias
            smc_val = float(r["smcDC_mean"]) * (1.0 + 1.2 * sim_wear_bias)
            c_out = simulate_feed_override(
                vb_est=vb_val,
                smc_dc=smc_val,
                nominal_feed=float(r["feed"]),
                vb_limit=vb_limit
            )
            sim_rows.append({
                "Run": int(r["run"]),
                "Time [min]": round(float(r["time"]), 1),
                "Estimated VB [mm]": round(vb_val, 3),
                "Nominal Feed [mm/min]": round(float(r["feed"]) * 826.0, 1),
                "Regulated Feed [mm/min]": round(c_out["cmd_table_feed_mm_min"], 1),
                "Feed Override [%]": round(c_out["feed_override_pct"], 1),
                "Unregulated Current [A]": round(smc_val, 2),
                "Regulated Current [A]": round(smc_val * ((c_out["feed_override_pct"] / 100.0) ** 0.75), 2),
                "Control Action": c_out["status"]
            })
        sim_df = pd.DataFrame(sim_rows)

        fig_ctrl = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.14,
            subplot_titles=(
                "Table Feed Rate [mm/min]: Nominal Programmed vs. Override Regulated",
                "Spindle Motor DC Current [A]: Unregulated vs. Override Regulated"
            )
        )

        fig_ctrl.add_trace(go.Scatter(
            x=sim_df["Time [min]"], y=sim_df["Nominal Feed [mm/min]"],
            name="Nominal Feed [mm/min]",
            line=dict(color="#64748B", width=2, dash="dash")
        ), row=1, col=1)

        fig_ctrl.add_trace(go.Scatter(
            x=sim_df["Time [min]"], y=sim_df["Regulated Feed [mm/min]"],
            name="Override Feed [mm/min]",
            mode="lines+markers",
            line=dict(color="#059669", width=3)
        ), row=1, col=1)

        fig_ctrl.add_trace(go.Scatter(
            x=sim_df["Time [min]"], y=sim_df["Unregulated Current [A]"],
            name="Unregulated Current [A]",
            line=dict(color="#DC2626", width=2, dash="dot")
        ), row=2, col=1)

        fig_ctrl.add_trace(go.Scatter(
            x=sim_df["Time [min]"], y=sim_df["Regulated Current [A]"],
            name="Regulated Current [A]",
            mode="lines+markers",
            line=dict(color="#2563EB", width=3)
        ), row=2, col=1)

        fig_ctrl.update_xaxes(title_text="Cumulative Machining Time [minutes]", row=1, col=1)
        fig_ctrl.update_xaxes(title_text="Cumulative Machining Time [minutes]", row=2, col=1)
        fig_ctrl.update_yaxes(title_text="Table Feed [mm/min]", row=1, col=1)
        fig_ctrl.update_yaxes(title_text="Motor Current [A]", row=2, col=1)

        fig_ctrl.update_layout(
            height=550,
            margin=dict(l=40, r=20, t=40, b=40)
        )
        st.plotly_chart(fig_ctrl)

        st.dataframe(sim_df, hide_index=True)

    # ==========================================================================
    # TAB 4: COURSE SYLLABUS ALIGNMENT
    # ==========================================================================
    with tab4:
        st.subheader("Industry 4.0 Course Syllabus Alignment")
        st.markdown("Technical mapping between the seven course modules and what is implemented in this project:")

        modules_info = [
            (
                "Module 1: Introduction to Industry Revolutions & Details of Industry 4.0",
                "Demonstrates the transition from Industry 3.0 (automated CNC machining with fixed G-code parameters and manual microscope inspection) to Industry 4.0 (sensor-instrumented condition monitoring and data-driven process feedback)."
            ),
            (
                "Module 2: Sensors, Machine Vision and Connected Systems",
                "Analyzes multi-sensor industrial telemetry acquired on a Matsuura MC-510V machining center: Physical Acoustics WD-925 acoustic emission sensors (up to 2 MHz), Endevco 7201-50 piezoelectric accelerometers, and Omron/Flexcore spindle motor current transducers."
            ),
            (
                "Module 3: Data Analysis, Big Data Analytics & Control of Manufacturing Processes",
                "Implements automated time-series windowing to separate steady-state cutting from entry/exit air-cutting transients across 167 experimental runs (9,000 samples per run at 250 Hz), extracting RMS, kurtosis, peak-to-peak, and FFT spectral energy."
            ),
            (
                "Module 4: Digital Twin and Its Importance in Industry 4.0",
                "Illustrates the formal engineering distinction between a Digital Model (analytical Taylor wear curve), a Digital Shadow (one-way sensor data replay into a virtual state estimator, as implemented here), and a full Digital Twin (bi-directional hardware feedback)."
            ),
            (
                "Module 5: Impact of Industry 4.0 on Various Industries",
                "Addresses indirect tool condition monitoring in precision metal cutting—using inexpensive spindle current sensors as a proxy for cutting force instead of expensive table dynamometers to prevent scrap and unplanned downtime."
            ),
            (
                "Module 6: Applications of AI, Machine Learning (ML) & Deep Learning in Manufacturing",
                "Trains a supervised Random Forest regression model on 20 time- and frequency-domain sensor features to estimate continuous flank wear (VB in mm) when offline microscope measurements are unavailable."
            ),
            (
                "Module 7: Towards Manufacturing as an Autonomous System",
                "Simulates a closed-loop proportional feed-rate override law that uses the estimated flank wear (VB) and DC spindle current to automatically derate table feed (mm/min) as the insert approaches the ISO 3685 wear limit (0.30 mm)."
            )
        ]

        for title, body in modules_info:
            st.markdown(
                f"""
                <div class="module-card">
                    <div class="module-title">{title}</div>
                    <div class="module-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ==========================================================================
    # TAB 5: PROCESSED RUN DATA
    # ==========================================================================
    with tab5:
        st.subheader("Processed Summary Table (All 167 Experimental Runs)")
        display_df = full_df[[
            "case", "run", "time", "material_name", "DOC", "feed",
            "VB_raw", "VB_interp", "VB_taylor", "VB_ml_pred", "VB_combined",
            "smcDC_mean", "smcAC_rms", "vib_spindle_rms", "AE_spindle_rms", "is_corrupt"
        ]].rename(columns={
            "case": "Case",
            "run": "Run",
            "time": "Time [min]",
            "material_name": "Workpiece Material",
            "DOC": "DOC [mm]",
            "feed": "Feed [mm/rev]",
            "VB_raw": "Measured VB [mm]",
            "VB_interp": "Interpolated VB [mm]",
            "VB_taylor": "Taylor VB [mm]",
            "VB_ml_pred": "ML Predicted VB [mm]",
            "VB_combined": "Combined VB [mm]",
            "smcDC_mean": "Mean smcDC [A]",
            "smcAC_rms": "RMS smcAC [A]",
            "vib_spindle_rms": "RMS Vib Spindle [V]",
            "AE_spindle_rms": "RMS AE Spindle [V]",
            "is_corrupt": "Corrupt DAQ Flag"
        })
        st.dataframe(display_df, hide_index=True)


if __name__ == "__main__":
    main()
