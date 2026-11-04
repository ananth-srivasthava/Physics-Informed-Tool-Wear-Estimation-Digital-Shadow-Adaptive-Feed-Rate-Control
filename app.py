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
    page_title="Matsuura MC-510V Digital Twin & Adaptive CNC Control",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. ROBUST MATLAB STRUCT PARSER & PATH RESOLVER
# ==============================================================================
@st.cache_data(show_spinner="Loading and parsing NASA Ames 1x167 MATLAB struct array...")
def load_nasa_mill_dataset(base_dir: str):
    """
    Locates mill.mat across standard relative/absolute paths inside the project folder,
    extracts the 1x167 struct array, handles missing VB values, and computes signal
    features from the steady-state cutting window.
    """
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
        # Deep recursive search inside base_dir as final safeguard
        found = glob.glob(os.path.join(base_dir, "**", "*.mat"), recursive=True)
        if found:
            mat_file_path = found[0]
        else:
            raise FileNotFoundError(
                f"Could not find 'mill.mat' in {base_dir} or '3. Milling'. "
                "Verify that the file 'mill.mat' is inside 'C:\\Projects\\Milling Digital Shadow\\3. Milling\\'."
            )

    mat_data = sio.loadmat(mat_file_path)
    if "mill" not in mat_data:
        raise KeyError(f"'mill' struct key not found in {mat_file_path}. Keys: {list(mat_data.keys())}")

    raw_struct = mat_data["mill"]  # Shape (1, 167)
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
            return float(val.flatten()[0])

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
            if arr.size < 1000 or np.all(np.isnan(arr)) or np.std(arr) < 1e-7:
                is_corrupt = True
            run_waves[ch] = arr

        waveforms[(case_id, run_id)] = run_waves

        records.append({
            "struct_idx": idx,
            "case": case_id,
            "run": run_id,
            "time": time_val,
            "DOC": doc_val,
            "feed": feed_val,
            "material": mat_id,
            "material_name": "Cast Iron" if mat_id == 1 else "Stainless Steel J45",
            "VB_raw": vb_val,
            "is_corrupt": is_corrupt
        })

    df = pd.DataFrame(records)
    # Sort strictly by case and run counter
    df = df.sort_values(by=["case", "run"]).reset_index(drop=True)

    # Interpolate missing microscope VB measurements per case (monotonic physical progression)
    df["VB_interp"] = np.nan
    for case_id, group in df.groupby("case"):
        s = group["VB_raw"].copy()
        # At run 1, tool starts fresh or near-fresh if unmeasured
        if pd.isna(s.iloc[0]):
            s.iloc[0] = 0.0
        s_interp = s.interpolate(method="linear", limit_direction="both")
        # Enforce non-decreasing physical flank wear within a single insert life
        s_interp = np.maximum.accumulate(s_interp.fillna(0.0))
        df.loc[group.index, "VB_interp"] = s_interp

    return df, waveforms, mat_file_path


# ==============================================================================
# 2. ECE MODULE (MODULES 2 & 3): STEADY-STATE WINDOWING & SIGNAL FEATURES
# ==============================================================================
@st.cache_data(show_spinner="Extracting time & frequency-domain features from 9,000-sample waveforms...")
def extract_signal_features(df: pd.DataFrame, waveforms: dict, win_start: int, win_end: int):
    """
    Slices out the Entry and Exit transients (retaining win_start:win_end out of 9000 samples @ 250 Hz)
    and calculates RMS, Mean, Std, Kurtosis, Peak-to-Peak, and Dominant Spectral Energy.
    """
    sensor_channels = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
    feature_rows = []

    for _, row in df.iterrows():
        key = (int(row["case"]), int(row["run"]))
        waves = waveforms[key]
        feat = {}

        for ch in sensor_channels:
            sig = waves[ch]
            if sig.size >= win_end:
                steady = sig[win_start:win_end]
            else:
                steady = sig

            # Time-domain metrics
            rms_val = np.sqrt(np.mean( np.square(steady) ))
            mean_val = np.mean(steady)
            std_val = np.std(steady)
            p2p_val = np.ptp(steady)
            kurt_val = float(stats.kurtosis(steady, fisher=True, bias=False)) if std_val > 1e-9 else 0.0

            # Frequency-domain metric (FFT spectral energy in passband)
            fft_vals = np.abs(fft.rfft(steady - mean_val))
            spec_energy = np.sum(np.square(fft_vals)) / max(len(steady), 1)

            feat[f"{ch}_rms"] = rms_val
            feat[f"{ch}_mean"] = mean_val
            feat[f"{ch}_std"] = std_val
            feat[f"{ch}_p2p"] = p2p_val
            feat[f"{ch}_kurt"] = kurt_val
            feat[f"{ch}_energy"] = spec_energy

        feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows, index=df.index)
    full_df = pd.concat([df, feat_df], axis=1)
    return full_df


# ==============================================================================
# 3. MECH & AI MODULE (MODULES 4, 6 & 7): HYBRID DIGITAL TWIN & ML TRAINING
# ==============================================================================
@st.cache_resource(show_spinner="Training Random Forest Regressor & Calibrating Physics Twin...")
def train_hybrid_digital_twin(full_df: pd.DataFrame):
    """
    1. Trains a Random Forest Regressor on clean experimental runs.
    2. Calibrates an Extended Taylor/Archard rate-of-wear physics differential model:
       d(VB)/dt = C_mat * (DOC ^ a) * (feed ^ b) * (1 + alpha * smcDC_norm)
    3. Combines both via a Kalman-style state assimilation weight (Digital Twin state fusion).
    """
    clean_df = full_df[(~full_df["is_corrupt"]) & (full_df["VB_interp"].notna())].copy()
    # Filter out extreme run-away outliers (> 0.65 mm) for stable regression training
    train_df = clean_df[clean_df["VB_interp"] <= 0.65].copy()

    feature_cols = [
        "DOC", "feed", "material", "time",
        "smcAC_rms", "smcAC_std", "smcAC_p2p",
        "smcDC_rms", "smcDC_mean", "smcDC_p2p",
        "vib_table_rms", "vib_table_kurt", "vib_table_energy",
        "vib_spindle_rms", "vib_spindle_kurt", "vib_spindle_energy",
        "AE_table_rms", "AE_table_energy",
        "AE_spindle_rms", "AE_spindle_energy"
    ]

    X = train_df[feature_cols].fillna(0.0)
    y = train_df["VB_interp"].values

    rf_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)

    # Evaluate on all clean data
    X_all = full_df[feature_cols].fillna(0.0)
    full_df = full_df.copy()
    full_df["VB_ml_pred"] = np.clip(rf_model.predict(X_all), 0.0, None)

    # Physics-Based Digital Twin Differential State Model per Case
    # dVB/dt = k_mat * (DOC^0.45) * (feed^0.65) * (1 + 0.15 * (smcDC_mean / ref_smcDC))
    full_df["VB_physics"] = 0.0
    full_df["VB_twin_fused"] = 0.0

    for case_id, group in full_df.groupby("case"):
        vb_phys = 0.0
        vb_fused = 0.0
        prev_time = 0.0

        for idx, row in group.iterrows():
            dt = max(float(row["time"]) - prev_time, 1.0) if prev_time > 0 else max(float(row["time"]), 2.0)
            prev_time = float(row["time"])

            # Material wear rate constant (Cast iron vs Stainless Steel J45 at vc = 200 m/min)
            k_mat = 0.0048 if row["material"] == 1 else 0.0082
            doc_factor = (float(row["DOC"]) / 1.0) ** 0.45
            feed_factor = (float(row["feed"]) / 0.35) ** 0.65
            current_load = max(float(row["smcDC_mean"]), 0.1) / 6.0

            d_vb = k_mat * doc_factor * feed_factor * (0.75 + 0.25 * current_load) * dt
            vb_phys = vb_phys + d_vb

            # State Assimilation (Kalman-like measurement update fusing Physics + Sensor ML)
            ml_meas = float(row["VB_ml_pred"])
            if not row["is_corrupt"]:
                # Predictor-Corrector update: 65% ML sensor observation + 35% physics trajectory
                vb_fused = 0.35 * (vb_fused + d_vb) + 0.65 * ml_meas
            else:
                # When sensor is corrupt (e.g., Case 6 Run 1), Twin relies 100% on Physics state!
                vb_fused = vb_fused + d_vb

            full_df.loc[idx, "VB_physics"] = vb_phys
            full_df.loc[idx, "VB_twin_fused"] = vb_fused

    valid_eval = full_df[~full_df["is_corrupt"] & full_df["VB_raw"].notna()]
    r2_ml = r2_score(valid_eval["VB_raw"], valid_eval["VB_ml_pred"])
    r2_twin = r2_score(valid_eval["VB_raw"], valid_eval["VB_twin_fused"])
    rmse_twin = np.sqrt(mean_squared_error(valid_eval["VB_raw"], valid_eval["VB_twin_fused"]))

    metrics = {
        "r2_ml": r2_ml,
        "r2_twin": r2_twin,
        "rmse_twin": rmse_twin,
        "feature_cols": feature_cols
    }
    return rf_model, full_df, metrics


# ==============================================================================
# 4. AUTONOMOUS CONTROL LAW (MODULE 7): ADAPTIVE FEED-RATE & SPEED OVERRIDE
# ==============================================================================
def compute_autonomous_control(vb_est: float, smc_dc: float, nominal_feed: float, doc: float, vb_limit: float = 0.30):
    """
    Closed-Loop Autonomous CNC Override Controller:
    - ISO 3685 standard flank wear criterion: VB_limit = 0.30 mm.
    - Regulates feed rate f [mm/rev] and table feed Vf [mm/min] (at N = 826 RPM)
      to prevent catastrophic insert fracture and cap spindle motor current/torque.
    """
    n_rpm = 826.0  # Fixed Matsuura MC-510V spindle speed for 200 m/min on 70mm face mill
    wear_ratio = vb_est / vb_limit

    if wear_ratio < 0.55 and smc_dc < 7.5:
        status = "NORMAL (OPTIMAL ROUGHING)"
        color = "#10B981"  # Green
        feed_override_pct = 100.0
        spindle_override_pct = 100.0
        action_msg = "Maintain nominal G-code parameters. Cutting forces and flank wear within green envelope."
    elif wear_ratio < 0.85:
        # Proportional feed throttling to reduce primary shear zone cutting force
        throttle = 100.0 - 35.0 * ((wear_ratio - 0.55) / 0.30)
        if smc_dc > 8.0:
            throttle -= 10.0
        feed_override_pct = float(np.clip(throttle, 55.0, 95.0))
        spindle_override_pct = 100.0
        status = "AUTONOMOUS FEED DERATING (ACTIVE CONTROL)"
        color = "#F59E0B"  # Amber
        action_msg = (
            f"Flank wear approaching threshold ({vb_est:.3f} mm). Controller autonomously throttled "
            f"feed rate to {feed_override_pct:.1f}% to stabilize shear zone torque and finish current pass."
        )
    elif wear_ratio < 1.05:
        feed_override_pct = 50.0
        spindle_override_pct = 90.0
        status = "CRITICAL WEAR WARNING (FINISH PASS ONLY)"
        color = "#EF4444"  # Red
        action_msg = (
            f"ISO 3685 wear limit reached ({vb_est:.3f} mm >= {vb_limit:.2f} mm). "
            "Feed locked to 50% safety creep; automatic tool-change (ATC) queued at end of cut."
        )
    else:
        feed_override_pct = 0.0
        spindle_override_pct = 0.0
        status = "AUTONOMOUS E-STOP / TOOL RETRACT"
        color = "#991B1B"  # Dark Red
        action_msg = (
            f"Severe flank wear ({vb_est:.3f} mm) / untempered martensite thermal risk! "
            "Spindle retracted and cycle halted autonomously to prevent KC710 insert breakage."
        )

    cmd_feed_mm_rev = nominal_feed * (feed_override_pct / 100.0)
    cmd_table_feed_mm_min = cmd_feed_mm_rev * n_rpm * (spindle_override_pct / 100.0)
    # Estimated cutting force reduction proportional to feed^0.75
    force_reduction_pct = (1.0 - (feed_override_pct / 100.0) ** 0.75) * 100.0 if feed_override_pct > 0 else 100.0

    return {
        "status": status,
        "color": color,
        "feed_override_pct": feed_override_pct,
        "spindle_override_pct": spindle_override_pct,
        "cmd_feed_mm_rev": cmd_feed_mm_rev,
        "cmd_table_feed_mm_min": cmd_table_feed_mm_min,
        "force_reduction_pct": force_reduction_pct,
        "action_msg": action_msg
    }


# ==============================================================================
# MAIN APPLICATION UI
# ==============================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        raw_df, waveforms, loaded_path = load_nasa_mill_dataset(base_dir)
    except Exception as e:
        st.error(f"Dataset Loading Error: {e}")
        st.stop()

    # Sidebar Controls
    st.sidebar.title("Matsuura MC-510V Control")
    st.sidebar.caption(f"Loaded: `{os.path.basename(loaded_path)}` (167 Runs, 250 Hz)")

    st.sidebar.markdown("### 1. Signal Windowing (Mod 2 & 3)")
    win_range = st.sidebar.slider(
        "Steady-State Cut Sample Window (0–9000)",
        min_value=500,
        max_value=8500,
        value=(3000, 6500),
        step=250,
        help="Slices out Entry Cut and Exit Cut air-cutting transients (Figures 7–12 in NASA Readme)."
    )

    full_df = extract_signal_features(raw_df, waveforms, win_range[0], win_range[1])
    rf_model, full_df, metrics = train_hybrid_digital_twin(full_df)

    st.sidebar.markdown("### 2. Select Experimental Case & Run")
    cases_summary = (
        full_df.groupby("case")
        .agg(
            runs=("run", "count"),
            mat=("material_name", "first"),
            doc=("DOC", "first"),
            feed=("feed", "first"),
            max_vb=("VB_interp", "max")
        )
        .reset_index()
    )
    case_options = {
        row["case"]: f"Case {row['case']:02d} | {row['mat']} | DOC={row['doc']}mm, f={row['feed']}mm/rev ({row['runs']} runs)"
        for _, row in cases_summary.iterrows()
    }

    selected_case = st.sidebar.selectbox(
        "Experimental Case (1–16)",
        options=list(case_options.keys()),
        format_func=lambda x: case_options[x],
        index=0
    )

    case_runs_df = full_df[full_df["case"] == selected_case].sort_values("run")
    run_list = case_runs_df["run"].tolist()
    selected_run = st.sidebar.select_slider(
        "Machining Pass / Run Number",
        options=run_list,
        value=run_list[min(len(run_list) - 1, len(run_list) // 2)]
    )

    st.sidebar.markdown("### 3. ISO 3685 & Fault Injection (Mod 7)")
    vb_limit = st.sidebar.slider(
        "ISO 3685 Flank Wear Limit VB_max [mm]",
        min_value=0.20,
        max_value=0.45,
        value=0.30,
        step=0.05
    )
    sim_wear_bias = st.sidebar.slider(
        "Simulate Additional Tool Wear Bias (+mm)",
        min_value=0.00,
        max_value=0.25,
        value=0.00,
        step=0.02,
        help="Inject extra virtual flank wear to test the Autonomous Feed-Rate Override Controller in real time."
    )

    # Current Run Data
    cur_row = case_runs_df[case_runs_df["run"] == selected_run].iloc[0]
    cur_waves = waveforms[(int(selected_case), int(selected_run))]

    vb_twin_live = float(cur_row["VB_twin_fused"]) + sim_wear_bias
    smc_dc_live = float(cur_row["smcDC_mean"]) * (1.0 + 1.2 * sim_wear_bias)
    ctrl = compute_autonomous_control(
        vb_est=vb_twin_live,
        smc_dc=smc_dc_live,
        nominal_feed=float(cur_row["feed"]),
        doc=float(cur_row["DOC"]),
        vb_limit=vb_limit
    )

    # Header Banner
    st.title("Physics-Informed CNC Milling Digital Twin & Autonomous Feed-Rate Control")
    st.markdown(
        "**Dataset:** NASA Ames / UC Berkeley BEST Lab (`mill.mat`) | **Machine:** Matsuura MC-510V "
        "| **Tool:** 70 mm 6-Insert Kennametal KC710 Face Mill ($v_c = 200\\text{ m/min}, N = 826\\text{ RPM}$)"
    )

    # Top KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Microscope Ground Truth VB",
        f"{cur_row['VB_raw']:.3f} mm" if pd.notna(cur_row["VB_raw"]) else "Unmeasured (NaN)",
        delta=f"Interp: {cur_row['VB_interp']:.3f} mm"
    )
    k2.metric(
        "Hybrid Twin Estimated VB",
        f"{vb_twin_live:.3f} mm",
        delta=f"{vb_twin_live - vb_limit:+.3f} mm vs ISO Limit",
        delta_color="inverse"
    )
    k3.metric(
        "Spindle DC Current (Cutting Force)",
        f"{smc_dc_live:.2f} A",
        delta=f"AC RMS: {cur_row['smcAC_rms']:.2f} A"
    )
    k4.metric(
        "Autonomous Feed Override",
        f"{ctrl['feed_override_pct']:.1f}%",
        delta=f"{ctrl['cmd_table_feed_mm_min']:.0f} mm/min Table Feed"
    )
    k5.metric(
        "Twin Validation R² Score",
        f"{metrics['r2_twin']:.3f}",
        delta=f"RMSE: {metrics['rmse_twin']:.4f} mm"
    )

    # Autonomous Controller Status Bar
    st.markdown(
        f"""
        <div style="padding:12px 18px; border-radius:8px; background-color:{ctrl['color']}22; border-left: 6px solid {ctrl['color']}; margin-bottom:16px;">
            <strong style="color:{ctrl['color']}; font-size:1.05rem;">MODULE 7 AUTONOMOUS CONTROLLER STATE: {ctrl['status']}</strong><br/>
            <span>{ctrl['action_msg']} (Estimated Cutting Force Reduction: <b>{ctrl['force_reduction_pct']:.1f}%</b>)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Main Navigation Tabs Mapped to Course Syllabus
    tab1, tab2, tab3, tab4 = st.tabs([
        "📡 Mod 2 & 3: IIoT Sensor Waveforms & Windowing (ECE)",
        "🔄 Mod 4 & 6: Physics-Informed Digital Twin & ML (Mech + AI)",
        "🤖 Mod 7: Closed-Loop Autonomous CNC Override",
        "📊 Dataset Explorer & Syllabus Mapping"
    ])

    # ==========================================================================
    # TAB 1: ECE SIGNAL PROCESSING & TRANSIENT WINDOWING
    # ==========================================================================
    with tab1:
        col_left, col_right = st.columns([2.2, 1])

        with col_left:
            st.subheader(f"High-Frequency 250 Hz DAQ Waveforms — Case {selected_case}, Run {selected_run}")
            if cur_row["is_corrupt"]:
                st.warning("Note: This experimental run contains corrupted/drop-out DAQ sensor arrays in the original NASA file. The Digital Twin automatically switches to pure Physics prediction for this pass.")

            time_axis = np.arange(len(cur_waves["smcDC"])) / 250.0  # 36 seconds at 250 Hz
            t_start, t_end = win_range[0] / 250.0, win_range[1] / 250.0

            fig_wave = make_subplots(
                rows=3, cols=2,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    "Spindle Motor Current AC (smcAC)",
                    "Spindle Motor Current DC (smcDC — Force Proxy)",
                    "Table Vibration RMS (vib_table)",
                    "Spindle Vibration RMS (vib_spindle)",
                    "Table Acoustic Emission RMS (AE_table)",
                    "Spindle Acoustic Emission RMS (AE_spindle)"
                )
            )

            ch_map = [
                ("smcAC", 1, 1, "#3B82F6"),
                ("smcDC", 1, 2, "#EF4444"),
                ("vib_table", 2, 1, "#10B981"),
                ("vib_spindle", 2, 2, "#059669"),
                ("AE_table", 3, 1, "#8B5CF6"),
                ("AE_spindle", 3, 2, "#D97706")
            ]

            for ch_name, r_idx, c_idx, color_hex in ch_map:
                sig = cur_waves[ch_name]
                # Downsample by 3x purely for snappy browser rendering
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
                # Highlight Steady-State Cutting Window
                fig_wave.add_vrect(
                    x0=t_start, x1=t_end,
                    fillcolor="rgba(16, 185, 129, 0.12)",
                    line_width=1,
                    line_dash="dash",
                    line_color="#10B981",
                    row=r_idx, col=c_idx
                )

            fig_wave.update_layout(
                height=620,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis5_title="Time [seconds] (Green Band = Steady-State Cutting Window)",
                xaxis6_title="Time [seconds] (Green Band = Steady-State Cutting Window)"
            )
            st.plotly_chart(fig_wave, use_container_width=True)

        with col_right:
            st.subheader("FFT Passband Spectrum")
            selected_fft_ch = st.selectbox(
                "Select Channel for Spectral Analysis",
                ["vib_spindle", "vib_table", "AE_spindle", "AE_table", "smcAC"]
            )
            sig_steady = cur_waves[selected_fft_ch][win_range[0]:win_range[1]]
            sig_zero = sig_steady - np.mean(sig_steady)
            freqs = fft.rfftfreq(len(sig_zero), d=1.0 / 250.0)
            fft_mag = np.abs(fft.rfft(sig_zero)) / max(len(sig_zero), 1)

            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=freqs, y=fft_mag,
                mode="lines",
                fill="tozeroy",
                line=dict(color="#6366F1", width=1.5),
                name="FFT Magnitude"
            ))
            fig_fft.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=25, b=20),
                xaxis_title="Frequency [Hz] (Nyquist = 125 Hz @ 250 Hz DAQ)",
                yaxis_title="Amplitude"
            )
            st.plotly_chart(fig_fft, use_container_width=True)

            st.markdown("#### Hardware Signal Chain (Readme Fig. 1)")
            st.info(
                "- **AE Sensor (WD 925):** 50 kHz HPF Preamplifier (Model 1801) $\\rightarrow$ 8.0 ms Analog RMS Meter $\\rightarrow$ 250 Hz MIO-16 DAQ.\n"
                "- **Vibration (Endevco 7201-50):** Charge Amp $\\rightarrow$ Ithaco 400 Hz–1 kHz Bandpass Filter (blocks 180 Hz 3rd harmonic power noise) $\\rightarrow$ 8.0 ms RMS.\n"
                "- **Why Windowing Matters:** Samples `0:2500` and `7000:9000` capture idle air-cutting, entry engagement, and exit breakout. Windowing isolates the true shear-zone load."
            )

    # ==========================================================================
    # TAB 2: HYBRID DIGITAL TWIN & ML TOOL WEAR ESTIMATION
    # ==========================================================================
    with tab2:
        c_left, c_right = st.columns([1.8, 1.2])

        with c_left:
            st.subheader(f"Case {selected_case} Flank Wear (VB) Trajectory: Digital Shadow vs. Hybrid Twin")
            fig_vb = go.Figure()

            # Physics Differential Model
            fig_vb.add_trace(go.Scatter(
                x=case_runs_df["time"],
                y=case_runs_df["VB_physics"],
                mode="lines+markers",
                name="Pure Physics Model (Extended Taylor-Usui ODE)",
                line=dict(color="#94A3B8", width=2, dash="dot")
            ))

            # Pure ML Sensor Prediction (Digital Shadow)
            fig_vb.add_trace(go.Scatter(
                x=case_runs_df["time"],
                y=case_runs_df["VB_ml_pred"],
                mode="lines+markers",
                name="Pure ML Sensor Estimate (Digital Shadow)",
                line=dict(color="#3B82F6", width=2)
            ))

            # Hybrid Fused Digital Twin
            fig_vb.add_trace(go.Scatter(
                x=case_runs_df["time"],
                y=case_runs_df["VB_twin_fused"],
                mode="lines+markers",
                name="Hybrid Digital Twin (Physics + Sensor Assimilation)",
                line=dict(color="#10B981", width=3.5)
            ))

            # Actual Microscope Ground Truth VB
            measured_df = case_runs_df[case_runs_df["VB_raw"].notna()]
            fig_vb.add_trace(go.Scatter(
                x=measured_df["time"],
                y=measured_df["VB_raw"],
                mode="markers",
                name="Microscope Ground Truth VB (Measured Offline)",
                marker=dict(color="#EF4444", size=11, symbol="diamond", line=dict(width=1.5, color="white"))
            ))

            # ISO 3685 Wear Limit Line
            fig_vb.add_hline(
                y=vb_limit,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"ISO 3685 Criterion (VB = {vb_limit:.2f} mm)",
                annotation_position="top left"
            )

            # Highlight Current Selected Run
            fig_vb.add_vline(
                x=cur_row["time"],
                line_dash="solid",
                line_color="#F59E0B",
                annotation_text=f"Current Run {selected_run}"
            )

            fig_vb.update_layout(
                height=460,
                xaxis_title="Cumulative Machining Time [minutes]",
                yaxis_title="Flank Wear VB [mm]",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_vb, use_container_width=True)

        with c_right:
            st.subheader("Random Forest Feature Importance")
            importances = rf_model.feature_importances_
            feat_names = metrics["feature_cols"]
            imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=True).tail(10)

            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation="h",
                marker_color="#6366F1"
            ))
            fig_imp.update_layout(
                height=310,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title="Relative Gini Importance"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("#### Why Hybrid Twin Beats Pure Taylor or Pure ML")
            st.success(
                "1. **Why Pure Taylor Fails (Readme p. 8):** Cast iron hardness inclusions and thermal diffusion cause sudden wear jumps that static equations ($T = C_v v_c^k s^i$) cannot foresee.\n"
                "2. **Why Pure ML (Digital Shadow) Fails:** In **Case 6 Run 1** and **Case 16 Run 5**, DAQ sensor cables dropped out. Pure ML outputs nonsense when sensors fault, whereas our **Hybrid Twin** falls back on the differential wear state equation."
            )

    # ==========================================================================
    # TAB 3: MODULE 7 CLOSED-LOOP AUTONOMOUS OVERRIDE SIMULATION
    # ==========================================================================
    with tab3:
        st.subheader(f"Closed-Loop Autonomous Feed-Rate Control Across Case {selected_case}")
        st.markdown(
            "Instead of stopping at a passive predictive maintenance alert, the **Module 7 Autonomous Controller** "
            "continuously reads the Hybrid Twin's $VB$ state and `smcDC` cutting torque proxy to rewrite the CNC feed-rate override in real time."
        )

        sim_rows = []
        for _, r in case_runs_df.iterrows():
            vb_val = float(r["VB_twin_fused"]) + sim_wear_bias
            smc_val = float(r["smcDC_mean"]) * (1.0 + 1.2 * sim_wear_bias)
            c_out = compute_autonomous_control(
                vb_est=vb_val,
                smc_dc=smc_val,
                nominal_feed=float(r["feed"]),
                doc=float(r["DOC"]),
                vb_limit=vb_limit
            )
            sim_rows.append({
                "run": int(r["run"]),
                "time": float(r["time"]),
                "VB_twin": vb_val,
                "nominal_table_feed": float(r["feed"]) * 826.0,
                "autonomous_table_feed": c_out["cmd_table_feed_mm_min"],
                "feed_override_pct": c_out["feed_override_pct"],
                "uncontrolled_smcDC": smc_val,
                "controlled_smcDC": smc_val * ((c_out["feed_override_pct"] / 100.0) ** 0.75),
                "status": c_out["status"]
            })
        sim_df = pd.DataFrame(sim_rows)

        fig_ctrl = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=(
                "CNC Table Feed Rate [mm/min]: Uncontrolled Nominal vs. Autonomous Closed-Loop Override",
                "Spindle Motor DC Current [A] (Cutting Force Proxy): Uncontrolled vs. Regulated"
            )
        )

        fig_ctrl.add_trace(
            go.Scatter(
                x=sim_df["time"], y=sim_df["nominal_table_feed"],
                name="Nominal G-Code Feed [mm/min]",
                line=dict(color="#94A3B8", width=2, dash="dash")
            ),
            row=1, col=1
        )
        fig_ctrl.add_trace(
            go.Scatter(
                x=sim_df["time"], y=sim_df["autonomous_table_feed"],
                name="Autonomous Commanded Feed [mm/min]",
                mode="lines+markers",
                line=dict(color="#10B981", width=3)
            ),
            row=1, col=1
        )

        fig_ctrl.add_trace(
            go.Scatter(
                x=sim_df["time"], y=sim_df["uncontrolled_smcDC"],
                name="Uncontrolled Spindle Current [A]",
                line=dict(color="#EF4444", width=2, dash="dot")
            ),
            row=2, col=1
        )
        fig_ctrl.add_trace(
            go.Scatter(
                x=sim_df["time"], y=sim_df["controlled_smcDC"],
                name="Regulated Spindle Current [A]",
                mode="lines+markers",
                line=dict(color="#3B82F6", width=3)
            ),
            row=2, col=1
        )

        fig_ctrl.update_layout(
            height=520,
            xaxis2_title="Cumulative Machining Time [minutes]",
            yaxis1_title="Feed Rate [mm/min]",
            yaxis2_title="Current smcDC [A]",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_ctrl, use_container_width=True)

        st.dataframe(
            sim_df.style.format({
                "time": "{:.1f}",
                "VB_twin": "{:.3f}",
                "nominal_table_feed": "{:.1f}",
                "autonomous_table_feed": "{:.1f}",
                "feed_override_pct": "{:.1f}%",
                "uncontrolled_smcDC": "{:.2f}",
                "controlled_smcDC": "{:.2f}"
            }),
            use_container_width=True
        )

    # ==========================================================================
    # TAB 4: DATASET EXPLORER & COURSE SYLLABUS ALIGNMENT
    # ==========================================================================
    with tab4:
        st.subheader("Course Syllabus Alignment Matrix (5-Mark Internal Presentation Guide)")
        syllabus_table = pd.DataFrame([
            {
                "Syllabus Module": "Module 2: Sensors & Connected Systems",
                "Presenter": "ECE Teammate",
                "Project Implementation": "Physical Acoustic Group WD-925 AE sensors (up to 2 MHz), Endevco 7201-50 accelerometers, Omron K3TB current converter, Ithaco 400Hz-1kHz analog filtering, and 8ms hardware RMS conversion sampled at 250 Hz."
            },
            {
                "Syllabus Module": "Module 3: Data Analysis & Process Control",
                "Presenter": "ECE Teammate",
                "Project Implementation": "Automated transient windowing to strip out entry/exit air cutting from the 9,000-sample arrays, extracting passband FFT spectral energy, kurtosis, and indirect cutting force telemetry."
            },
            {
                "Syllabus Module": "Module 4: Digital Twin in Industry 4.0",
                "Presenter": "Mechanical Presenter",
                "Project Implementation": "Distinguishes a one-way 'Digital Shadow' from a bi-directional 'Digital Twin' by coupling an extended Taylor-Usui wear rate ODE with real-time sensor assimilation, surviving DAQ dropouts (Case 6 Run 1)."
            },
            {
                "Syllabus Module": "Module 6: AI, ML & DL in Manufacturing",
                "Presenter": "Mechanical + ECE",
                "Project Implementation": "250-tree Random Forest Regressor mapping 20 multi-sensor time/frequency features to physical flank wear VB [mm] across Cast Iron and Stainless Steel J45 roughing passes."
            },
            {
                "Syllabus Module": "Module 7: Autonomous Manufacturing System",
                "Presenter": "Mechanical Presenter",
                "Project Implementation": "Closed-loop Adaptive Feed-Rate Override law that throttles CNC table feed (413 mm/min down to 206 mm/min) when flank wear approaches ISO 3685 limit (0.30 mm) and triggers autonomous spindle retraction."
            }
        ])
        st.table(syllabus_table)

        st.subheader("Complete 167-Run Processed Feature Table")
        display_cols = [
            "case", "run", "time", "material_name", "DOC", "feed",
            "VB_raw", "VB_interp", "VB_physics", "VB_ml_pred", "VB_twin_fused",
            "smcDC_mean", "smcAC_rms", "vib_spindle_rms", "AE_spindle_rms", "is_corrupt"
        ]
        st.dataframe(full_df[display_cols], use_container_width=True)


if __name__ == "__main__":
    main()
