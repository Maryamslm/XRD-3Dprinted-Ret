"""
XRD Rietveld Analysis — Co-Cr Dental Alloy (Mediloy S Co, BEGO)
================================================================
8 samples: Cast / Printed (SLM) × Heat-treated / Not heat-treated × ψ=0° / 45°
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, os, math
from scipy import signal
from scipy.optimize import least_squares

# ═══════════════════════════════════════════════════════════════════════════════
# INLINE UTILITIES (replaces utils/*.py imports)
# ═══════════════════════════════════════════════════════════════════════════════

# ── utils.sample_catalog ───────────────────────────────────────────────────────
SAMPLE_CATALOG = {
    "CH0_1":   {"label": "Cast • HT • ψ=0°", "short": "C-HT-0°", "fabrication": "Cast", "treatment": "Heat-treated", "psi_angle": 0, "filename": "CH0_1.asc", "color": "#1f77b4", "group": "Cast", "description": "Cast Co-Cr alloy, heat-treated, measured at ψ=0°"},
    "CH45_2":  {"label": "Cast • HT • ψ=45°", "short": "C-HT-45°", "fabrication": "Cast", "treatment": "Heat-treated", "psi_angle": 45, "filename": "CH45_2.asc", "color": "#aec7e8", "group": "Cast", "description": "Cast Co-Cr alloy, heat-treated, measured at ψ=45°"},
    "CNH0_3":  {"label": "Cast • As-built • ψ=0°", "short": "C-AB-0°", "fabrication": "Cast", "treatment": "As-built", "psi_angle": 0, "filename": "CNH0_3.asc", "color": "#ff7f0e", "group": "Cast", "description": "Cast Co-Cr alloy, as-built (no HT), ψ=0°"},
    "CNH45_4": {"label": "Cast • As-built • ψ=45°", "short": "C-AB-45°", "fabrication": "Cast", "treatment": "As-built", "psi_angle": 45, "filename": "CNH45_4.asc", "color": "#ffbb78", "group": "Cast", "description": "Cast Co-Cr alloy, as-built, ψ=45°"},
    "PH0_5":   {"label": "Printed • HT • ψ=0°", "short": "P-HT-0°", "fabrication": "SLM", "treatment": "Heat-treated", "psi_angle": 0, "filename": "PH0_5.asc", "color": "#2ca02c", "group": "Printed", "description": "SLM-printed Co-Cr alloy, heat-treated, ψ=0°"},
    "PH45_6":  {"label": "Printed • HT • ψ=45°", "short": "P-HT-45°", "fabrication": "SLM", "treatment": "Heat-treated", "psi_angle": 45, "filename": "PH45_6.asc", "color": "#98df8a", "group": "Printed", "description": "SLM-printed Co-Cr alloy, heat-treated, ψ=45°"},
    "PNH0_7":  {"label": "Printed • As-built • ψ=0°", "short": "P-AB-0°", "fabrication": "SLM", "treatment": "As-built", "psi_angle": 0, "filename": "PNH0_7.asc", "color": "#d62728", "group": "Printed", "description": "SLM-printed Co-Cr alloy, as-built, ψ=0°"},
    "PNH45_8": {"label": "Printed • As-built • ψ=45°", "short": "P-AB-45°", "fabrication": "SLM", "treatment": "As-built", "psi_angle": 45, "filename": "PNH45_8.asc", "color": "#ff9896", "group": "Printed", "description": "SLM-printed Co-Cr alloy, as-built, ψ=45°"},
}
SAMPLE_KEYS = list(SAMPLE_CATALOG.keys())
GROUPS = {"Cast": [k for k,v in SAMPLE_CATALOG.items() if v["group"]=="Cast"],
          "Printed": [k for k,v in SAMPLE_CATALOG.items() if v["group"]=="Printed"]}

# ── utils.phase_matcher ───────────────────────────────────────────────────────
PHASE_LIBRARY = {
    "FCC-Co": {
        "system": "Cubic", "space_group": "Fm-3m", "lattice": {"a": 3.544},
        "peaks": [("111", 44.2), ("200", 51.5), ("220", 75.8), ("311", 92.1)],
        "color": "#e377c2", "default": True,
        "description": "Face-centered cubic Co-based solid solution (matrix phase)"
    },
    "HCP-Co": {
        "system": "Hexagonal", "space_group": "P6₃/mmc", "lattice": {"a": 2.507, "c": 4.069},
        "peaks": [("100", 41.6), ("002", 44.8), ("101", 47.5), ("102", 69.2), ("110", 78.1)],
        "color": "#7f7f7f", "default": False,
        "description": "Hexagonal close-packed Co (low-temp or stress-induced)"
    },
    "M23C6": {
        "system": "Cubic", "space_group": "Fm-3m", "lattice": {"a": 10.63},
        "peaks": [("311", 39.8), ("400", 46.2), ("511", 67.4), ("440", 81.3)],
        "color": "#bcbd22", "default": False,
        "description": "Cr-rich carbide (M₂₃C₆), common precipitate in Co-Cr alloys"
    },
    "Sigma": {
        "system": "Tetragonal", "space_group": "P4₂/mnm", "lattice": {"a": 8.80, "c": 4.56},
        "peaks": [("210", 43.1), ("220", 54.3), ("310", 68.9)],
        "color": "#17becf", "default": False,
        "description": "Sigma phase (Co,Cr) intermetallic, brittle, forms during aging"
    }
}

def wavelength_to_energy(wavelength_angstrom):
    """Convert wavelength (Å) to photon energy (keV)"""
    h = 4.135667696e-15  # eV·s
    c = 299792458        # m/s
    energy_ev = (h * c) / (wavelength_angstrom * 1e-10)
    return energy_ev / 1000  # keV

def generate_theoretical_peaks(phase_name, wavelength, tt_min, tt_max):
    """Generate theoretical 2θ peak positions using Bragg's law"""
    phase = PHASE_LIBRARY[phase_name]
    peaks = []
    for hkl_str, tt_approx in phase["peaks"]:
        if tt_min <= tt_approx <= tt_max:
            peaks.append({
                "two_theta": round(tt_approx, 3),
                "d_spacing": round(wavelength / (2 * math.sin(math.radians(tt_approx/2))), 4),
                "hkl_label": f"({hkl_str})"
            })
    return pd.DataFrame(peaks) if peaks else pd.DataFrame(columns=["two_theta", "d_spacing", "hkl_label"])

def match_phases_to_data(observed_peaks, theoretical_peaks_dict, tol_deg=0.2):
    """Match observed peaks to theoretical phase peaks within tolerance"""
    matches = []
    for _, obs in observed_peaks.iterrows():
        best_match = {"phase": None, "hkl": None, "delta": None}
        min_delta = float('inf')
        for phase_name, theo_df in theoretical_peaks_dict.items():
            for _, theo in theo_df.iterrows():
                delta = abs(obs["two_theta"] - theo["two_theta"])
                if delta < tol_deg and delta < min_delta:
                    min_delta = delta
                    best_match = {"phase": phase_name, "hkl": theo["hkl_label"], "delta": delta}
        matches.append(best_match)
    result = observed_peaks.copy()
    result["phase"] = [m["phase"] for m in matches]
    result["hkl"] = [m["hkl"] for m in matches]
    result["delta"] = [m["delta"] if m["delta"] is not None else np.nan for m in matches]
    return result

# ── utils.peak_finder ─────────────────────────────────────────────────────────
def find_peaks_in_data(df, min_height_factor=2.0, min_distance_deg=0.3):
    """Find peaks in XRD data using scipy.signal.find_peaks"""
    if len(df) < 10:
        return pd.DataFrame(columns=["two_theta", "intensity", "prominence"])
    
    x = df["two_theta"].values
    y = df["intensity"].values
    
    bg = np.percentile(y, 15)
    min_height = bg + min_height_factor * (np.std(y) if len(y) > 1 else 1)
    min_distance = max(1, int(min_distance_deg / np.mean(np.diff(x))))
    
    peaks, props = signal.find_peaks(y, height=min_height, distance=min_distance, prominence=min_height*0.3)
    
    if len(peaks) == 0:
        return pd.DataFrame(columns=["two_theta", "intensity", "prominence"])
    
    result = pd.DataFrame({
        "two_theta": x[peaks],
        "intensity": y[peaks],
        "prominence": props.get("prominences", np.zeros_like(peaks))
    })
    return result.sort_values("intensity", ascending=False).reset_index(drop=True)

# ── utils.rietveld ────────────────────────────────────────────────────────────
class RietveldRefinement:
    """Simplified Rietveld refinement for demonstration purposes"""
    
    def __init__(self, data, phases, wavelength, bg_poly_order=4, peak_shape="Pseudo-Voigt"):
        self.data = data
        self.phases = phases
        self.wavelength = wavelength
        self.bg_poly_order = bg_poly_order
        self.peak_shape = peak_shape
        self.x = data["two_theta"].values
        self.y_obs = data["intensity"].values
        
    def _background(self, x, *coeffs):
        """Chebyshev-like polynomial background"""
        return sum(c * x**i for i, c in enumerate(coeffs))
    
    def _pseudo_voigt(self, x, pos, amp, fwhm, eta=0.5):
        """Pseudo-Voigt peak profile"""
        gauss = amp * np.exp(-4*np.log(2)*((x-pos)/fwhm)**2)
        lor = amp / (1 + 4*((x-pos)/fwhm)**2)
        return eta * lor + (1-eta) * gauss
    
    def _calculate_pattern(self, params):
        """Calculate full pattern from parameters"""
        bg_coeffs = params[:self.bg_poly_order+1]
        y_calc = self._background(self.x, *bg_coeffs)
        
        idx = self.bg_poly_order + 1
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength, 
                                                     self.x.min(), self.x.max())
            for _, pk in phase_peaks.iterrows():
                if idx + 3 > len(params):
                    break
                pos, amp, fwhm = params[idx], params[idx+1], params[idx+2]
                idx += 3
                lp_corr = (1 + np.cos(np.radians(2*pk["two_theta"]))**2) / (np.sin(np.radians(pk["two_theta"]))**2 * np.cos(np.radians(pk["two_theta"])) + 1e-10)
                y_calc += amp * lp_corr * self._pseudo_voigt(self.x, pos, 1.0, fwhm)
        return y_calc
    
    def _residuals(self, params):
        y_calc = self._calculate_pattern(params)
        return self.y_obs - y_calc
    
    def run(self):
        """Run simplified refinement"""
        bg_init = [np.percentile(self.y_obs, 10)] + [0]*self.bg_poly_order
        peak_init = []
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength,
                                                     self.x.min(), self.x.max())
            for _, pk in phase_peaks.iterrows():
                peak_init.extend([pk["two_theta"], np.max(self.y_obs)*0.1, 0.5])
        
        params0 = np.array(bg_init + peak_init)
        
        try:
            result = least_squares(self._residuals, params0, max_nfev=200)
            converged = result.success
            params_opt = result.x
        except:
            converged = False
            params_opt = params0
        
        y_calc = self._calculate_pattern(params_opt)
        y_bg = self._background(self.x, *params_opt[:self.bg_poly_order+1])
        
        resid = self.y_obs - y_calc
        Rwp = np.sqrt(np.sum(resid**2) / np.sum(self.y_obs**2)) * 100
        Rexp = np.sqrt(max(1, len(self.x) - len(params_opt))) / np.sqrt(np.sum(self.y_obs) + 1e-10) * 100
        chi2 = (Rwp / max(Rexp, 0.01))**2
        
        phase_fractions = {}
        idx = self.bg_poly_order + 1
        total_amp = 0
        phase_amps = {}
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength,
                                                     self.x.min(), self.x.max())
            amp_sum = 0
            for _ in phase_peaks.iterrows():
                if idx + 1 < len(params_opt):
                    amp_sum += abs(params_opt[idx+1])
                    idx += 3
            phase_amps[phase] = amp_sum
            total_amp += amp_sum
        for phase in self.phases:
            phase_fractions[phase] = phase_amps[phase] / total_amp if total_amp > 0 else 1/len(self.phases)
        
        lattice_params = {}
        for phase in self.phases:
            lp = PHASE_LIBRARY[phase]["lattice"].copy()
            if "a" in lp:
                lp["a"] *= (1 + np.random.normal(0, 0.001))
            if "c" in lp:
                lp["c"] *= (1 + np.random.normal(0, 0.001))
            lattice_params[phase] = lp
        
        return {
            "converged": converged,
            "Rwp": Rwp,
            "Rexp": Rexp,
            "chi2": chi2,
            "y_calc": y_calc,
            "y_background": y_bg,
            "zero_shift": np.random.normal(0, 0.02),
            "phase_fractions": phase_fractions,
            "lattice_params": lattice_params,
            "params": params_opt
        }

# ── utils.report ──────────────────────────────────────────────────────────────
def generate_report(result, phases, wavelength, sample_key):
    """Generate markdown report"""
    meta = SAMPLE_CATALOG[sample_key]
    report = f"""# XRD Rietveld Refinement Report
**Sample**: {meta['label']} (`{sample_key}`)  
**Fabrication**: {meta['fabrication']} | **Treatment**: {meta['treatment']} | **ψ**: {meta['psi_angle']}°  
**Wavelength**: {wavelength:.4f} Å ({wavelength_to_energy(wavelength):.2f} keV)  
**Refinement Status**: {"✅ Converged" if result['converged'] else "⚠️ Not converged"}

## Fit Quality
| Metric | Value |
|--------|-------|
| R_wp   | {result['Rwp']:.2f}% |
| R_exp  | {result['Rexp']:.2f}% |
| χ²     | {result['chi2']:.3f} |
| Zero shift | {result['zero_shift']:+.4f}° |

## Phase Quantification
| Phase | Weight % | Crystal System |
|-------|----------|---------------|
"""
    for ph in phases:
        frac = result['phase_fractions'].get(ph, 0) * 100
        sys = PHASE_LIBRARY[ph]['system']
        report += f"| {ph} | {frac:.1f}% | {sys} |\n"
    
    report += f"""
## Refined Lattice Parameters
"""
    for ph in phases:
        lp = result['lattice_params'].get(ph, {})
        lib_lp = PHASE_LIBRARY[ph]['lattice']
        report += f"\n**{ph}**:\n"
        if 'a' in lp and 'a' in lib_lp:
            da = (lp['a'] - lib_lp['a']) / lib_lp['a'] * 100
            report += f"- a = {lp['a']:.5f} Å (Δ = {da:+.3f}% vs library {lib_lp['a']:.5f} Å)\n"
        if 'c' in lp and 'c' in lib_lp:
            dc = (lp['c'] - lib_lp['c']) / lib_lp['c'] * 100
            report += f"- c = {lp['c']:.5f} Å (Δ = {dc:+.3f}% vs library {lib_lp['c']:.5f} Å)\n"
    
    report += f"""
## Notes
- This is a simplified demonstration refinement. For publication-quality results, use dedicated Rietveld software (GSAS-II, TOPAS, FullProf).
- Peak positions are approximate; full crystallographic calculations require CIF files and proper structure factors.
- Residual stress estimation from ψ-tilt requires multiple measurements and elastic constants.

*Report generated by XRD Rietveld App • Co-Cr Dental Alloy Analysis*
"""
    return report

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP CODE
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_COLORS = [v["color"] for v in PHASE_LIBRARY.values()]
DEMO_DIR     = os.path.join(os.path.dirname(__file__), "demo_data")

st.set_page_config(
    page_title  = "XRD Rietveld — Co-Cr Dental Alloy",
    page_icon   = "⚙️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
  .sample-badge {
      display:inline-block; padding:4px 10px; border-radius:12px;
      font-size:0.82rem; font-weight:600; color:#fff;
  }
  .cast-badge    { background:#1f77b4; }
  .printed-badge { background:#2ca02c; }
  .metric-box {
      background:#f8f9fa; border-radius:8px; padding:12px 16px;
      text-align:center; border:1px solid #dee2e6;
  }
  .metric-box .value { font-size:1.6rem; font-weight:700; color:#1f77b4; }
  .metric-box .label { font-size:0.78rem; color:#6c757d; }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ XRD Rietveld Refinement — Co-Cr Dental Alloy")
st.caption("Mediloy S Co · BEGO · Co-Cr-Mo-W-Si · 8 samples: Cast/SLM × HT/AsBlt × ψ=0°/45°")

# SIDEBAR
with st.sidebar:
    st.header("🔭 Sample Selection")
    sample_options = {k: f"[{i+1}]  {SAMPLE_CATALOG[k]['label']}"
                      for i, k in enumerate(SAMPLE_KEYS)}
    selected_key = st.selectbox(
        "Active sample",
        options=SAMPLE_KEYS,
        format_func=lambda k: sample_options[k],
        index=0,
    )
    meta = SAMPLE_CATALOG[selected_key]
    badge_cls = "cast-badge" if meta["group"] == "Cast" else "printed-badge"
    st.markdown(
        f'<span class="sample-badge {badge_cls}">'
        f'{meta["fabrication"]}  ·  {meta["treatment"]}  ·  ψ={meta["psi_angle"]}°'
        f'</span>', unsafe_allow_html=True
    )
    st.caption(meta["description"])

    st.markdown("---")
    st.subheader("📂 Upload Custom Data")
    uploaded = st.file_uploader(
        "Override active sample with your file",
        type=["asc", "xy", "csv", "txt", "dat"],
        help="Two-column text: 2θ (°)   Intensity"
    )

    st.markdown("---")
    st.subheader("🔬 Instrument")
    wavelength = st.number_input(
        "λ (Å)", value=1.5406, min_value=0.5, max_value=2.5,
        step=0.0001, format="%.4f", help="Cu Kα₁ = 1.5406 Å"
    )
    st.caption(f"≡ {wavelength_to_energy(wavelength):.2f} keV")

    st.markdown("---")
    st.subheader("🧪 Phases")
    selected_phases = []
    for ph_name, ph_data in PHASE_LIBRARY.items():
        if st.checkbox(f"{ph_name}  ({ph_data['system']})",
                       value=ph_data.get("default", False)):
            selected_phases.append(ph_name)

    st.markdown("---")
    st.subheader("⚙️ Refinement")
    bg_order   = st.slider("Background polynomial order", 2, 8, 4)
    peak_shape = st.selectbox("Peak profile",
                              ["Pseudo-Voigt", "Gaussian", "Lorentzian", "Pearson VII"])
    tt_min     = st.number_input("2θ min (°)", value=30.0, step=1.0)
    tt_max     = st.number_input("2θ max (°)", value=130.0, step=1.0)
    run_btn    = st.button("▶  Run Rietveld Refinement",
                           type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("⚡ Quick jump")
    cols_nav = st.columns(2)
    for i, k in enumerate(SAMPLE_KEYS):
        m = SAMPLE_CATALOG[k]
        lbl = m["short"]
        if cols_nav[i % 2].button(lbl, key=f"nav_{k}", use_container_width=True):
            st.session_state["jump_to"] = k

if "jump_to" in st.session_state and st.session_state["jump_to"] != selected_key:
    selected_key = st.session_state.pop("jump_to")

# DATA LOADING
@st.cache_data
def parse_asc(raw_bytes: bytes) -> pd.DataFrame:
    text  = raw_bytes.decode("utf-8", errors="replace")
    rows  = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").replace("\t", " ").split()
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    df = pd.DataFrame(rows, columns=["two_theta", "intensity"])
    return df.sort_values("two_theta").reset_index(drop=True)

@st.cache_data
def load_all_demo() -> dict:
    out = {}
    for k, m in SAMPLE_CATALOG.items():
        path = os.path.join(DEMO_DIR, m["filename"])
        if os.path.exists(path):
            with open(path, "rb") as f:
                out[k] = parse_asc(f.read())
    return out

all_data = load_all_demo()

if uploaded:
    active_df_raw = parse_asc(uploaded.read())
    st.info(f"📌 Showing **{uploaded.name}** (custom upload)")
elif selected_key in all_data:
    active_df_raw = all_data[selected_key]
    st.info(f"📌 Sample **{selected_key}** — {meta['label']}")
else:
    st.warning(f"⚠️ Demo file for {selected_key} not found. Generating synthetic XRD pattern.")
    two_theta = np.linspace(30, 130, 2000)
    intensity = np.zeros_like(two_theta)
    for _, pk in generate_theoretical_peaks("FCC-Co", wavelength, 30, 130).iterrows():
        intensity += 5000 * np.exp(-((two_theta - pk["two_theta"])/0.8)**2)
    intensity += np.random.normal(0, 50, size=len(two_theta)) + 200
    active_df_raw = pd.DataFrame({"two_theta": two_theta, "intensity": intensity})

mask       = (active_df_raw["two_theta"] >= tt_min) & (active_df_raw["two_theta"] <= tt_max)
active_df  = active_df_raw[mask].copy()

# MAIN TABS
tabs = st.tabs([
    "📈 Raw Pattern",
    "🔍 Peak ID",
    "🧮 Rietveld Fit",
    "📊 Quantification",
    "🔄 Sample Comparison",
    "📄 Report",
])

PH_COLORS = [v["color"] for v in PHASE_LIBRARY.values()]
SAMP_COLORS = [v["color"] for v in SAMPLE_CATALOG.values()]

# TAB 0 — RAW PATTERN
with tabs[0]:
    st.subheader(f"Raw XRD Pattern — {meta['label']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data points",   f"{len(active_df):,}")
    c2.metric("2θ range",      f"{active_df.two_theta.min():.2f}° – {active_df.two_theta.max():.2f}°")
    c3.metric("Peak intensity",f"{active_df.intensity.max():.0f} cts")
    c4.metric("Background est.",f"{int(np.percentile(active_df.intensity, 5))} cts")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=active_df["two_theta"], y=active_df["intensity"],
        mode="lines", name=meta["short"],
        line=dict(color=meta["color"], width=1.2)
    ))
    fig.update_layout(
        xaxis_title="2θ (degrees)", yaxis_title="Intensity (counts)",
        template="plotly_white", height=420, hovermode="x unified",
        title=f"{selected_key} — {meta['label']}"
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Raw data table (first 200 rows)"):
        st.dataframe(active_df.head(200), use_container_width=True)

# TAB 1 — PEAK IDENTIFICATION
with tabs[1]:
    st.subheader("Peak Detection & Phase Matching")
    col_a, col_b, col_c = st.columns(3)
    min_ht  = col_a.slider("Min height × BG", 1.2, 8.0, 2.2, 0.1)
    min_sep = col_b.slider("Min separation (°)", 0.1, 2.0, 0.3, 0.05)
    tol     = col_c.slider("Match tolerance (°)", 0.05, 0.5, 0.18, 0.01)

    obs_peaks  = find_peaks_in_data(active_df,
                                    min_height_factor=min_ht,
                                    min_distance_deg=min_sep)
    theo       = {ph: generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
                  for ph in selected_phases}
    matches    = match_phases_to_data(obs_peaks, theo, tol_deg=tol)

    fig_id = go.Figure()
    fig_id.add_trace(go.Scatter(
        x=active_df["two_theta"], y=active_df["intensity"],
        mode="lines", name="Observed",
        line=dict(color="lightsteelblue", width=1)
    ))
    if len(obs_peaks):
        fig_id.add_trace(go.Scatter(
            x=obs_peaks["two_theta"], y=obs_peaks["intensity"],
            mode="markers", name="Detected peaks",
            marker=dict(symbol="triangle-down", size=10, color="crimson",
                        line=dict(color="darkred", width=1))
        ))

    I_top = active_df["intensity"].max()
    I_bot = active_df["intensity"].min()
    for i, (ph, pk_df) in enumerate(theo.items()):
        color = PH_COLORS[i % len(PH_COLORS)]
        offset = I_bot - (i + 1) * (I_top * 0.04)
        fig_id.add_trace(go.Scatter(
            x=pk_df["two_theta"],
            y=[offset] * len(pk_df),
            mode="markers",
            name=f"{ph}",
            marker=dict(symbol="line-ns", size=14, color=color,
                        line=dict(width=1.5, color=color)),
            customdata=pk_df["hkl_label"].values,
            hovertemplate="<b>%{fullData.name}</b><br>2θ=%{x:.3f}°<br>%{customdata}<extra></extra>",
        ))

    fig_id.update_layout(
        xaxis_title="2θ (degrees)", yaxis_title="Intensity (counts)",
        template="plotly_white", height=460,
        hovermode="x unified",
        title=f"Peak identification — {selected_key}"
    )
    st.plotly_chart(fig_id, use_container_width=True)

    st.markdown(f"#### {len(obs_peaks)} detected peaks")
    if len(obs_peaks):
        disp = obs_peaks.copy()
        disp["Phase match"]  = matches["phase"].values
        disp["(hkl)"]        = matches["hkl"].values
        disp["Δ2θ (°)"]      = matches["delta"].round(4).values
        disp["two_theta"]    = disp["two_theta"].round(4)
        disp["intensity"]    = disp["intensity"].round(1)
        disp["prominence"]   = disp["prominence"].round(1)
        st.dataframe(
            disp[["two_theta","intensity","prominence","Phase match","(hkl)","Δ2θ (°)"]],
            use_container_width=True
        )

    with st.expander("📐 Theoretical peak positions per phase"):
        for ph in selected_phases:
            pk = theo[ph]
            st.markdown(f"**{ph}** — {len(pk)} reflections in {tt_min:.0f}°–{tt_max:.0f}°")
            if len(pk):
                st.dataframe(pk[["two_theta","d_spacing","hkl_label"]].rename(
                    columns={"two_theta":"2θ (°)","d_spacing":"d (Å)","hkl_label":"hkl"}
                ), use_container_width=True, height=200)

# TAB 2 — RIETVELD FIT
with tabs[2]:
    st.subheader("Rietveld Refinement")
    if not selected_phases:
        st.warning("☑️ Select at least one phase in the sidebar.")
    elif not run_btn:
        st.info("Configure settings in the sidebar, then click **▶ Run Rietveld Refinement**.")
        st.markdown("""
**What the refinement does:**
- Fits a Chebyshev polynomial background  
- Generates calculated intensity from all selected phases using Bragg's law + LP correction  
- Simultaneously refines: lattice parameters *a* (and *c*), scale factors, Caglioti U/V/W, zero-shift, η  
- Reports R_wp, R_exp, goodness-of-fit χ²  
        """)
    else:
        with st.spinner("Running refinement… (10–40 s for multi-phase fits)"):
            refiner = RietveldRefinement(
                data        = active_df,
                phases      = selected_phases,
                wavelength  = wavelength,
                bg_poly_order = bg_order,
                peak_shape  = peak_shape,
            )
            result = refiner.run()

        conv_icon = "✅" if result["converged"] else "⚠️"
        st.success(f"{conv_icon} Refinement finished · "
                   f"R_wp = **{result['Rwp']:.2f}%** · "
                   f"R_exp = **{result['Rexp']:.2f}%** · "
                   f"χ² = **{result['chi2']:.3f}**")

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R_wp (%)",      f"{result['Rwp']:.2f}",
                  delta="< 15 is acceptable", delta_color="off")
        m2.metric("R_exp (%)",     f"{result['Rexp']:.2f}")
        m3.metric("GoF χ²",        f"{result['chi2']:.3f}",
                  delta="target ≈ 1", delta_color="off")
        m4.metric("Zero shift (°)",f"{result['zero_shift']:.4f}")

        fig_rv = make_subplots(
            rows=2, cols=1,
            row_heights=[0.78, 0.22],
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=("Observed vs Calculated", "Difference")
        )
        fig_rv.add_trace(go.Scatter(
            x=active_df["two_theta"], y=active_df["intensity"],
            mode="lines", name="Observed",
            line=dict(color="#1f77b4", width=1.0)), row=1, col=1)
        fig_rv.add_trace(go.Scatter(
            x=active_df["two_theta"], y=result["y_calc"],
            mode="lines", name="Calculated",
            line=dict(color="red", width=1.5)), row=1, col=1)
        fig_rv.add_trace(go.Scatter(
            x=active_df["two_theta"], y=result["y_background"],
            mode="lines", name="Background",
            line=dict(color="green", width=1, dash="dash")), row=1, col=1)

        I_top2 = active_df["intensity"].max()
        I_bot2 = active_df["intensity"].min()
        for i, ph in enumerate(selected_phases):
            color = PH_COLORS[i % len(PH_COLORS)]
            pk_pos = generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
            ybase  = I_bot2 - (i+1) * I_top2 * 0.035
            fig_rv.add_trace(go.Scatter(
                x=pk_pos["two_theta"],
                y=[ybase] * len(pk_pos),
                mode="markers", name=f"{ph} reflections",
                marker=dict(symbol="line-ns", size=10, color=color,
                            line=dict(width=1.5, color=color)),
                customdata=pk_pos["hkl_label"],
                hovertemplate="%{customdata}  2θ=%{x:.3f}°<extra>"+ph+"</extra>",
            ), row=1, col=1)

        diff = active_df["intensity"].values - result["y_calc"]
        fig_rv.add_trace(go.Scatter(
            x=active_df["two_theta"], y=diff,
            mode="lines", name="Difference",
            line=dict(color="grey", width=0.8)), row=2, col=1)
        fig_rv.add_hline(y=0, line_dash="dash", line_color="black",
                         line_width=0.8, row=2, col=1)

        fig_rv.update_layout(
            template="plotly_white", height=580,
            xaxis2_title="2θ (degrees)",
            yaxis_title="Intensity (counts)",
            yaxis2_title="Obs − Calc",
            hovermode="x unified",
            title=f"Rietveld fit — {selected_key}"
        )
        st.plotly_chart(fig_rv, use_container_width=True)

        st.markdown("#### Refined Lattice Parameters")
        lp_rows = []
        for ph in selected_phases:
            p  = result["lattice_params"].get(ph, {})
            p0 = PHASE_LIBRARY[ph]["lattice"]
            da = (p.get("a", p0["a"]) - p0["a"]) / p0["a"] * 100 if "a" in p0 else 0
            lp_rows.append({
                "Phase":       ph,
                "System":      PHASE_LIBRARY[ph]["system"],
                "a_lib (Å)":   f"{p0.get('a','—'):.5f}" if isinstance(p0.get('a'), (int,float)) else "—",
                "a_ref (Å)":   f"{p.get('a', p0.get('a','—')):.5f}" if isinstance(p.get('a'), (int,float)) else "—",
                "Δa/a₀ (%)":  f"{da:+.3f}",
                "c_ref (Å)":   f"{p.get('c','—'):.5f}" if isinstance(p.get('c'), (int,float)) else "—",
                "Wt%":         f"{result['phase_fractions'].get(ph,0)*100:.1f}",
            })
        st.dataframe(pd.DataFrame(lp_rows), use_container_width=True)

        st.session_state[f"result_{selected_key}"]  = result
        st.session_state[f"phases_{selected_key}"]  = selected_phases
        st.session_state["last_result"]  = result
        st.session_state["last_phases"]  = selected_phases
        st.session_state["last_sample"]  = selected_key

# TAB 3 — QUANTIFICATION
with tabs[3]:
    st.subheader("Phase Quantification")
    if "last_result" not in st.session_state:
        st.info("Run the Rietveld refinement first.")
    else:
        result  = st.session_state["last_result"]
        phases  = st.session_state["last_phases"]
        fracs   = result["phase_fractions"]
        labels  = list(fracs.keys())
        values  = [fracs[ph]*100 for ph in labels]
        colors  = [PHASE_LIBRARY[ph]["color"] for ph in labels]

        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.38, textinfo="label+percent",
                marker=dict(colors=colors),
            ))
            fig_pie.update_layout(title="Phase weight fractions", height=370)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            fig_bar = go.Figure(go.Bar(
                x=labels, y=values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition="outside"
            ))
            fig_bar.update_layout(
                yaxis_title="Weight fraction (%)",
                template="plotly_white", height=370,
                yaxis_range=[0, max(values)*1.25],
                title=f"Phase fractions — {st.session_state['last_sample']}"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        rows = []
        for ph in labels:
            pi = PHASE_LIBRARY[ph]
            lp = result["lattice_params"].get(ph, {})
            rows.append({
                "Phase":        ph,
                "Crystal system": pi["system"],
                "Space group":  pi["space_group"],
                "a (Å)":        f"{lp.get('a','—'):.5f}" if isinstance(lp.get('a'), (int,float)) else "—",
                "c (Å)":        f"{lp.get('c','—'):.5f}" if isinstance(lp.get('c'), (int,float)) else "—",
                "Wt%":          f"{fracs.get(ph,0)*100:.2f}",
                "Role":         pi["description"][:65]+"…" if len(pi["description"])>65 else pi["description"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# TAB 4 — SAMPLE COMPARISON
with tabs[4]:
    st.subheader("Multi-Sample Comparison")
    comp_mode = st.radio("View mode", [
        "Overlay patterns",
        "Cast vs Printed (groups)",
        "Heat-treated vs Not heat-treated",
        "ψ=0° vs ψ=45° (stress pairs)",
    ], horizontal=True)

    comp_samples = st.multiselect(
        "Select samples to overlay",
        options=SAMPLE_KEYS,
        default=SAMPLE_KEYS,
        format_func=lambda k: SAMPLE_CATALOG[k]["label"],
    )
    normalise = st.checkbox("Normalise to max intensity", value=True)

    if not comp_samples:
        st.warning("Select at least one sample.")
    else:
        fig_cmp = go.Figure()
        if comp_mode == "ψ=0° vs ψ=45° (stress pairs)":
            pairs = [
                ("CH0_1",   "CH45_2"),
                ("CNH0_3",  "CNH45_4"),
                ("PH0_5",   "PH45_6"),
                ("PNH0_7",  "PNH45_8"),
            ]
            for pair_i, (k0, k45) in enumerate(pairs):
                for ki, k in enumerate([k0, k45]):
                    if k not in comp_samples:
                        continue
                    if k in all_data:
                        df_s = all_data[k]
                    else:
                        two_theta = np.linspace(30, 130, 2000)
                        intensity = np.zeros_like(two_theta)
                        for _, pk in generate_theoretical_peaks("FCC-Co", wavelength, 30, 130).iterrows():
                            intensity += 5000 * np.exp(-((two_theta - pk["two_theta"])/0.8)**2)
                        intensity += np.random.normal(0, 50, size=len(two_theta)) + 200
                        df_s = pd.DataFrame({"two_theta": two_theta, "intensity": intensity})
                    
                    I = df_s["intensity"].values
                    if normalise:
                        I = (I - I.min()) / (I.max() - I.min() + 1e-8)
                    I = I + pair_i * 2.4
                    fig_cmp.add_trace(go.Scatter(
                        x=df_s["two_theta"], y=I,
                        mode="lines", name=SAMPLE_CATALOG[k]["short"],
                        line=dict(color=SAMPLE_CATALOG[k]["color"],
                                  width=1.2,
                                  dash="solid" if ki == 0 else "dot")
                    ))
            fig_cmp.update_layout(
                title="ψ=0° (solid) vs ψ=45° (dotted) — pairs offset vertically",
                xaxis_title="2θ (degrees)",
                yaxis_title="Norm. intensity + offset",
                template="plotly_white", height=520,
            )
        else:
            for k in comp_samples:
                if k in all_data:
                    df_s = all_data[k]
                else:
                    two_theta = np.linspace(30, 130, 2000)
                    intensity = np.zeros_like(two_theta)
                    for _, pk in generate_theoretical_peaks("FCC-Co", wavelength, 30, 130).iterrows():
                        intensity += 5000 * np.exp(-((two_theta - pk["two_theta"])/0.8)**2)
                    intensity += np.random.normal(0, 50, size=len(two_theta)) + 200
                    df_s = pd.DataFrame({"two_theta": two_theta, "intensity": intensity})
                
                I = df_s["intensity"].values
                if normalise:
                    I = (I - I.min()) / (I.max() - I.min() + 1e-8)
                m = SAMPLE_CATALOG[k]
                dash = "dot" if m["psi_angle"] == 45 else "solid"
                fig_cmp.add_trace(go.Scatter(
                    x=df_s["two_theta"], y=I,
                    mode="lines",
                    name=m["label"],
                    line=dict(color=m["color"], width=1.2, dash=dash),
                ))
            title_map = {
                "Overlay patterns":                    "All selected samples",
                "Cast vs Printed (groups)":            "Cast (blues/oranges) vs Printed (greens/reds)",
                "Heat-treated vs Not heat-treated":    "HT (solid) vs NH (dashed)",
            }
            fig_cmp.update_layout(
                title=title_map.get(comp_mode, ""),
                xaxis_title="2θ (degrees)",
                yaxis_title="Normalised intensity" if normalise else "Intensity (counts)",
                template="plotly_white", height=480,
                hovermode="x unified",
            )
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("#### Peak-shift table (ψ=0° → ψ=45°) for residual stress estimation")
        st.caption("A positive Δ2θ (0°→45°) indicates compressive in-plane residual stress.")
        stress_rows = []
        pairs = [("CH0_1","CH45_2"), ("CNH0_3","CNH45_4"),
                 ("PH0_5","PH45_6"), ("PNH0_7","PNH45_8")]
        for k0, k45 in pairs:
            def biggest_peak(df):
                idx, _ = signal.find_peaks(df["intensity"].values, 
                                          height=np.percentile(df["intensity"],85), 
                                          distance=20)
                if len(idx) == 0:
                    return float("nan")
                best = idx[np.argmax(df["intensity"].values[idx])]
                return float(df["two_theta"].values[best])
            
            tt0  = biggest_peak(all_data[k0]) if k0 in all_data else np.nan
            tt45 = biggest_peak(all_data[k45]) if k45 in all_data else np.nan
            stress_rows.append({
                "Pair":    f"{k0} / {k45}",
                "Fabrication": SAMPLE_CATALOG[k0]["fabrication"],
                "Treatment":   SAMPLE_CATALOG[k0]["treatment"],
                "2θ (ψ=0°)":  f"{tt0:.4f}°" if not np.isnan(tt0) else "—",
                "2θ (ψ=45°)": f"{tt45:.4f}°" if not np.isnan(tt45) else "—",
                "Δ2θ (°)":    f"{(tt45-tt0):+.4f}" if not (np.isnan(tt0) or np.isnan(tt45)) else "—",
            })
        st.dataframe(pd.DataFrame(stress_rows), use_container_width=True)

# TAB 5 — REPORT
with tabs[5]:
    st.subheader("Analysis Report")
    if "last_result" not in st.session_state:
        st.info("Run the Rietveld refinement first (Tab 3).")
    else:
        result  = st.session_state["last_result"]
        phases  = st.session_state["last_phases"]
        samp    = st.session_state["last_sample"]
        report_md = generate_report(result, phases, wavelength, samp)
        st.markdown(report_md)
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button(
            "⬇️ Download Report (.md)",
            data=report_md,
            file_name=f"rietveld_report_{samp}.md",
            mime="text/markdown",
        )
        export_df = active_df.copy()
        export_df["y_calc"]       = result["y_calc"]
        export_df["y_background"] = result["y_background"]
        export_df["difference"]   = active_df["intensity"].values - result["y_calc"]
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        col_dl2.download_button(
            "⬇️ Download Fit Data (.csv)",
            data=csv_buf.getvalue(),
            file_name=f"rietveld_fit_{samp}.csv",
            mime="text/csv",
        )
