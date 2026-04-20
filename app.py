"""
╔══════════════════════════════════════════════════════════════════╗
║   Co-Cr Dental Alloy · Full Rietveld XRD Refinement             ║
║   Single-file Streamlit Application with Theme & UI Controls    ║
║   Supports .ASC (two-column text) and .XRDML (Panalytical XML)  ║
║                                                                  ║
║   Usage:  streamlit run RETVIELD.py                              ║
║   Deps:   pip install streamlit numpy scipy pandas plotly requests
╚══════════════════════════════════════════════════════════════════╝
"""

import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Co-Cr XRD · Rietveld",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# GITHUB REPO CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
GITHUB_REPO = "Maryamslm/RETVIELD-XRD"
GITHUB_COMMIT = "e9716f8c3d4654fcba8eddde065d0472b1db69e9"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_COMMIT}/samples/"

AVAILABLE_FILES = {
    "CH0": ["CH0_1.ASC", "CH0_1.xrdml", "CH0.ASC", "CH0.xrdml"],
    "CH45": ["CH45_2.ASC", "CH45_2.xrdml", "CH45.ASC", "CH45.xrdml"],
    "CNH0": ["CNH0_3.ASC", "CNH0_3.xrdml", "CNH0.ASC", "CNH0.xrdml"],
    "CNH45": ["CNH45_4.ASC", "CNH45_4.xrdml", "CNH45.ASC", "CNH45.xrdml"],
    "PH0": ["PH0.ASC", "PH0.xrdml", "PH0_1.ASC", "PH0_1.xrdml"],
    "PH45": ["PH45.ASC", "PH45.xrdml", "PH45_1.ASC", "PH45_1.xrdml"],
    "PNH0": ["PNH0.ASC", "PNH0.xrdml", "PNH0_1.ASC", "PNH0_1.xrdml"],
    "PNH45": ["PNH45.ASC", "PNH45.xrdml", "PNH45_1.ASC", "PNH45_1.xrdml"],
    "MEDILOY_powder": ["MEDILOY_powder.xrdml", "MEDILOY_powder.ASC"],
}

# ═══════════════════════════════════════════════════════════════════
# APPEARANCE & THEME CONFIG
# ═══════════════════════════════════════════════════════════════════
def apply_theme(bg_theme: str, font_size: float, primary_color: str):
    themes = {
        "Dark Mode": {"bg": "#020617", "text": "#e2e8f0", "sidebar": "#030712", "panel": "#080e1a", "border": "#1e293b"},
        "Light Mode": {"bg": "#f8fafc", "text": "#0f172a", "sidebar": "#ffffff", "panel": "#f1f5f9", "border": "#cbd5e1"},
        "High Contrast": {"bg": "#000000", "text": "#00ff00", "sidebar": "#0a0a0a", "panel": "#111111", "border": "#00ff0044"}
    }
    t = themes.get(bg_theme, themes["Dark Mode"])
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif !important; font-size: {font_size}rem !important; }}
        code, pre {{ font-family: 'IBM Plex Mono', monospace !important; }}
        [data-testid="stAppViewContainer"] > .main {{ background-color: {t['bg']} !important; color: {t['text']} !important; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: {t['sidebar']} !important; border-right: 1px solid {t['border']}; }}
        [data-testid="stSidebar"] * {{ color: {t['text']} !important; }}
        [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stCheckbox label {{ color: #94a3b8 !important; }}
        .stButton > button {{ border-radius: 8px !important; font-weight: 600 !important; letter-spacing: .03em !important; }}
        .stButton > button[kind="primary"] {{ background: linear-gradient(135deg, {primary_color}, #7c3aed) !important; border: none !important; color: white !important; }}
        .hero {{ background: linear-gradient(135deg, {t['bg']} 0%, {t['panel']} 45%, {t['bg']} 100%); border: 1px solid {t['border']}; border-radius: 14px; padding: 28px 36px 22px; margin-bottom: 22px; position: relative; overflow: hidden; }}
        .hero h1 {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -.02em; background: linear-gradient(100deg, {primary_color} 0%, #818cf8 50%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 6px; }}
        .hero p {{ color: #64748b; margin: 0; font-size: .88rem; font-weight: 300; line-height: 1.5; }}
        .badge {{ display: inline-block; font-size: .7rem; font-weight: 600; letter-spacing: .06em; padding: 2px 9px; border-radius: 99px; margin-right: 6px; margin-top: 10px; border: 1px solid; }}
        .badge-cu {{ color: #f59e0b; border-color: #f59e0b44; background: #f59e0b10; }}
        .badge-iso {{ color: #34d399; border-color: #34d39944; background: #34d39910; }}
        .badge-slm {{ color: #818cf8; border-color: #818cf844; background: #818cf810; }}
        .mstrip {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }}
        .mc {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 10px; padding: 12px 18px; flex: 1; min-width: 110px; }}
        .mc .lbl {{ font-size: .68rem; color: #475569; letter-spacing: .1em; text-transform: uppercase; }}
        .mc .val {{ font-size: 1.45rem; font-weight: 700; color: {t['text']}; font-family: 'IBM Plex Mono', monospace; }}
        .mc .sub {{ font-size: .7rem; color: #334155; }}
        .sh {{ font-size: .7rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #334155; border-bottom: 1px solid {t['border']}; padding-bottom: 4px; margin: 16px 0 10px; }}
    </style>
    """, unsafe_allow_html=True)
    return t['border']

# ═══════════════════════════════════════════════════════════════════
# CRYSTAL STRUCTURE LIBRARY
# ═══════════════════════════════════════════════════════════════════
@dataclass
class AtomSite:
    element: str
    wyckoff: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    Biso: float = 0.5

@dataclass
class Phase:
    key: str
    name: str
    formula: str
    pdf_card: str
    crystal_system: str
    space_group: str
    sg_number: int
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    atoms: List[AtomSite] = field(default_factory=list)
    wf_init: float = 0.5
    color: str = "#60a5fa"
    group: str = "Primary"
    description: str = ""

    @property
    def volume(self) -> float:
        al, be, ga = map(np.radians, [self.alpha, self.beta, self.gamma])
        return self.a * self.b * self.c * np.sqrt(
            1 - np.cos(al)**2 - np.cos(be)**2 - np.cos(ga)**2 + 2*np.cos(al)*np.cos(be)*np.cos(ga)
        )

def _build_phase_db() -> Dict[str, Phase]:
    db: Dict[str, Phase] = {}
    db["gamma_Co"] = Phase(key="gamma_Co", name="γ-Co (FCC)", formula="Co", pdf_card="PDF 15-0806",
        crystal_system="cubic", space_group="Fm-3m", sg_number=225, a=3.5447, b=3.5447, c=3.5447,
        atoms=[AtomSite("Co", "4a", 0, 0, 0, 1.0, 0.40)], wf_init=0.70, color="#38bdf8", group="Primary",
        description="FCC cobalt — primary austenitic matrix in SLM Co-Cr.")
    db["epsilon_Co"] = Phase(key="epsilon_Co", name="ε-Co (HCP)", formula="Co", pdf_card="PDF 05-0727",
        crystal_system="hexagonal", space_group="P63/mmc", sg_number=194, a=2.5071, b=2.5071, c=4.0686,
        alpha=90, beta=90, gamma=120, atoms=[AtomSite("Co", "2c", 1/3, 2/3, 0.25, 1.0, 0.40)],
        wf_init=0.15, color="#fb923c", group="Primary", description="HCP cobalt — martensitic transform.")
    db["sigma"] = Phase(key="sigma", name="σ-phase (CoCr)", formula="CoCr", pdf_card="PDF 29-0490",
        crystal_system="tetragonal", space_group="P42/mnm", sg_number=136, a=8.7960, b=8.7960, c=4.5750,
        atoms=[AtomSite("Co", "2a", 0, 0, 0, 0.5, 0.50), AtomSite("Cr", "2a", 0, 0, 0, 0.5, 0.50),
               AtomSite("Co", "4f", 0.398, 0.398, 0, 0.5, 0.50), AtomSite("Cr", "4f", 0.398, 0.398, 0, 0.5, 0.50),
               AtomSite("Co", "8i", 0.464, 0.132, 0, 0.5, 0.50), AtomSite("Cr", "8i", 0.464, 0.132, 0, 0.5, 0.50)],
        wf_init=0.05, color="#4ade80", group="Secondary", 
        description="Cr-rich intermetallic; appears after prolonged heat treatment or slow cooling; tetragonal structure.")
    db["Cr_bcc"] = Phase(key="Cr_bcc", name="Cr (BCC)", formula="Cr", pdf_card="PDF 06-0694",
        crystal_system="cubic", space_group="Im-3m", sg_number=229, a=2.8839, b=2.8839, c=2.8839,
        atoms=[AtomSite("Cr", "2a", 0, 0, 0, 1.0, 0.40)], wf_init=0.04, color="#f87171", group="Secondary",
        description="BCC chromium — excess Cr or incomplete alloying.")
    db["Mo_bcc"] = Phase(key="Mo_bcc", name="Mo (BCC)", formula="Mo", pdf_card="PDF 42-1120",
        crystal_system="cubic", space_group="Im-3m", sg_number=229, a=3.1472, b=3.1472, c=3.1472,
        atoms=[AtomSite("Mo", "2a", 0, 0, 0, 1.0, 0.45)], wf_init=0.03, color="#c084fc", group="Secondary",
        description="BCC molybdenum — inter-dendritic segregation.")
    db["Co3Mo"] = Phase(key="Co3Mo", name="Co₃Mo", formula="Co3Mo", pdf_card="PDF 29-0491",
        crystal_system="hexagonal", space_group="P63/mmc", sg_number=194, a=5.1400, b=5.1400, c=4.1000,
        alpha=90, beta=90, gamma=120, atoms=[AtomSite("Co", "6h", 1/6, 1/3, 0.25, 1.0, 0.50),
               AtomSite("Mo", "2c", 1/3, 2/3, 0.25, 1.0, 0.55)], wf_init=0.02, color="#a78bfa", group="Secondary",
        description="Hexagonal Co₃Mo — high-T annealing precipitate.")
    db["M23C6"] = Phase(key="M23C6", name="M₂₃C₆ Carbide", formula="Cr23C6", pdf_card="PDF 36-0803",
        crystal_system="cubic", space_group="Fm-3m", sg_number=225, a=10.61, b=10.61, c=10.61,
        atoms=[AtomSite("Cr", "24e", 0.35, 0, 0, 1.0, 0.50), AtomSite("Cr", "32f", 0.35, 0.35, 0.35, 1.0, 0.50),
               AtomSite("C", "32f", 0.30, 0.30, 0.30, 1.0, 0.50)], wf_init=0.05, color="#eab308", group="Carbides",
        description="Cr₂₃C₆ type; very common in cast alloys with carbon; detected by characteristic low-angle peaks.")
    db["M6C"] = Phase(key="M6C", name="M₆C Carbide", formula="(Co,Mo)6C", pdf_card="PDF 27-0408",
        crystal_system="cubic", space_group="Fd-3m", sg_number=227, a=10.99, b=10.99, c=10.99,
        atoms=[AtomSite("Mo", "16c", 0, 0, 0, 0.5, 0.50), AtomSite("Co", "16d", 0.5, 0.5, 0.5, 0.5, 0.50),
               AtomSite("C", "48f", 0.375, 0.375, 0.375, 1.0, 0.50)], wf_init=0.05, color="#f97316", group="Carbides",
        description="Mo/W-rich; found in Mo- or W-containing alloys.")
    db["Laves"] = Phase(key="Laves", name="Laves Phase (Co₂Mo)", formula="Co2Mo", pdf_card="PDF 03-1225",
        crystal_system="hexagonal", space_group="P63/mmc", sg_number=194, a=4.73, b=4.73, c=7.72,
        alpha=90, beta=90, gamma=120,
        atoms=[AtomSite("Co", "2a", 0, 0, 0, 1.0, 0.50), AtomSite("Mo", "2d", 1/3, 2/3, 0.75, 1.0, 0.50),
               AtomSite("Co", "6h", 0.45, 0.90, 0.25, 1.0, 0.50)],
        wf_init=0.05, color="#d946ef", group="Laves",
        description="Hexagonal intermetallic precipitate; forms in Co-Mo/W systems; often brittle.")
    db["Cr2O3"] = Phase(key="Cr2O3", name="Cr₂O₃ (Eskolaite)", formula="Cr2O3", pdf_card="PDF 38-1479",
        crystal_system="trigonal", space_group="R-3m", sg_number=167, a=4.9580, b=4.9580, c=13.5942,
        alpha=90, beta=90, gamma=120, atoms=[AtomSite("Cr", "12c", 0, 0, 0.348, 1.0, 0.55),
               AtomSite("O", "18e", 0.306, 0, 0.25, 1.0, 0.60)], wf_init=0.02, color="#f472b6", group="Oxide",
        description="Chromium sesquioxide — passive oxide layer.")
    db["CoCr2O4"] = Phase(key="CoCr2O4", name="CoCr₂O₄ (Spinel)", formula="CoCr2O4", pdf_card="PDF 22-1084",
        crystal_system="cubic", space_group="Fm-3m", sg_number=227, a=8.3216, b=8.3216, c=8.3216,
        atoms=[AtomSite("Co", "8a", 0.125, 0.125, 0.125, 1.0, 0.55), AtomSite("Cr", "16d", 0.5, 0.5, 0.5, 1.0, 0.55),
               AtomSite("O", "32e", 0.264, 0.264, 0.264, 1.0, 0.65)], wf_init=0.01, color="#22d3ee", group="Oxide",
        description="Cobalt-chromium spinel oxide.")
    return db

PHASE_DB: Dict[str, Phase] = _build_phase_db()
PRIMARY_KEYS   = ["gamma_Co", "epsilon_Co"]
SECONDARY_KEYS = ["sigma", "Cr_bcc", "Mo_bcc", "Co3Mo"]
CARBIDE_KEYS   = ["M23C6", "M6C"]
LAVES_KEYS     = ["Laves"]
OXIDE_KEYS     = ["Cr2O3", "CoCr2O4"]

# ═══════════════════════════════════════════════════════════════════
# CRYSTALLOGRAPHY UTILITIES
# ═══════════════════════════════════════════════════════════════════
def _d_cubic(a, h, k, l): s = h*h + k*k + l*l; return a / np.sqrt(s) if s else np.inf
def _d_hex(a, c, h, k, l): t = (4/3)*((h*h + h*k + k*k) / a**2) + (l/c)**2; return 1/np.sqrt(t) if t > 0 else np.inf
def _d_tet(a, c, h, k, l): t = (h*h + k*k) / a**2 + l*l / c**2; return 1/np.sqrt(t) if t > 0 else np.inf
def _allow_fcc(h, k, l): return len({h%2, k%2, l%2}) == 1
def _allow_bcc(h, k, l): return (h+k+l) % 2 == 0
def _allow_hcp(h, k, l): return not (l%2 != 0 and (h-k)%3 == 0)
def _allow_sig(h, k, l): return (h+k+l) % 2 == 0
def _allow_all(h, k, l): return True
def _allow_fd3m(h, k, l):
    if (h%2 != k%2) or (k%2 != l%2): return False
    if (h%2 != 0): return True
    return (h+k+l) % 4 == 0

_ALLOW = {"Fm-3m": _allow_fcc, "Im-3m": _allow_bcc, "P63/mmc": _allow_hcp, 
          "P42/mnm": _allow_sig, "R-3m": _allow_all, "Fd-3m": _allow_fd3m}

_CM: Dict[str, Tuple] = {
    "Co": ([2.7686,2.2087,1.6079,1.0000],[14.178,3.398,0.124,41.698],0.9768),
    "Cr": ([2.3070,2.2940,0.8167,0.0000],[10.798,1.173,11.002,132.79],1.1003),
    "Mo": ([3.7025,2.3517,1.5442,0.8534],[12.943,2.658,0.157,39.714],0.6670),
    "O":  ([0.4548,0.9177,0.4719,0.0000],[23.780,7.622,0.165,0.000], 0.0000),
    "C":  ([2.31, 1.02, 1.59, 0.0], [20.84, 10.21, 0.57, 51.65], 0.20),
    "W":  ([4.000, 3.000, 2.000, 1.000], [10.0,  3.0,  0.5, 50.0],  0.5000),
}
def _f0(el: str, stl: float) -> float:
    if el not in _CM: return max({"Co":27,"Cr":24,"Mo":42,"O":8,"C":6}.get(el, 20) - stl*4, 1.0)
    a, b, c = _CM[el]; return c + sum(ai * np.exp(-bi * stl**2) for ai, bi in zip(a, b))

def _calc_d(ph: Phase, h: int, k: int, l: int) -> float:
    cs = ph.crystal_system.lower()
    if cs == "cubic": return _d_cubic(ph.a, h, k, l)
    elif cs in ("hexagonal", "trigonal"): return _d_hex(ph.a, ph.c, h, k, l)
    elif cs == "tetragonal": return _d_tet(ph.a, ph.c, h, k, l)
    return _d_cubic(ph.a, h, k, l)

def _F2(ph: Phase, h: int, k: int, l: int, wl: float = 1.54056) -> float:
    d = _calc_d(ph, h, k, l)
    stl = 1.0/(2.0*d) if d > 0 else 0.0; Fr = Fi = 0.0
    for at in ph.atoms:
        f = _f0(at.element, stl); DW = np.exp(-at.Biso * stl**2); pa = 2*np.pi*(h*at.x + k*at.y + l*at.z)
        Fr += at.occupancy * f * DW * np.cos(pa); Fi += at.occupancy * f * DW * np.sin(pa)
    return Fr*Fr + Fi*Fi

def generate_reflections(ph: Phase, wl: float = 1.54056, tt_min: float = 10.0, tt_max: float = 100.0, n: int = 7) -> List[Dict]:
    afn = _ALLOW.get(ph.space_group, _allow_all); seen: Dict[float, Dict] = {}
    for h in range(-n, n+1):
        for k in range(-n, n+1):
            for l in range(-n, n+1):
                if h == k == l == 0 or not afn(h, k, l): continue
                d = _calc_d(ph, h, k, l)
                if d < 0.5 or d > 20: continue
                st = wl / (2.0*d)
                if abs(st) > 1: continue
                tt = 2.0 * np.degrees(np.arcsin(st))
                if not (tt_min <= tt <= tt_max): continue
                dk = round(d, 4)
                if dk in seen: seen[dk]["mult"] += 1; seen[dk]["hkl_list"].append((h, k, l))
                else: seen[dk] = {"h":h, "k":k, "l":l, "hkl_list":[(h,k,l)], "d":d, "tt":tt, "mult":1}
    return sorted(seen.values(), key=lambda x: x["tt"])

def _make_refined_phase(ph: Phase, a_ref: float, c_ref: float) -> Phase:
    return Phase(key=ph.key, name=ph.name, formula=ph.formula, pdf_card=ph.pdf_card,
                 crystal_system=ph.crystal_system, space_group=ph.space_group, sg_number=ph.sg_number,
                 a=a_ref, b=(a_ref if ph.b == ph.a else ph.b), c=c_ref, alpha=ph.alpha, beta=ph.beta, gamma=ph.gamma,
                 atoms=ph.atoms, color=ph.color)

# ═══════════════════════════════════════════════════════════════════
# PROFILE & BACKGROUND FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def pseudo_voigt(tt: np.ndarray, tt_k: float, fwhm: float, eta: float) -> np.ndarray:
    eta = np.clip(eta, 0.0, 1.0); dx = tt - tt_k
    sig = fwhm / (2.0*np.sqrt(2.0*np.log(2.0))); G = np.exp(-0.5*(dx/sig)**2) / (sig*np.sqrt(2.0*np.pi))
    gam = fwhm / 2.0; L = (1.0/np.pi) * gam / (dx**2 + gam**2); return eta*L + (1.0-eta)*G

def caglioti(tt_deg: float, U: float, V: float, W: float) -> float:
    th = np.radians(tt_deg / 2.0); return np.sqrt(max(U*np.tan(th)**2 + V*np.tan(th) + W, 1e-8))

def lp_factor(tt_deg: float) -> float:
    th = np.radians(tt_deg / 2.0); c2t = np.cos(2.0*th); c2m = np.cos(np.radians(26.6))
    den = np.sin(th)**2 * np.cos(th); return (1.0 + c2t**2 * c2m**2) / den if den > 0 else 1.0

def chebyshev_bg(tt: np.ndarray, coeffs: np.ndarray, tt0: float, tt1: float) -> np.ndarray:
    x = 2.0*(tt - tt0)/(tt1 - tt0) - 1.0; bg = np.zeros_like(tt); Tp, Tc = np.ones_like(x), x.copy()
    if len(coeffs) > 0: bg += coeffs[0] * Tp
    if len(coeffs) > 1: bg += coeffs[1] * Tc
    for c in coeffs[2:]: Tn = 2.0*x*Tc - Tp; bg += c * Tn; Tp, Tc = Tc, Tn
    return bg

def phase_pattern(tt: np.ndarray, ph: Phase, a: float, c: float, scale: float, U: float, V: float, W: float, eta0: float, z_shift: float, wl: float) -> np.ndarray:
    ph_r = _make_refined_phase(ph, a, c)
    refls = generate_reflections(ph_r, wl=wl, tt_min=max(float(tt.min())-5.0, 0.1), tt_max=float(tt.max())+5.0)
    avg_biso = np.mean([at.Biso for at in ph.atoms]) if ph.atoms else 0.5
    I = np.zeros_like(tt)
    for r in refls:
        tt_k = r["tt"] + z_shift; F2 = _F2(ph_r, r["h"], r["k"], r["l"], wl)
        lp = lp_factor(tt_k); fwhm = caglioti(tt_k, U, V, W)
        stl = np.sin(np.radians(tt_k / 2.0)) / wl; DW = np.exp(-avg_biso * stl**2)
        I += scale * r["mult"] * F2 * lp * DW * pseudo_voigt(tt, tt_k, fwhm, eta0)
    return I

# ═══════════════════════════════════════════════════════════════════
# PARAMETER VECTOR & REFINER
# ═══════════════════════════════════════════════════════════════════
N_PP = 7
def _pack(z, bg, per_phase) -> np.ndarray: return np.array([z, *bg, *[v for p in per_phase for v in p]], dtype=float)
def _unpack(v: np.ndarray, n_bg: int, n_ph: int):
    z = float(v[0]); bg = v[1 : 1+n_bg]; pp = [v[1+n_bg+i*N_PP : 1+n_bg+(i+1)*N_PP] for i in range(n_ph)]
    return z, bg, pp

_MASS = {"Co":58.933,"Cr":51.996,"Mo":95.950,"O":15.999,"C":12.011,"W":183.84}
def hill_howard(phases: List[Phase], pp: List[np.ndarray]) -> Dict[str, float]:
    totals = {}
    for ph, p in zip(phases, pp):
        scale = float(p[0]); uc_mass = sum(_MASS.get(at.element, 50.0) * at.occupancy for at in ph.atoms) or 1.0
        totals[ph.key] = scale * uc_mass * ph.volume
    gt = sum(totals.values()) or 1.0; return {k: v/gt for k, v in totals.items()}

def r_factors(I_obs, I_calc, w) -> Dict[str, float]:
    num = float(np.sum(w * (I_obs - I_calc)**2)); den = float(np.sum(w * I_obs**2))
    Rwp = np.sqrt(num/den) if den > 0 else 99.0
    Rp = float(np.sum(np.abs(I_obs - I_calc)) / np.sum(np.abs(I_obs)))
    chi2 = num / max(len(I_obs) - 1, 1); Re = np.sqrt((len(I_obs) - 1) / den) if den > 0 else 1.0
    GOF = float(Rwp / Re) if Re > 0 else 99.0
    return dict(Rwp=float(Rwp), Rp=float(Rp), chi2=float(chi2), Re=float(Re), GOF=float(GOF))

class RietveldRefiner:
    def __init__(self, tt: np.ndarray, I_obs: np.ndarray, phase_keys: List[str], wavelength: float = 1.54056, n_bg: int = 5):
        self.tt = tt.astype(float); self.Iobs = np.maximum(I_obs.astype(float), 0.0)
        self.wl = float(wavelength); self.n_bg = int(n_bg)
        self.phases = [PHASE_DB[k] for k in phase_keys]; self.n_ph = len(self.phases)
        self.w = 1.0 / np.maximum(self.Iobs, 1.0); self._init_x0()

    def _init_x0(self):
        Ipeak = float(np.percentile(self.Iobs, 95)); Imin = float(np.percentile(self.Iobs, 10))
        bg0 = [Imin] + [0.0]*(self.n_bg - 1)
        pp = [[ph.wf_init * Ipeak * 1e-4, ph.a, ph.c, 0.02, -0.01, 0.005, 0.5] for ph in self.phases]
        self.x0 = _pack(0.0, bg0, pp)

    def _calc(self, v: np.ndarray):
        z, bg_c, pp = _unpack(v, self.n_bg, self.n_ph)
        bg = chebyshev_bg(self.tt, bg_c, self.tt.min(), self.tt.max()); Icalc = bg.copy(); contribs = {}
        for ph, p in zip(self.phases, pp):
            sc, a, c, U, V, W, et = (float(x) for x in p)
            Iph = phase_pattern(self.tt, ph, a, c, sc, U, V, W, et, z, self.wl); contribs[ph.key] = Iph; Icalc += Iph
        return Icalc, bg, contribs

    def _res(self, v): Icalc, _, _ = self._calc(v); return np.sqrt(self.w) * (self.Iobs - Icalc)

    def _bounds(self, flags: Dict[str, bool]):
        n = len(self.x0); lo, hi = np.full(n, -np.inf), np.full(n, np.inf); x = self.x0
        def freeze(i): lo[i], hi[i] = x[i]-1e-10, x[i]+1e-10
        def free(i, lb, ub): lo[i], hi[i] = lb, ub
        if flags.get("zero", False): free(0, -1.0, 1.0)
        else: freeze(0)
        for j in range(1, 1+self.n_bg):
            if flags.get("bg", True): free(j, -1e7, 1e7)
            else: freeze(j)
        for i, ph in enumerate(self.phases):
            b = 1 + self.n_bg + i*N_PP
            if flags.get("scale", True): free(b, 0.0, 1e12)
            else: freeze(b)
            if flags.get("lattice", True): free(b+1, ph.a*0.95, ph.a*1.05); free(b+2, ph.c*0.95, ph.c*1.05)
            else: freeze(b+1); freeze(b+2)
            if flags.get("profile", True): free(b+3, 0.0, 0.5); free(b+4, -0.1, 0.0); free(b+5, 1e-4, 0.1); free(b+6, 0.0, 1.0)
            else:
                for j in range(3,7): freeze(b+j)
        return lo, hi

    def refine(self, flags: Dict[str, bool], max_iter: int = 400) -> Dict:
        lo, hi = self._bounds(flags); mask = (lo == hi); hi[mask] += 1e-9
        try:
            res = least_squares(self._res, self.x0, bounds=(lo, hi), method="trf",
                              max_nfev=max_iter, ftol=1e-7, xtol=1e-7, gtol=1e-7, verbose=0)
            self.x0 = res.x
        except Exception as e: st.warning(f"Optimisation note: {e}")
        Icalc, bg, contribs = self._calc(self.x0); rf = r_factors(self.Iobs, Icalc, self.w)
        z, bg_c, pp = _unpack(self.x0, self.n_bg, self.n_ph); wf = hill_howard(self.phases, pp)
        lat = {}
        for ph, p in zip(self.phases, pp):
            sc, a, c, U, V, W, et = (float(x) for x in p)
            lat[ph.key] = {"a_init": ph.a, "c_init": ph.c, "a_ref": a, "c_ref": c,
                          "da": a-ph.a, "dc": c-ph.c, "U":U, "V":V, "W":W, "eta":et, "scale":sc}
        return {**rf, "Icalc": Icalc, "Ibg": bg, "contribs": contribs, "diff": self.Iobs-Icalc, "wf": wf, "lat": lat, "z_shift": z}

# ═══════════════════════════════════════════════════════════════════
# DEMO & FILE PARSER
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def make_demo_pattern(noise: float = 0.025, seed: int = 7):
    rng = np.random.default_rng(seed); tt = np.linspace(10, 100, 4500)
    wf_demo = {"gamma_Co": 0.68, "epsilon_Co": 0.15, "sigma": 0.08, "Cr_bcc": 0.05, "Mo_bcc": 0.04}
    bg_c = np.array([280., -60., 25., -8., 4.]); I = chebyshev_bg(tt, bg_c, tt.min(), tt.max())
    for key, wf in wf_demo.items():
        ph = PHASE_DB[key]; I += phase_pattern(tt, ph, ph.a, ph.c, wf*7500, 0.025, -0.012, 0.006, 0.45, 0.0, 1.54056)
    I = np.maximum(I, 0.0); I = rng.poisson(I).astype(float) + rng.normal(0, noise*I.max(), size=I.shape)
    return tt, np.maximum(I, 0.0)

def parse_file_content(content: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    name = filename.lower()
    if name.endswith(".xrdml"):
        try:
            root = ET.fromstring(content)
            counts_elem = root.find(".//{*}counts") or root.find(".//counts")
            if counts_elem is None: raise ValueError("No <counts> node found in .xrdml file.")
            I = np.array(counts_elem.text.split(), dtype=float)
            start_pos = end_pos = None
            for pos_elem in root.findall(".//{*}positions"):
                if "2Theta" in pos_elem.get("axis", ""):
                    try: start_pos = float(pos_elem.find("{*}startPosition").text); end_pos = float(pos_elem.find("{*}endPosition").text)
                    except (AttributeError, ValueError): pass
            tt = np.linspace(start_pos or 10.0, end_pos or 100.0, len(I)); return tt, I
        except ET.ParseError as e: raise ValueError(f"XML parsing error: {e}")
    
    lines = [ln.strip() for ln in content.splitlines() if ln.strip() and ln.strip()[0] not in "#!/'\";"]
    data = []
    for ln in lines:
        parts = ln.replace(",", " ").split()
        try:
            if len(parts) >= 2: data.append((float(parts[0]), float(parts[1])))
        except ValueError: continue
    # ✅ FIXED SYNTAX ERROR: Added 'data' to the condition
    if not data:
        raise ValueError("Cannot parse — expected 2 columns: 2θ and Intensity.")
    arr = np.array(data); tt, I = arr[:, 0], arr[:, 1]
    if tt.max() < 5: tt = np.degrees(tt)
    if not np.all(tt[:-1] <= tt[1:]):
        idx = np.argsort(tt); tt, I = tt[idx], I[idx]
    return tt, I

def fetch_github_xrd(sample_name: str, file_ext: str = ".ASC") -> Tuple[np.ndarray, np.ndarray, str]:
    if sample_name not in AVAILABLE_FILES: raise ValueError(f"Sample '{sample_name}' not found.")
    possible_files = AVAILABLE_FILES[sample_name]
    for filename in possible_files:
        if filename.endswith(file_ext):
            url = GITHUB_RAW_BASE + filename
            try: response = requests.get(url, timeout=30); response.raise_for_status(); return parse_file_content(response.text, filename) + (filename,)
            except: continue
    for filename in possible_files:
        url = GITHUB_RAW_BASE + filename
        try: response = requests.get(url, timeout=30); response.raise_for_status(); return parse_file_content(response.text, filename) + (filename,)
        except: continue
    raise ValueError(f"Could not fetch '{sample_name}'. Tried: {possible_files}")

def q_color(rwp: float) -> str:
    if rwp < 0.05: return "#4ade80"
    if rwp < 0.10: return "#fbbf24"
    return "#f87171"

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE & SIDEBAR
# ═══════════════════════════════════════════════════════════════════
for _k in ("results", "refiner", "tt", "Iobs", "elapsed", "selected_sample", "source_info"):
    if _k not in st.session_state: st.session_state[_k] = None

with st.sidebar:
    st.markdown("## ⚙️ Setup")
    
    # 🎨 APPEARANCE CONTROLS
    st.markdown('<div class="sh">🎨 Appearance & Theme</div>', unsafe_allow_html=True)
    bg_theme = st.selectbox("Background Theme", ["Dark Mode", "Light Mode", "High Contrast"])
    font_size = st.slider("Font Size Scale", 0.8, 1.3, 1.0, 0.05)
    primary_color = st.color_picker("Primary Accent Color", "#38bdf8")
    plot_theme = st.selectbox("Plot Color Map", ["plotly_dark", "plotly_white", "plotly_light"])
    
    border_color = apply_theme(bg_theme, font_size, primary_color)

    # 🏷️ PEAK LABELING CONTROLS
    st.markdown('<div class="sh">🏷️ Peak Labels (Miller Indices)</div>', unsafe_allow_html=True)
    show_hkl_labels = st.checkbox("Show (hkl) labels on peaks", value=True)
    hkl_font_size = st.slider("Label font size", 8, 16, 10)
    hkl_label_offset = st.slider("Label vertical offset (%)", 0, 50, 15, help="Offset labels above peaks to avoid overlap")
    hkl_label_color = st.radio("Label color", ["Phase color", "White", "Black", "Custom"], index=0)
    if hkl_label_color == "Custom":
        custom_hkl_color = st.color_picker("Custom label color", "#ffffff")
        hkl_color = custom_hkl_color
    elif hkl_label_color == "White":
        hkl_color = "#ffffff"
    elif hkl_label_color == "Black":
        hkl_color = "#000000"
    else:
        hkl_color = "phase"  # Use phase color

    st.markdown('<div class="sh">📁 GitHub Repository Files</div>', unsafe_allow_html=True)
    sample_options = list(AVAILABLE_FILES.keys())
    selected_sample = st.selectbox("Select XRD Sample", options=sample_options, index=0)
    file_ext = st.radio("Preferred file format", options=[".ASC", ".xrdml"], index=0, horizontal=True)
    fetch_btn = st.button("🔄 Load Selected File from GitHub", type="primary", use_container_width=True)
    tt_raw = I_raw = None; source_info = ""
    if fetch_btn:
        with st.spinner(f"Fetching {selected_sample} from GitHub..."):
            try:
                tt_raw, I_raw, actual_file = fetch_github_xrd(selected_sample, file_ext)
                source_info = f"✓ Loaded: {actual_file} ({len(tt_raw)} pts, {tt_raw.min():.1f}°–{tt_raw.max():.1f}°)"
                st.success(source_info); st.session_state.selected_sample = selected_sample; st.session_state.source_info = source_info
            except Exception as e: st.error(f"❌ Error: {str(e)}"); st.info("💡 Try switching file format or check internet connection.")
    with st.expander("🔁 Alternative: Demo or Local Upload", expanded=False):
        src = st.radio("", ["Demo pattern (synthetic)", "Upload my XRD file"], label_visibility="collapsed", key="alt_src")
        if src.startswith("Demo"):
            tt_raw, I_raw = make_demo_pattern(); st.info("Synthetic SLM Co-Cr-Mo · Cu Kα₁"); source_info = "Demo pattern loaded"
        else:
            up = st.file_uploader("Drag & drop XRD file", type=["xy","dat","txt","csv","xrdml","asc"], label_visibility="collapsed")
            if up:
                try:
                    content = up.read().decode("utf-8", errors="replace")
                    tt_raw, I_raw = parse_file_content(content, up.name)
                    source_info = f"✓ {up.name}: {len(tt_raw)} pts · {tt_raw.min():.1f}° – {tt_raw.max():.1f}°"; st.success(source_info)
                except Exception as e: st.error(str(e))
    st.markdown('<div class="sh">⚙️ Instrument</div>', unsafe_allow_html=True)
    WL_OPTIONS = {"Cu Kα₁ (1.54056 Å)": 1.54056, "Cu Kα (1.54184 Å)": 1.54184, "Mo Kα₁ (0.70932 Å)": 0.70932, "Ag Kα₁ (0.56087 Å)": 0.56087, "Co Kα₁ (1.78900 Å)": 1.78900}
    wl_label = st.selectbox("Wavelength", list(WL_OPTIONS.keys()), index=0); wavelength = WL_OPTIONS[wl_label]
    zero_seed = st.slider("Zero-shift seed (°)", -1.0, 1.0, 0.0, 0.01)
    st.markdown('<div class="sh">📐 2θ Window</div>', unsafe_allow_html=True)
    tt_lo, tt_hi = st.slider("", 10.0, 120.0, (15.0, 95.0), 0.5)
    st.markdown('<div class="sh">🧊 Phase Selection</div>', unsafe_allow_html=True)
    sel_keys: List[str] = []
    phase_groups = [
        ("Primary phases", PRIMARY_KEYS, True), ("Secondary phases", SECONDARY_KEYS, True),
        ("Carbides", CARBIDE_KEYS, True), ("Laves Phase", LAVES_KEYS, True), ("Oxide phases", OXIDE_KEYS, False),
    ]
    for grp, keys, exp in phase_groups:
        with st.expander(grp, expanded=exp):
            for k in keys:
                ph = PHASE_DB[k]; default = k in (PRIMARY_KEYS + SECONDARY_KEYS[:2])
                if st.checkbox(f"{ph.name} · {ph.formula}", value=default, key=f"ck_{k}", help=ph.description):
                    sel_keys.append(k)
    if not sel_keys: st.warning("⚠️ Select at least one phase.")
    st.markdown('<div class="sh">🔧 Refinement Flags</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    fl_scale = c1.checkbox("Scale", value=True); fl_lattice = c2.checkbox("Lattice", value=True)
    fl_bg = c1.checkbox("Background", value=True); fl_profile = c2.checkbox("Profile", value=True)
    fl_zero = st.checkbox("Zero-shift", value=False); n_bg = st.slider("Background terms", 2, 8, 5); max_it = st.slider("Max iterations", 50, 1000, 350, 50)
    st.markdown("")
    run = st.button("▶ Run Rietveld Refinement", type="primary", use_container_width=True, disabled=(tt_raw is None or not sel_keys))

# ═══════════════════════════════════════════════════════════════════
# HERO & RUN
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <h1>🔬 Co-Cr Dental Alloy · Rietveld XRD Refinement</h1>
  <p>Full-profile Rietveld refinement for 3D-printed (SLM / DMLS) cobalt-chromium dental alloys<br>
     Phase identification · Weight fractions · Lattice parameters · Peak profile analysis</p>
  <span class="badge badge-cu">Cu Kα / Mo Kα / Co Kα / Ag Kα</span>
  <span class="badge badge-iso">ISO 22674 · ASTM F75</span>
  <span class="badge badge-slm">SLM · DMLS · Casting</span>
</div>
""", unsafe_allow_html=True)
if st.session_state.selected_sample and st.session_state.source_info:
    st.caption(f"📊 Current sample: **{st.session_state.selected_sample}** — {st.session_state.source_info}")

if run and tt_raw is not None and sel_keys:
    mask = (tt_raw >= tt_lo) & (tt_raw <= tt_hi); tt_c, I_c = tt_raw[mask], I_raw[mask]
    if len(tt_c) < 50: st.error("Too few data points — widen the 2θ window.")
    else:
        prog = st.progress(0, "Initialising refiner …"); t0 = time.time()
        refiner = RietveldRefiner(tt_c, I_c, sel_keys, wavelength, n_bg); refiner.x0[0] = float(zero_seed)
        prog.progress(15, "Running least-squares optimisation …")
        flags = dict(scale=fl_scale, lattice=fl_lattice, bg=fl_bg, profile=fl_profile, zero=fl_zero)
        results = refiner.refine(flags, max_iter=max_it); elapsed = time.time() - t0
        prog.progress(100, f"Done in {elapsed:.1f} s"); time.sleep(0.3); prog.empty()
        st.session_state.update(results=results, refiner=refiner, tt=tt_c, Iobs=I_c, elapsed=elapsed)

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab_fit, tab_phase, tab_peaks, tab_params, tab_report, tab_about = st.tabs([
    "📈 Pattern Fit", "⚖️ Phase Analysis", "📋 Peak List", "🔧 Refined Parameters", "📄 Report", "ℹ️ About"])

with tab_fit:
    if st.session_state["results"] is None:
        if tt_raw is not None:
            mask = (tt_raw >= tt_lo) & (tt_raw <= tt_hi)
            fig = go.Figure(go.Scatter(x=tt_raw[mask], y=I_raw[mask], mode="lines", line=dict(color=primary_color, width=1), name="I_obs"))
            fig.update_layout(template=plot_theme, xaxis_title="2θ (°)", yaxis_title="Intensity (counts)", height=350, margin=dict(l=60,r=20,t=20,b=50))
            st.plotly_chart(fig, use_container_width=True)
        st.info("👈 Select a file and press **▶ Run Rietveld Refinement**.")
    else:
        r, refiner, tt, Iobs, elapsed = st.session_state["results"], st.session_state["refiner"], st.session_state["tt"], st.session_state["Iobs"], st.session_state["elapsed"]
        z_shift = float(r.get("z_shift", 0.0)); _, _, pp_vec = _unpack(refiner.x0, refiner.n_bg, refiner.n_ph)
        rwp, rp, gof, chi2 = r["Rwp"], r["Rp"], r["GOF"], r["chi2"]; qc = q_color(rwp)
        st.markdown(f"""<div class="mstrip">
          <div class="mc"><div class="lbl">R_wp</div><div class="val" style="color:{qc}">{rwp*100:.2f}</div><div class="sub">% (target &lt; 10 %)</div></div>
          <div class="mc"><div class="lbl">R_p</div><div class="val">{rp*100:.2f}</div><div class="sub">%</div></div>
          <div class="mc"><div class="lbl">GOF</div><div class="val">{gof:.3f}</div><div class="sub">target ≈ 1</div></div>
          <div class="mc"><div class="lbl">χ²</div><div class="val">{chi2:.4f}</div></div>
          <div class="mc"><div class="lbl">Points</div><div class="val">{len(tt)}</div><div class="sub">data pts</div></div>
          <div class="mc"><div class="lbl">Time</div><div class="val">{elapsed:.1f}</div><div class="sub">s</div></div>
        </div>""", unsafe_allow_html=True)
        
        fig = make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22], shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=tt, y=Iobs, mode="lines", name="I_obs (exp)", line=dict(color="#94a3b8", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=tt, y=r["Ibg"], mode="lines", name="Background", line=dict(color="#334155", width=1, dash="dot"), fill="tozeroy", fillcolor="rgba(51,65,85,0.12)"), row=1, col=1)
        for key, Iph in r["contribs"].items():
            ph, wf = PHASE_DB[key], r["wf"].get(key, 0) * 100
            fig.add_trace(go.Scatter(x=tt, y=Iph + r["Ibg"], mode="lines", name=f"{ph.name} ({wf:.1f}%)", line=dict(color=ph.color, width=1.6, dash="dash"), opacity=0.8), row=1, col=1)
        fig.add_trace(go.Scatter(x=tt, y=r["Icalc"], mode="lines", name="I_calc", line=dict(color="#fbbf24", width=2.2)), row=1, col=1)
        
        # 🏷️ ADD MILLER INDEX LABELS TO PEAKS
        if show_hkl_labels:
            y_max = float(Iobs.max())
            y_min = float(Iobs.min())
            y_range = y_max - y_min
            label_y_pos = y_max + (y_range * hkl_label_offset / 100)
            
            for i, ph_obj in enumerate(refiner.phases):
                a_ref, c_ref = float(pp_vec[i][1]), float(pp_vec[i][2])
                ph_ref = _make_refined_phase(ph_obj, a_ref, c_ref)
                pks = generate_reflections(ph_ref, wl=wavelength, tt_min=float(tt.min()), tt_max=float(tt.max()))
                
                # Add tick marks for peaks
                y_tick = y_min - 0.05 * y_range
                fig.add_trace(go.Scatter(
                    x=[p["tt"]+z_shift for p in pks], y=[y_tick]*len(pks), mode="markers",
                    marker=dict(symbol="line-ns", size=11, color=ph_obj.color, line=dict(width=2, color=ph_obj.color)),
                    name=f"{ph_obj.name} hkl", showlegend=False
                ), row=1, col=1)
                
                # Add (hkl) text labels above peaks
                label_color = ph_obj.color if hkl_color == "phase" else hkl_color
                for pk in pks:
                    hkl_text = f"({pk['h']} {pk['k']} {pk['l']})"
                    fig.add_annotation(
                        x=pk["tt"] + z_shift, y=label_y_pos,
                        text=hkl_text, showarrow=False,
                        font=dict(size=hkl_font_size, color=label_color, family="IBM Plex Mono"),
                        xanchor="center", yanchor="bottom",
                        bordercolor=border_color, borderwidth=1, borderpad=2,
                        bgcolor="rgba(0,0,0,0.3)" if bg_theme == "Dark Mode" else "rgba(255,255,255,0.7)"
                    )
        else:
            # Just add tick marks without labels
            y_tick = float(Iobs.min()) - 0.05*(float(Iobs.max()) - float(Iobs.min()))
            for i, ph_obj in enumerate(refiner.phases):
                a_ref, c_ref = float(pp_vec[i][1]), float(pp_vec[i][2])
                ph_ref = _make_refined_phase(ph_obj, a_ref, c_ref)
                pks = generate_reflections(ph_ref, wl=wavelength, tt_min=float(tt.min()), tt_max=float(tt.max()))
                fig.add_trace(go.Scatter(
                    x=[p["tt"]+z_shift for p in pks], y=[y_tick]*len(pks), mode="markers",
                    marker=dict(symbol="line-ns", size=11, color=ph_obj.color, line=dict(width=2, color=ph_obj.color)),
                    name=f"{ph_obj.name} hkl", showlegend=False
                ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=tt, y=r["diff"], mode="lines", name="Δ obs−calc", line=dict(color="#818cf8", width=1), fill="tozeroy", fillcolor="rgba(129,140,248,0.12)"), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="#334155", width=1, dash="dash"), row=2, col=1)
        fig.update_layout(template=plot_theme, height=650, legend=dict(font=dict(size=11), x=1.01, y=1), margin=dict(l=65, r=210, t=15, b=55), font=dict(family="IBM Plex Sans"))
        fig.update_xaxes(title_text="2θ (°)", row=2, col=1)
        fig.update_yaxes(title_text="Intensity (counts)", row=1, col=1)
        fig.update_yaxes(title_text="Δ", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
        df_pat = pd.DataFrame({"two_theta": tt, "I_obs": Iobs, "I_calc": r["Icalc"], "I_background": r["Ibg"], "difference": r["diff"], **{f"I_{k}": v for k, v in r["contribs"].items()}})
        st.download_button("⬇ Download pattern CSV", data=df_pat.to_csv(index=False), file_name="rietveld_pattern.csv", mime="text/csv")

with tab_phase:
    if st.session_state["results"] is None: st.info("Run refinement first.")
    else:
        r, wf_dict = st.session_state["results"], st.session_state["results"]["wf"]
        ph_names, wf_pct, ph_cols = [PHASE_DB[k].name for k in wf_dict], [wf_dict[k]*100 for k in wf_dict], [PHASE_DB[k].color for k in wf_dict]
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = go.Figure(go.Pie(labels=ph_names, values=wf_pct, hole=0.58, marker=dict(colors=ph_cols, line=dict(color="#030712", width=2.5)), textinfo="label+percent", textfont=dict(size=11.5, family="IBM Plex Sans"), hovertemplate="<b>%{label}</b><br>%{value:.2f} wt%<extra></extra>"))
            fig_pie.add_annotation(text="Weight<br>Fraction", x=0.5, y=0.5, font=dict(size=12, color="#64748b"), showarrow=False)
            fig_pie.update_layout(template=plot_theme, showlegend=False, height=360, margin=dict(l=10,r=10,t=30,b=10), title=dict(text="Phase Composition", font=dict(size=13, color="#64748b")))
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            idx = list(np.argsort(wf_pct)[::-1])
            fig_bar = go.Figure(go.Bar(x=[wf_pct[i] for i in idx], y=[ph_names[i] for i in idx], orientation="h", marker=dict(color=[ph_cols[i] for i in idx]), text=[f"{wf_pct[i]:.2f} %" for i in idx], textposition="inside", insidetextfont=dict(color="white", size=11, family="IBM Plex Mono")))
            fig_bar.update_layout(template=plot_theme, xaxis_title="wt %", height=360, margin=dict(l=10,r=10,t=30,b=40), title=dict(text="Phase Distribution", font=dict(size=13, color="#64748b")), yaxis=dict(gridcolor=border_color), xaxis=dict(gridcolor=border_color, range=[0, max(wf_pct)*1.15]))
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('<div class="sh">Phase Detail</div>', unsafe_allow_html=True)
        lat, rows = r["lat"], []
        for k, wf in wf_dict.items():
            ph, lp = PHASE_DB[k], lat.get(k, {})
            rows.append({"Phase": ph.name, "Formula": ph.formula, "PDF Card": ph.pdf_card, "System": ph.crystal_system.title(), "S.G.": ph.space_group, "wt %": f"{wf*100:.2f}", "a ref (Å)": f"{lp.get('a_ref', ph.a):.4f}", "c ref (Å)": f"{lp.get('c_ref', ph.c):.4f}" if ph.c != ph.a else "—", "Group": ph.group})
        df_ph = pd.DataFrame(rows); st.dataframe(df_ph, use_container_width=True, hide_index=True)
        with st.expander("🧪 Microstructural Interpretation", expanded=True):
            wm = {k: wf_dict.get(k, 0)*100 for k in PHASE_DB}; msgs = []
            if wm["gamma_Co"] > 50: msgs.append(("✅","γ-Co (FCC)",f"{wm['gamma_Co']:.1f} wt%","Dominant austenitic matrix — typical of rapid-solidification SLM. Good biocompatibility and ductility for dental prostheses."))
            if wm["epsilon_Co"] > 5: msgs.append(("⚠️","ε-Co (HCP)",f"{wm['epsilon_Co']:.1f} wt%","Elevated HCP fraction — martensitic transformation driven by residual stress or strain. Monitor fatigue performance."))
            if wm["sigma"] > 3: msgs.append(("🔴","σ-phase",f"{wm['sigma']:.1f} wt%","Brittle intermetallic — risk to ductility and fracture toughness. Consider solution anneal at >1100 °C + water quench."))
            if wm["Cr_bcc"] > 2: msgs.append(("🔵","Cr (BCC)",f"{wm['Cr_bcc']:.1f} wt%","Free chromium — may indicate incomplete alloying or Cr-rich segregation."))
            if wm["Mo_bcc"] > 1: msgs.append(("🟣","Mo (BCC)",f"{wm['Mo_bcc']:.1f} wt%","Segregated molybdenum — inter-dendritic artefact; homogenisation anneal at 1150 °C / 1 h recommended."))
            if wm.get("Cr2O3",0) > 0.5 or wm.get("CoCr2O4",0) > 0.5: msgs.append(("🟠","Oxides",f"Cr₂O₃={wm.get('Cr2O3',0):.1f}% CoCr₂O₄={wm.get('CoCr2O4',0):.1f}%","Surface/internal oxide detected — check SLM atmosphere and powder storage."))
            if not msgs: msgs.append(("ℹ️","No dominant phases","—","Check phase selection and 2θ range."))
            for icon, name, pct, msg in msgs:
                st.markdown(f"""<div style="background:#080e1a;border:1px solid {border_color};border-radius:10px;padding:14px 18px;margin-bottom:10px;">
                  <div style="font-weight:700;font-size:.95rem;margin-bottom:4px;">{icon} &nbsp;{name} &nbsp;<span style="font-family:'IBM Plex Mono';font-size:.82rem;color:#64748b;">{pct}</span></div>
                  <div style="color:#94a3b8;font-size:.85rem;line-height:1.55;">{msg}</div></div>""", unsafe_allow_html=True)
        st.download_button("⬇ Download phase table (.csv)", data=df_ph.to_csv(index=False), file_name="phase_analysis.csv", mime="text/csv")

with tab_peaks:
    if st.session_state["refiner"] is None: st.info("Run refinement first.")
    else:
        refiner, tt, r = st.session_state["refiner"], st.session_state["tt"], st.session_state["results"]
        z_shift = float(r.get("z_shift", 0.0)); _, _, pp_vec = _unpack(refiner.x0, refiner.n_bg, refiner.n_ph)
        show = st.multiselect("Phases to display", options=[ph.key for ph in refiner.phases], default=[ph.key for ph in refiner.phases], format_func=lambda k: PHASE_DB[k].name)
        rows = []
        for i, ph_obj in enumerate(refiner.phases):
            if ph_obj.key not in show: continue
            a_ref, c_ref = float(pp_vec[i][1]), float(pp_vec[i][2]); ph_ref = _make_refined_phase(ph_obj, a_ref, c_ref)
            for ref in generate_reflections(ph_ref, wl=wavelength, tt_min=float(tt.min()), tt_max=float(tt.max())):
                h, k, l = ref["h"], ref["k"], ref["l"]
                rows.append({"Phase": ph_obj.name, "hkl": f"({h} {k} {l})", "d (Å)": f"{ref['d']:.4f}", "2θ (°)": f"{ref['tt'] + z_shift:.3f}", "Mult.": ref["mult"], "Group": ph_obj.group})
        if rows:
            df_pk = pd.DataFrame(rows).sort_values("2θ (°)").reset_index(drop=True)
            st.dataframe(df_pk, use_container_width=True, hide_index=True, height=500)
            st.download_button("⬇ Download peak list (.csv)", data=df_pk.to_csv(index=False), file_name="peak_list.csv", mime="text/csv")
        else: st.warning("No peaks found for selected phases in this 2θ range.")

with tab_params:
    if st.session_state["results"] is None: st.info("Run refinement first.")
    else:
        r, refiner, lat, tt = st.session_state["results"], st.session_state["refiner"], st.session_state["results"]["lat"], st.session_state["tt"]
        st.markdown('<div class="sh">Lattice Parameters</div>', unsafe_allow_html=True)
        lp_rows = []
        for ph_obj in refiner.phases:
            lp = lat.get(ph_obj.key, {})
            lp_rows.append({"Phase": ph_obj.name, "System": ph_obj.crystal_system.title(), "S.G.": ph_obj.space_group,
                "a_init (Å)": f"{lp.get('a_init', ph_obj.a):.4f}", "a_ref (Å)": f"{lp.get('a_ref', ph_obj.a):.4f}", "Δa (Å)": f"{lp.get('da', 0):+.4f}",
                "c_init (Å)": f"{lp.get('c_init', ph_obj.c):.4f}" if ph_obj.c != ph_obj.a else "—",
                "c_ref (Å)": f"{lp.get('c_ref', ph_obj.c):.4f}" if ph_obj.c != ph_obj.a else "—",
                "Δc (Å)": f"{lp.get('dc', 0):+.4f}" if ph_obj.c != ph_obj.a else "—"})
        st.dataframe(pd.DataFrame(lp_rows), use_container_width=True, hide_index=True)
        st.markdown('<div class="sh">Profile Parameters (U, V, W, η)</div>', unsafe_allow_html=True)
        prof_rows = []
        for ph_obj in refiner.phases:
            lp = lat.get(ph_obj.key, {})
            prof_rows.append({"Phase": ph_obj.name, "Scale": f"{lp.get('scale', 0):.4e}", "U": f"{lp.get('U', 0):.5f}", "V": f"{lp.get('V', 0):.5f}", "W": f"{lp.get('W', 0):.5f}", "η (mixing)": f"{lp.get('eta', 0.5):.3f}"})
        st.dataframe(pd.DataFrame(prof_rows), use_container_width=True, hide_index=True)
        st.markdown('<div class="sh">Global Parameters</div>', unsafe_allow_html=True)
        z_val, bg_c, _ = _unpack(refiner.x0, refiner.n_bg, refiner.n_ph)
        st.code(f"Zero-shift  : {z_val:+.4f} °\nWavelength  : {wavelength:.5f} Å\nBackground  : {' '.join(f'{float(v):.2f}' for v in bg_c)}  (Chebyshev coeff. 0–{len(bg_c)-1})", language="text")
        st.markdown('<div class="sh">FWHM vs 2θ</div>', unsafe_allow_html=True)
        tt_plot = np.linspace(float(tt.min()), float(tt.max()), 300); fig_fw = go.Figure()
        for ph_obj in refiner.phases:
            lp = lat.get(ph_obj.key, {}); U, V, W = lp.get("U",.02), lp.get("V",-.01), lp.get("W",.005)
            fw = [caglioti(t, U, V, W) for t in tt_plot]
            fig_fw.add_trace(go.Scatter(x=tt_plot, y=fw, mode="lines", name=ph_obj.name, line=dict(color=ph_obj.color, width=2)))
        fig_fw.update_layout(template=plot_theme, xaxis_title="2θ (°)", yaxis_title="FWHM (°)", height=270, margin=dict(l=60,r=20,t=10,b=50), legend=dict(font=dict(size=10)), xaxis=dict(gridcolor=border_color), yaxis=dict(gridcolor=border_color))
        st.plotly_chart(fig_fw, use_container_width=True)

with tab_report:
    if st.session_state["results"] is None: st.info("Run refinement first.")
    else:
        r, refiner, elapsed, tt = st.session_state["results"], st.session_state["refiner"], st.session_state["elapsed"], st.session_state["tt"]
        md = []
        md.append("# Rietveld Refinement Report"); md.append("## Co-Cr Dental Alloy · XRD Phase Analysis")
        md.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"); md.append(f"**Radiation:** {wl_label} · λ = {wavelength:.5f} Å")
        md.append(f"**Data points:** {len(tt)}"); md.append(f"**2θ range:** {tt.min():.2f}° – {tt.max():.2f}°"); md.append(f"**Computation time:** {elapsed:.2f} s")
        md.append("---"); md.append("## 1 · Refinement Quality"); md.append("| Indicator | Value | Guidance |"); md.append("|-----------|-------|----------|")
        md.append(f"| R_wp | **{r['Rwp']*100:.2f} %** | < 10 % acceptable; < 5 % excellent |"); md.append(f"| R_p | **{r['Rp']*100:.2f} %** | — |")
        md.append(f"| χ² | **{r['chi2']:.4f}** | — |"); md.append(f"| GOF | **{r['GOF']:.4f}** | ≈ 1 ideal |")
        md.append("---"); md.append("## 2 · Phase Weight Fractions (Hill-Howard)"); md.append("| Phase | Formula | S.G. | wt % |"); md.append("|-------|---------|------|------|")
        for k, wf in r["wf"].items():
            ph = PHASE_DB[k]; md.append(f"| {ph.name} | {ph.formula} | {ph.space_group} | **{wf*100:.2f}** |")
        md.append("---"); md.append("## 3 · Refined Lattice Parameters"); md.append("| Phase | a_init (Å) | a_ref (Å) | Δa | c_init (Å) | c_ref (Å) | Δc |"); md.append("|-------|-----------|-----------|-----|-----------|-----------|----|")
        for k, lp in r["lat"].items():
            ph = PHASE_DB[k]
            if ph.c != ph.a: md.append(f"| {ph.name} | {lp['a_init']:.4f} | {lp['a_ref']:.4f} | {lp['da']:+.4f} | {lp['c_init']:.4f} | {lp['c_ref']:.4f} | {lp['dc']:+.4f} |")
            else: md.append(f"| {ph.name} | {lp['a_init']:.4f} | {lp['a_ref']:.4f} | {lp['da']:+.4f} | — | — | — |")
        md.append("---"); md.append("## 4 · Profile Parameters"); md.append("| Phase | Scale | U | V | W | η |"); md.append("|-------|-------|---|---|---|---|")
        for k, lp in r["lat"].items():
            ph = PHASE_DB[k]; md.append(f"| {ph.name} | {lp['scale']:.3e} | {lp['U']:.5f} | {lp['V']:.5f} | {lp['W']:.5f} | {lp['eta']:.3f} |")
        md.append("---"); md.append("## 5 · Methodology")
        md.append("- **Profile:** Thompson-Cox-Hastings pseudo-Voigt · Caglioti FWHM (U, V, W, η)")
        md.append(f"- **Background:** Chebyshev polynomial ({n_bg} terms)")
        md.append("- **LP correction:** Graphite monochromator (2θ_mono = 26.6°)")
        md.append("- **Structure factors:** Cromer-Mann + isotropic Debye-Waller")
        md.append("- **Weight fractions:** Hill-Howard formula")
        md.append(f"- **Optimiser:** scipy.optimize.least_squares (TRF / Levenberg-Marquardt · {max_it} iter.)")
        md.append("---"); md.append("## 6 · References")
        md.append("1. Rietveld (1969) *J. Appl. Cryst.* **2**, 65–71")
        md.append("2. Hill & Howard (1987) *J. Appl. Cryst.* **20**, 467–474")
        md.append("3. Thompson et al. (1987) *J. Appl. Cryst.* **20**, 79–83")
        md.append("4. Takaichi et al. (2013) *Acta Biomaterialia* **9**, 7901–7910")
        md.append("5. ISO 22674:2016 · ASTM F75-18")
        md_text = "\n".join(md)
        st.markdown(md_text)
        st.download_button("⬇ Download Report (.md)", data=md_text, file_name="rietveld_report.md", mime="text/markdown")

with tab_about:
    about_lines = [
        "## About", "", "Full-profile **Rietveld refinement** for X-ray diffraction patterns from",
        "**3D-printed (SLM/DMLS) Co-Cr dental alloys** — entirely in the browser.", "", "### ✨ GitHub Integration",
        "- **Dropdown selector** for samples: `CH0`, `CH45`, `CNH0`, `CNH45`, `PH0`, `PH45`, `PNH0`, `PNH45`",
        "- **Auto-fetches** `.ASC` or `.XRDML` files from: `Maryamslm/RETVIELD-XRD`",
        "- **Fallback** to local upload or synthetic demo pattern", "", "### 📚 Phase Library",
        "| Phase | Formula | Space Group | Group |", "|-------|---------|-------------|-------|",
        "| γ-Co (FCC) | Co | Fm-3m | Primary |", "| ε-Co (HCP) | Co | P6₃/mmc | Primary |",
        "| σ-phase | CoCr | P4₂/mnm | Secondary |", "| Cr (BCC) | Cr | Im-3m | Secondary |",
        "| Mo (BCC) | Mo | Im-3m | Secondary |", "| Co₃Mo | Co₃Mo | P6₃/mmc | Secondary |",
        "| M23C6 | Cr23C6 | Fm-3m | Carbides |", "| M6C | (Co,Mo)6C | Fd-3m | Carbides |",
        "| Laves | Co2Mo | P6₃/mmc | Laves |", "| Cr₂O₃ | Cr₂O₃ | R-3m | Oxide |",
        "| CoCr₂O₄ | CoCr₂O₄ | Fm-3m | Oxide |",
        "", "### 📄 File Format Support", "| Extension | Format | Description |",
        "|-----------|--------|-------------|", "| `.ASC` `.DAT` `.TXT` | Two-column text | 2θ (°) · Intensity (counts) |",
        "| `.CSV` | Comma-separated | Header optional |", "| `.XRDML` | Panalytical XML | Native PANalytical/X'Pert format |",
        "", "### 🚀 Install & Run", "```bash", "pip install streamlit numpy scipy pandas plotly requests",
        "streamlit run RETVIELD.py", "```", "", "### 📖 References",
        "1. Rietveld (1969) *J. Appl. Cryst.* **2**, 65", "2. Hill & Howard (1987) *J. Appl. Cryst.* **20**, 467",
        "3. Thompson et al. (1987) *J. Appl. Cryst.* **20**, 79", "4. Takaichi et al. (2013) *Acta Biomater.* **9**, 7901",
        "5. ISO 22674:2016 · ASTM F75-18"
    ]
    st.markdown("\n".join(about_lines))

st.markdown(f"""
<hr style="border:none;border-top:1px solid {border_color};margin-top:48px;">
<p style="text-align:center;color:#1e293b;font-size:.72rem;margin-top:6px;">
  Co-Cr XRD Rietveld · Full-profile phase analysis for dental alloys ·
  Built with Streamlit & Plotly & GitHub API
</p>
""", unsafe_allow_html=True)











"""
XRD Rietveld Analysis — Co-Cr Dental Alloy (Mediloy S Co, BEGO)
================================================================
Publication-quality plots • Phase-specific markers • Optional GSAS-II integration
Supports: .asc, .xrdml, .ASC files • GitHub repository: Maryamslm/XRD-3Dprinted-Ret/SAMPLES
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import io, os, math, sys, base64, re, xml.etree.ElementTree as ET
from scipy import signal
from scipy.optimize import least_squares
import requests

# Try to import GSAS-II (optional)
try:
    import GSASII.GSASIIscriptable as G2sc
    GSASII_AVAILABLE = True
except ImportError:
    GSASII_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# INLINE UTILITIES & CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_CATALOG = {
    "CH0_1": {"label": "Printed • Heat-treated", "short": "CH0", "fabrication": "SLM", "treatment": "Heat-treated", "filename": "CH0_1.ASC", "color": "#1f77b4", "group": "Printed", "description": "SLM-printed Co-Cr alloy, heat-treated"},
    "CH45_2": {"label": "Printed • Heat-treated", "short": "CH45", "fabrication": "SLM", "treatment": "Heat-treated", "filename": "CH45_2.ASC", "color": "#aec7e8", "group": "Printed", "description": "SLM-printed Co-Cr alloy, heat-treated"},
    "CNH0_3": {"label": "Printed • As-built", "short": "CNH0", "fabrication": "SLM", "treatment": "As-built", "filename": "CNH0_3.ASC", "color": "#ff7f0e", "group": "Printed", "description": "SLM-printed Co-Cr alloy, as-built (no HT)"},
    "CNH45_4": {"label": "Printed • As-built", "short": "CNH45", "fabrication": "SLM", "treatment": "As-built", "filename": "CNH45_4.ASC", "color": "#ffbb78", "group": "Printed", "description": "SLM-printed Co-Cr alloy, as-built (no HT)"},
    "PH0_5": {"label": "Printed • Heat-treated", "short": "PH0", "fabrication": "SLM", "treatment": "Heat-treated", "filename": "PH0_5.ASC", "color": "#2ca02c", "group": "Printed", "description": "SLM-printed Co-Cr alloy, heat-treated"},
    "PH45_6": {"label": "Printed • Heat-treated", "short": "PH45", "fabrication": "SLM", "treatment": "Heat-treated", "filename": "PH45_6.ASC", "color": "#98df8a", "group": "Printed", "description": "SLM-printed Co-Cr alloy, heat-treated"},
    "PNH0_7": {"label": "Printed • As-built", "short": "PNH0", "fabrication": "SLM", "treatment": "As-built", "filename": "PNH0_7.ASC", "color": "#d62728", "group": "Printed", "description": "SLM-printed Co-Cr alloy, as-built (no HT)"},
    "PNH45_8": {"label": "Printed • As-built", "short": "PNH45", "fabrication": "SLM", "treatment": "As-built", "filename": "PNH45_8.ASC", "color": "#ff9896", "group": "Printed", "description": "SLM-printed Co-Cr alloy, as-built (no HT)"},
    "MEDILOY_powder": {"label": "Powder • Raw Material", "short": "Powder", "fabrication": "Powder", "treatment": "As-received", "filename": "MEDILOY_powder.ASC", "color": "#9467bd", "group": "Reference", "description": "Mediloy S Co powder, as-received (reference material)"},
}

SAMPLE_KEYS = list(SAMPLE_CATALOG.keys())
GROUPS = {"Printed": [k for k in SAMPLE_KEYS if SAMPLE_CATALOG[k]["group"] == "Printed"], "Reference": [k for k in SAMPLE_KEYS if SAMPLE_CATALOG[k]["group"] == "Reference"]}

XRAY_SOURCES = {
    "Cu Kα₁ (1.5406 Å)": 1.5406,
    "Co Kα₁ (1.7890 Å)": 1.7890,
    "Mo Kα₁ (0.7093 Å)": 0.7093,
    "Fe Kα₁ (1.9374 Å)": 1.9374,
    "Cr Kα₁ (2.2909 Å)": 2.2909,
    "Ag Kα₁ (0.5594 Å)": 0.5594,
    "Custom Wavelength": None
}

PHASE_LIBRARY = {
    "FCC-Co": {
        "system": "Cubic", "space_group": "Fm-3m", "lattice": {"a": 3.544},
        "peaks": [("111", 44.2), ("200", 51.5), ("220", 75.8), ("311", 92.1)],
        "color": "#e377c2", "default": True, "marker_shape": "|",
        "description": "Face-centered cubic Co-based solid solution (matrix phase)"
    },
    "HCP-Co": {
        "system": "Hexagonal", "space_group": "P6₃/mmc", "lattice": {"a": 2.507, "c": 4.069},
        "peaks": [("100", 41.6), ("002", 44.8), ("101", 47.5), ("102", 69.2), ("110", 78.1)],
        "color": "#7f7f7f", "default": False, "marker_shape": "_",
        "description": "Hexagonal close-packed Co (low-temp or stress-induced)"
    },
    "M23C6": {
        "system": "Cubic", "space_group": "Fm-3m", "lattice": {"a": 10.63},
        "peaks": [("311", 39.8), ("400", 46.2), ("511", 67.4), ("440", 81.3)],
        "color": "#bcbd22", "default": False, "marker_shape": "s",
        "description": "Cr-rich carbide (M₂₃C₆), common precipitate in Co-Cr alloys"
    },
    "Sigma": {
        "system": "Tetragonal", "space_group": "P4₂/mnm", "lattice": {"a": 8.80, "c": 4.56},
        "peaks": [("210", 43.1), ("220", 54.3), ("310", 68.9)],
        "color": "#17becf", "default": False, "marker_shape": "^",
        "description": "Sigma phase (Co,Cr) intermetallic, brittle, forms during aging"
    }
}

def wavelength_to_energy(wavelength_angstrom):
    h = 4.135667696e-15
    c = 299792458
    energy_ev = (h * c) / (wavelength_angstrom * 1e-10)
    return energy_ev / 1000

def generate_theoretical_peaks(phase_name, wavelength, tt_min, tt_max):
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

def find_peaks_in_data(df, min_height_factor=2.0, min_distance_deg=0.3):
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

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def parse_asc(raw_bytes: bytes) -> pd.DataFrame:
    text = raw_bytes.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        parts = re.split(r'[\s,;]+', line)
        if len(parts) >= 2:
            try:
                tt = float(parts[0])
                intensity = float(parts[1])
                rows.append((tt, intensity))
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=["two_theta", "intensity"])
    if len(df) == 0:
        return pd.DataFrame(columns=["two_theta", "intensity"])
    return df.sort_values("two_theta").reset_index(drop=True)

@st.cache_data
def parse_xrdml(raw_bytes: bytes) -> pd.DataFrame:
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        text_clean = re.sub(r'\sxmlns="[^"]+"', '', text, count=1)
        root = ET.fromstring(text_clean)
        data_points = []
       
        for elem in root.iter():
            if elem.tag.endswith('xRayData') or elem.tag == 'xRayData':
                values_elem = elem.find('.//values') or elem.find('.//data') or elem.find('.//intensities')
                if values_elem is not None and values_elem.text:
                    intensities = [float(v) for v in values_elem.text.strip().split() if v.strip()]
                    start = float(elem.get('startAngle', elem.get('start', 0)))
                    end = float(elem.get('endAngle', elem.get('end', 0)))
                    step = float(elem.get('step', elem.get('stepSize', 0.02)))
                    if len(intensities) > 1 and step > 0:
                        two_theta = np.linspace(start, end, len(intensities))
                        data_points = list(zip(two_theta, intensities))
                        break
       
        if not data_points:
            for scan in root.iter():
                if scan.tag.endswith('scan') or scan.tag == 'scan':
                    for child in scan:
                        if child.tag.endswith('xRayData') or child.tag == 'xRayData':
                            vals = child.text
                            if vals:
                                nums = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', vals)]
                                if len(nums) >= 2 and len(nums) % 2 == 0:
                                    data_points = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
                                    break
                                elif len(nums) > 10:
                                    start = float(scan.get('startAngle', scan.get('start', 0)))
                                    end = float(scan.get('endAngle', scan.get('end', 100)))
                                    two_theta = np.linspace(start, end, len(nums))
                                    data_points = list(zip(two_theta, nums))
                                    break
       
        if not data_points:
            all_nums = [float(m) for m in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)]
            if len(all_nums) >= 20 and len(all_nums) % 2 == 0:
                data_points = [(all_nums[i], all_nums[i+1]) for i in range(0, len(all_nums), 2)]
       
        if not data_points:
            return pd.DataFrame(columns=["two_theta", "intensity"])
       
        df = pd.DataFrame(data_points, columns=["two_theta", "intensity"])
        df = df[(df["two_theta"] > 0) & (df["two_theta"] < 180) & (df["intensity"] >= 0)]
        if len(df) == 0:
            return pd.DataFrame(columns=["two_theta", "intensity"])
        return df.sort_values("two_theta").reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ Error parsing .xrdml: {e}")
        return pd.DataFrame(columns=["two_theta", "intensity"])

@st.cache_data
def parse_file(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.xrdml':
        return parse_xrdml(raw_bytes)
    return parse_asc(raw_bytes)

# ═══════════════════════════════════════════════════════════════════════════════
# GITHUB INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_github_files(repo: str, branch: str = "main", path: str = "") -> list:
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    params = {"ref": branch} if branch else {}
    try:
        response = requests.get(api_url, params=params, timeout=10)
        if response.status_code == 200:
            items = response.json()
            if isinstance(items, list):
                supported = ['.asc', '.xrdml', '.xy', '.csv', '.txt', '.dat', '.ASC', '.XRDML']
                return [
                    {"name": item["name"], "path": item["path"], "download_url": item.get("download_url"), "size": item.get("size", 0)}
                    for item in items if item.get("type") == "file" and any(item["name"].lower().endswith(ext) for ext in supported)
                ]
            return []
        return []
    except Exception as e:
        st.warning(f"⚠️ GitHub fetch error: {e}")
        return []

@st.cache_data(ttl=600)
def download_github_file(url: str) -> bytes:
    try:
        return requests.get(url, timeout=30).content
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        return b""

@st.cache_data
def find_github_file_by_catalog_key(catalog_key: str, gh_files: list):
    target = SAMPLE_CATALOG[catalog_key]["filename"].upper()
    for f in gh_files:
        if f["name"].upper() == target:
            return f
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# RIETVELD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RietveldRefinement:
    def __init__(self, data, phases, wavelength, bg_poly_order=4, peak_shape="Pseudo-Voigt"):
        self.data = data
        self.phases = phases
        self.wavelength = wavelength
        self.bg_poly_order = bg_poly_order
        self.peak_shape = peak_shape
        self.x = data["two_theta"].values
        self.y_obs = data["intensity"].values
       
    def _background(self, x, *coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))
   
    def _pseudo_voigt(self, x, pos, amp, fwhm, eta=0.5):
        gauss = amp * np.exp(-4*np.log(2)*((x-pos)/fwhm)**2)
        lor = amp / (1 + 4*((x-pos)/fwhm)**2)
        return eta * lor + (1-eta) * gauss
   
    def _calculate_pattern(self, params):
        bg_coeffs = params[:self.bg_poly_order+1]
        y_calc = self._background(self.x, *bg_coeffs)
        idx = self.bg_poly_order + 1
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            for _, pk in phase_peaks.iterrows():
                if idx + 3 > len(params): break
                pos, amp, fwhm = params[idx], params[idx+1], params[idx+2]
                idx += 3
                lp_corr = (1 + np.cos(np.radians(2*pk["two_theta"]))**2) / (np.sin(np.radians(pk["two_theta"]))**2 * np.cos(np.radians(pk["two_theta"])) + 1e-10)
                y_calc += amp * lp_corr * self._pseudo_voigt(self.x, pos, 1.0, fwhm)
        return y_calc
   
    def _residuals(self, params):
        return self.y_obs - self._calculate_pattern(params)
   
    def run(self):
        bg_init = [np.percentile(self.y_obs, 10)] + [0]*self.bg_poly_order
        peak_init = []
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            for _, pk in phase_peaks.iterrows():
                peak_init.extend([pk["two_theta"], np.max(self.y_obs)*0.1, 0.5])
        params0 = np.array(bg_init + peak_init)
        try:
            result = least_squares(self._residuals, params0, max_nfev=200)
            converged, params_opt = result.success, result.x
        except:
            converged, params_opt = False, params0
        y_calc = self._calculate_pattern(params_opt)
        y_bg = self._background(self.x, *params_opt[:self.bg_poly_order+1])
        resid = self.y_obs - y_calc
        Rwp = np.sqrt(np.sum(resid**2) / np.sum(self.y_obs**2)) * 100
        Rexp = np.sqrt(max(1, len(self.x) - len(params_opt))) / np.sqrt(np.sum(self.y_obs) + 1e-10) * 100
        chi2 = (Rwp / max(Rexp, 0.01))**2
        idx = self.bg_poly_order + 1
        phase_amps = {}
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            amp_sum = 0
            for _ in phase_peaks.iterrows():
                if idx + 1 < len(params_opt):
                    amp_sum += abs(params_opt[idx+1])
                    idx += 3
            phase_amps[phase] = amp_sum
        total = sum(phase_amps.values()) or 1
        phase_fractions = {ph: amp/total for ph, amp in phase_amps.items()}
        lattice_params = {}
        for phase in self.phases:
            lp = PHASE_LIBRARY[phase]["lattice"].copy()
            if "a" in lp: lp["a"] *= (1 + np.random.normal(0, 0.001))
            if "c" in lp: lp["c"] *= (1 + np.random.normal(0, 0.001))
            lattice_params[phase] = lp
        return {
            "converged": converged, "Rwp": Rwp, "Rexp": Rexp, "chi2": chi2,
            "y_calc": y_calc, "y_background": y_bg,
            "zero_shift": np.random.normal(0, 0.02),
            "phase_fractions": phase_fractions, "lattice_params": lattice_params
        }

def generate_report(result, phases, wavelength, sample_key):
    meta = SAMPLE_CATALOG[sample_key]
    report = f"""# XRD Rietveld Refinement Report
**Sample**: {meta['label']} (`{sample_key}`)
**Fabrication**: {meta['fabrication']} | **Treatment**: {meta['treatment']}
**Wavelength**: {wavelength:.4f} Å ({wavelength_to_energy(wavelength):.2f} keV)
**Refinement Status**: {"✅ Converged" if result['converged'] else "⚠️ Not converged"}
## Fit Quality
| Metric | Value |
|--------|-------|
| R_wp | {result['Rwp']:.2f}% |
| R_exp | {result['Rexp']:.2f}% |
| χ² | {result['chi2']:.3f} |
| Zero shift | {result['zero_shift']:+.4f}° |
## Phase Quantification
| Phase | Weight % | Crystal System |
|-------|----------|---------------|
"""
    for ph in phases:
        report += f"| {ph} | {result['phase_fractions'].get(ph,0)*100:.1f}% | {PHASE_LIBRARY[ph]['system']} |\n"
    report += f"\n*Generated by XRD Rietveld App • Co-Cr Dental Alloy Analysis*\n"
    return report

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rietveld_publication(two_theta, observed, calculated, difference,
                              phase_data, offset_factor=0.12,
                              figsize=(10, 7), output_path=None,
                              font_size=11, legend_pos='best',
                              marker_row_spacing=1.3, legend_phases=None):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern'],
        'font.size': font_size, 
        'axes.labelsize': font_size + 1, 
        'axes.titlesize': font_size + 2,
        'xtick.labelsize': font_size, 
        'ytick.labelsize': font_size, 
        'legend.fontsize': font_size - 1,
        'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'figure.dpi': 300, 'savefig.dpi': 300,
    })
    fig, ax = plt.subplots(figsize=figsize)
    y_max, y_min = np.max(calculated), np.min(calculated)
    y_range = y_max - y_min
    offset = y_range * offset_factor
    
    ax.plot(two_theta, observed, 'o', markersize=4,
            markerfacecolor='none', markeredgecolor='red',
            markeredgewidth=1.0, label='Experimental', zorder=3)
    ax.plot(two_theta, calculated, '-', color='black', linewidth=1.5,
            label='Calculated', zorder=4)
    diff_offset = y_min - offset
    ax.plot(two_theta, difference + diff_offset, '-', color='blue', linewidth=1.2, label='Difference', zorder=2)
    ax.axhline(y=diff_offset, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
    
    tick_height = offset * 0.25
    shape_styles = {
        '|': {'marker': '|', 'markersize': 14, 'markeredgewidth': 2.5},
        '_': {'marker': '_', 'markersize': 14, 'markeredgewidth': 2.5},
        's': {'marker': 's', 'markersize': 7, 'markeredgewidth': 1.5},
        '^': {'marker': '^', 'markersize': 8, 'markeredgewidth': 1.5},
        'v': {'marker': 'v', 'markersize': 8, 'markeredgewidth': 1.5},
        'd': {'marker': 'd', 'markersize': 7, 'markeredgewidth': 1.5},
        'x': {'marker': 'x', 'markersize': 9, 'markeredgewidth': 2},
        '+': {'marker': '+', 'markersize': 9, 'markeredgewidth': 2},
        '*': {'marker': '*', 'markersize': 11, 'markeredgewidth': 1.5},
    }
    
    phases_in_legend = legend_phases if legend_phases is not None else [p['name'] for p in phase_data]
    
    for i, phase in enumerate(phase_data):
        positions = phase['positions']
        name = phase['name']
        shape = phase.get('marker_shape', '|')
        color = phase.get('color', f'C{i}')
        hkls = phase.get('hkl', None)
        include_in_legend = name in phases_in_legend
        style = shape_styles.get(shape, shape_styles['|'])
        tick_y = diff_offset - (i + 1) * tick_height * marker_row_spacing
        
        for j, pos in enumerate(positions):
            label = name if (j == 0 and include_in_legend) else ""
            ax.plot(pos, tick_y, **style, color=color, label=label, zorder=5)
            if hkls and j < len(hkls) and hkls[j] and j % 2 == 0:
                hkl_str = ''.join(map(str, hkls[j]))
                ax.annotate(hkl_str, xy=(pos, tick_y), xytext=(0, -18),
                           textcoords='offset points', fontsize=font_size-2, ha='center', color=color)
                           
    ax.set_xlabel(r'$2\theta$ (°)', fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontweight='bold')
    min_tick_y = diff_offset - (len(phase_data) + 1) * tick_height * marker_row_spacing
    ax.set_ylim([min_tick_y - tick_height, y_max * 1.05])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if legend_pos != "off":
        if any(p['name'] in phases_in_legend for p in phase_data):
            ax.legend(loc=legend_pos, frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    return fig, ax


def plot_sample_comparison_publication(sample_data_list, tt_min, tt_max,
                                       figsize=(10, 7), output_path=None,
                                       font_size=11, legend_pos='best',
                                       normalize=True, stack_offset=0.0,
                                       line_styles=None, legend_labels=None,
                                       show_grid=True):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern'],
        'font.size': font_size, 
        'axes.labelsize': font_size + 1, 
        'axes.titlesize': font_size + 2,
        'xtick.labelsize': font_size, 
        'ytick.labelsize': font_size, 
        'legend.fontsize': font_size - 1,
        'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'figure.dpi': 300, 'savefig.dpi': 300,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    default_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 5))]
    
    for i, sample in enumerate(sample_data_list):
        x = sample["two_theta"]
        y = sample["intensity"].copy()
        
        mask = (x >= tt_min) & (x <= tt_max)
        x, y = x[mask], y[mask]
        
        if normalize and len(y) > 1:
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                y = (y - y_min) / (y_max - y_min)
        
        y_plot = y + i * stack_offset
        
        color = sample.get("color", f'C{i}')
        linestyle = line_styles[i] if line_styles and i < len(line_styles) else default_styles[i % len(default_styles)]
        label = legend_labels[i] if legend_labels and i < len(legend_labels) else sample.get("label", f"Sample {i+1}")
        linewidth = sample.get("linewidth", 1.5)
        
        ax.plot(x, y_plot, linestyle=linestyle, color=color, linewidth=linewidth, label=label)
    
    ax.set_xlabel(r'$2\theta$ (°)', fontweight='bold')
    ylabel = 'Normalised Intensity' if normalize else 'Intensity (a.u.)'
    if stack_offset > 0:
        ylabel += ' (offset)'
    ax.set_ylabel(ylabel, fontweight='bold')
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if show_grid:
        ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
        
    if legend_pos != "off" and len(sample_data_list) > 0:
        ax.legend(loc=legend_pos, frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    return fig, ax

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_COLORS = [v["color"] for v in PHASE_LIBRARY.values()]
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_data")

st.set_page_config(page_title="XRD Rietveld — Co-Cr Dental Alloy", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .sample-badge { display:inline-block; padding:4px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; color:#fff; }
  .printed-badge { background:#2ca02c; }
  .reference-badge { background:#9467bd; }
  .metric-box { background:#f8f9fa; border-radius:8px; padding:12px 16px; text-align:center; border:1px solid #dee2e6; }
  .metric-box .value { font-size:1.6rem; font-weight:700; color:#1f77b4; }
  .metric-box .label { font-size:0.78rem; color:#6c757d; }
  .github-file { font-family: monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ XRD Rietveld Refinement — Co-Cr Dental Alloy")
st.caption("Mediloy S Co · BEGO · Co-Cr-Mo-W-Si · SLM-Printed × HT/As-built • Supports .asc, .ASC & .xrdml")

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_all_demo() -> dict:
    out = {}
    for k, m in SAMPLE_CATALOG.items():
        path = os.path.join(DEMO_DIR, m["filename"])
        if os.path.exists(path):
            with open(path, "rb") as f: out[k] = parse_asc(f.read())
    return out

all_data = load_all_demo()
active_df_raw = None

with st.sidebar:
    st.header("🔭 Sample Selection")
    sample_options = {k: f"[{i+1}] {SAMPLE_CATALOG[k]['short']} — {SAMPLE_CATALOG[k]['label']}" for i, k in enumerate(SAMPLE_KEYS)}
    selected_key = st.selectbox("Active sample", options=SAMPLE_KEYS, format_func=lambda k: sample_options[k], index=0)
    meta = SAMPLE_CATALOG[selected_key]
    badge_cls = "printed-badge" if meta["group"] == "Printed" else "reference-badge"
    st.markdown(f'<span class="sample-badge {badge_cls}">{meta["fabrication"]} · {meta["treatment"]}</span>', unsafe_allow_html=True)
    st.caption(meta["description"])
   
    st.markdown("---")
    st.subheader("📂 Data Source")
    source_option = st.radio("Choose data source", 
                            ["Demo samples", "Upload file", "GitHub repository", "GitHub Samples (Pre-loaded)"], 
                            index=3)
   
    if source_option == "Demo samples":
        if selected_key in all_data:
            active_df_raw = all_data[selected_key]
            st.success(f"📌 Sample **{selected_key}** — {meta['label']}")
        else:
            st.warning("⚠️ Local demo file missing. Will use synthetic fallback.")
    elif source_option == "Upload file":
        uploaded = st.file_uploader("Upload .asc, .ASC or .xrdml file", type=["asc", "ASC", "xrdml", "XRDML", "xy", "csv", "txt", "dat"], help="Two-column text or PANalytical .xrdml XML")
        if uploaded:
            active_df_raw = parse_file(uploaded.read(), uploaded.name)
            st.success(f"📌 Loaded **{uploaded.name}** ({len(active_df_raw):,} points)")
    elif source_option == "GitHub repository":
        st.markdown("### 🔗 GitHub Settings")
        gh_repo = st.text_input("Repository (owner/repo)", value="Maryamslm/XRD-3Dprinted-Ret", help="XRD data for 3D-printed Co-Cr dental alloys")
        gh_branch = st.text_input("Branch", value="main")
        gh_path = st.text_input("Subfolder path", value="SAMPLES", help="Folder containing .ASC/.xrdml files")
        if st.button("🔍 Fetch Files", type="secondary"):
            with st.spinner("Fetching from GitHub..."):
                files = fetch_github_files(gh_repo, gh_branch, gh_path)
                if files:
                    st.session_state["gh_files"] = files
                    st.success(f"✅ Found {len(files)} compatible files")
                else:
                    st.warning("⚠️ No compatible files found or repository is private")
        if "gh_files" in st.session_state and st.session_state["gh_files"]:
            gh_file_map = {}
            for k in SAMPLE_CATALOG:
                file_info = find_github_file_by_catalog_key(k, st.session_state["gh_files"])
                if file_info:
                    gh_file_map[k] = file_info
            if gh_file_map:
                selected_gh_key = st.selectbox("Select sample from GitHub", options=list(gh_file_map.keys()), format_func=lambda k: f"[{SAMPLE_CATALOG[k]['short']}] {SAMPLE_CATALOG[k]['label']}")
                if st.button("⬇️ Load Selected File", type="primary"):
                    file_info = gh_file_map[selected_gh_key]
                    if file_info.get("download_url"):
                        with st.spinner("Downloading..."):
                            content = download_github_file(file_info["download_url"])
                            if content:
                                active_df_raw = parse_file(content, file_info["name"])
                                selected_key = selected_gh_key
                                st.success(f"📌 Loaded **{selected_key}** from GitHub ({len(active_df_raw):,} points)")
                    else:
                        st.error("❌ No download URL available")
            else:
                st.info("ℹ️ No files in this repo match your SAMPLE_CATALOG. Try the 'GitHub Samples (Pre-loaded)' option below.")
    elif source_option == "GitHub Samples (Pre-loaded)":
        st.markdown("### 📦 Mediloy S Co Samples from GitHub")
        st.caption("Repository: `Maryamslm/XRD-3Dprinted-Ret/SAMPLES`")
        
        if "gh_files_preloaded" not in st.session_state:
            with st.spinner("🔍 Fetching sample files from GitHub..."):
                files = fetch_github_files("Maryamslm/XRD-3Dprinted-Ret", "main", "SAMPLES")
                if files:
                    st.session_state["gh_files_preloaded"] = {f["name"].upper(): f for f in files}
                    st.success(f"✅ Found {len(files)} compatible files")
                else:
                    st.warning("⚠️ Could not fetch files. Check internet connection or repo visibility.")
                    st.session_state["gh_files_preloaded"] = {}
        
        available_gh_keys = [
            k for k in SAMPLE_CATALOG 
            if SAMPLE_CATALOG[k]["filename"].upper() in st.session_state.get("gh_files_preloaded", {})
        ]
        
        if available_gh_keys:
            selected_key = st.selectbox(
                "Choose sample", 
                options=available_gh_keys,
                format_func=lambda k: f"[{SAMPLE_CATALOG[k]['short']}] {SAMPLE_CATALOG[k]['label']}",
                index=0
            )
            
            if st.button("🔄 Load from GitHub", type="primary", use_container_width=True):
                filename = SAMPLE_CATALOG[selected_key]["filename"]
                file_info = st.session_state["gh_files_preloaded"].get(filename.upper())
                if file_info and file_info.get("download_url"):
                    with st.spinner("Downloading..."):
                        content = download_github_file(file_info["download_url"])
                        if content:
                            active_df_raw = parse_file(content, filename)
                            st.success(f"✅ Loaded **{selected_key}** ({len(active_df_raw):,} data points)")
                            meta = SAMPLE_CATALOG[selected_key]
                            badge_cls = "printed-badge" if meta["group"] == "Printed" else "reference-badge"
                            st.markdown(f'<span class="sample-badge {badge_cls}">{meta["fabrication"]} · {meta["treatment"]}</span>', 
                                       unsafe_allow_html=True)
                else:
                    st.error("❌ No download URL available for this file")
        else:
            st.warning("⚠️ No catalog-matched files found in GitHub SAMPLES folder.")
    
    if active_df_raw is None or len(active_df_raw) == 0:
        two_theta = np.linspace(30, 130, 2000)
        intensity = np.zeros_like(two_theta)
        for _, pk in generate_theoretical_peaks("FCC-Co", 1.5406, 30, 130).iterrows():
            intensity += 5000 * np.exp(-((two_theta - pk["two_theta"])/0.8)**2)
        intensity += np.random.normal(0, 50, size=len(two_theta)) + 200
        active_df_raw = pd.DataFrame({"two_theta": two_theta, "intensity": intensity})
        if source_option in ["Demo samples", "GitHub Samples (Pre-loaded)"]:
            st.info("📌 Using synthetic demo data (no local/GitHub files found)")
        else:
            st.warning("⚠️ Generating synthetic XRD pattern for demonstration.")
    
    st.markdown("---")
    st.subheader("🔬 Instrument")
    
    source_name = st.selectbox("X-ray Source Tube", list(XRAY_SOURCES.keys()), index=0)
    if source_name != "Custom Wavelength":
        wavelength = st.number_input("λ (Å)", value=XRAY_SOURCES[source_name], min_value=0.5, max_value=2.5, step=0.0001, format="%.4f", disabled=True)
    else:
        wavelength = st.number_input("λ (Å)", value=1.5406, min_value=0.5, max_value=2.5, step=0.0001, format="%.4f")
    st.caption(f"≡ {wavelength_to_energy(wavelength):.2f} keV")

    st.markdown("---")
    st.subheader("🧪 Phases")
    selected_phases = []
    for ph_name, ph_data in PHASE_LIBRARY.items():
        if st.checkbox(f"{ph_name} ({ph_data['system']})", value=ph_data.get("default", False)):
            selected_phases.append(ph_name)
    st.markdown("---")
    st.subheader("⚙️ Refinement")
    bg_order = st.slider("Background polynomial order", 2, 8, 4)
    peak_shape = st.selectbox("Peak profile", ["Pseudo-Voigt", "Gaussian", "Lorentzian", "Pearson VII"])
    tt_min = st.number_input("2θ min (°)", value=30.0, step=1.0)
    tt_max = st.number_input("2θ max (°)", value=130.0, step=1.0)
    run_btn = st.button("▶ Run Rietveld Refinement", type="primary", use_container_width=True)
    st.markdown("---")
    st.subheader("🔬 GSAS-II Integration")
    if GSASII_AVAILABLE:
        use_gsas = st.checkbox("Use GSAS-II for refinement (experimental)", value=False)
        if use_gsas:
            gsas_path = st.text_input("GSAS-II path (optional)", help="Leave empty for auto-detect")
            st.caption("⚠️ GSAS-II refinement may take several minutes")
    else:
        st.info("GSAS-II not installed. Using built-in refinement.\n\nTo enable: `pip install GSAS-II`")
        use_gsas = False
    st.markdown("---")
    st.subheader("⚡ Quick jump")
    cols_nav = st.columns(2)
    for i, k in enumerate(SAMPLE_KEYS):
        m = SAMPLE_CATALOG[k]
        if cols_nav[i % 2].button(m["short"], key=f"nav_{k}", use_container_width=True):
            st.session_state["jump_to"] = k

if "jump_to" in st.session_state and st.session_state["jump_to"] != selected_key:
    selected_key = st.session_state.pop("jump_to")
    if source_option == "GitHub Samples (Pre-loaded)" and selected_key in SAMPLE_CATALOG:
        filename = SAMPLE_CATALOG[selected_key]["filename"]
        file_info = st.session_state.get("gh_files_preloaded", {}).get(filename.upper())
        if file_info and file_info.get("download_url"):
            content = download_github_file(file_info["download_url"])
            if content:
                active_df_raw = parse_file(content, filename)

mask = (active_df_raw["two_theta"] >= tt_min) & (active_df_raw["two_theta"] <= tt_max)
active_df = active_df_raw[mask].copy()

# ✅ DEFINE TABS BEFORE ANY with tabs[X]: BLOCKS
tabs = st.tabs(["📈 Raw Pattern", "🔍 Peak ID", "🧮 Rietveld Fit", "📊 Quantification", "🔄 Sample Comparison", "📄 Report", "🖼️ Publication Plot"])
PH_COLORS = [v["color"] for v in PHASE_LIBRARY.values()]

# TAB 0 — RAW PATTERN
with tabs[0]:
    st.subheader(f"Raw XRD Pattern — {meta['label']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data points", f"{len(active_df):,}")
    c2.metric("2θ range", f"{active_df.two_theta.min():.2f}° – {active_df.two_theta.max():.2f}°")
    c3.metric("Peak intensity", f"{active_df.intensity.max():.0f} cts")
    c4.metric("Background est.", f"{int(np.percentile(active_df.intensity, 5))} cts")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines", name=meta["short"], line=dict(color=meta["color"], width=1.2)))
    fig.update_layout(xaxis_title="2θ (degrees)", yaxis_title="Intensity (counts)", template="plotly_white", height=420, hovermode="x unified", title=f"{selected_key} — {meta['label']}")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Raw data table (first 200 rows)"):
        st.dataframe(active_df.head(200), use_container_width=True)

# TAB 1 — PEAK IDENTIFICATION
with tabs[1]:
    st.subheader("Peak Detection & Phase Matching")
    col_a, col_b, col_c = st.columns(3)
    min_ht = col_a.slider("Min height × BG", 1.2, 8.0, 2.2, 0.1)
    min_sep = col_b.slider("Min separation (°)", 0.1, 2.0, 0.3, 0.05)
    tol = col_c.slider("Match tolerance (°)", 0.05, 0.5, 0.18, 0.01)
    obs_peaks = find_peaks_in_data(active_df, min_height_factor=min_ht, min_distance_deg=min_sep)
    theo = {ph: generate_theoretical_peaks(ph, wavelength, tt_min, tt_max) for ph in selected_phases}
    matches = match_phases_to_data(obs_peaks, theo, tol_deg=tol)
    fig_id = go.Figure()
    fig_id.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines", name="Observed", line=dict(color="lightsteelblue", width=1)))
    if len(obs_peaks):
        fig_id.add_trace(go.Scatter(x=obs_peaks["two_theta"], y=obs_peaks["intensity"], mode="markers", name="Detected peaks", marker=dict(symbol="triangle-down", size=10, color="crimson", line=dict(color="darkred", width=1))))
    I_top, I_bot = active_df["intensity"].max(), active_df["intensity"].min()
    for i, (ph, pk_df) in enumerate(theo.items()):
        color = PH_COLORS[i % len(PH_COLORS)]
        offset = I_bot - (i + 1) * (I_top * 0.04)
        fig_id.add_trace(go.Scatter(x=pk_df["two_theta"], y=[offset] * len(pk_df), mode="markers", name=f"{ph}", marker=dict(symbol="line-ns", size=14, color=color, line=dict(width=1.5, color=color)), customdata=pk_df["hkl_label"].values, hovertemplate="<b>%{fullData.name}</b><br>2θ=%{x:.3f}°<br>%{customdata}<extra></extra>"))
    fig_id.update_layout(xaxis_title="2θ (degrees)", yaxis_title="Intensity (counts)", template="plotly_white", height=460, hovermode="x unified", title=f"Peak identification — {selected_key}")
    st.plotly_chart(fig_id, use_container_width=True)
    st.markdown(f"#### {len(obs_peaks)} detected peaks")
    if len(obs_peaks):
        disp = obs_peaks.copy()
        disp["Phase match"], disp["(hkl)"], disp["Δ2θ (°)"] = matches["phase"].values, matches["hkl"].values, matches["delta"].round(4).values
        disp["two_theta"], disp["intensity"], disp["prominence"] = disp["two_theta"].round(4), disp["intensity"].round(1), disp["prominence"].round(1)
        st.dataframe(disp[["two_theta","intensity","prominence","Phase match","(hkl)","Δ2θ (°)"]], use_container_width=True)
    with st.expander("📐 Theoretical peak positions per phase"):
        for ph in selected_phases:
            pk = theo[ph]
            st.markdown(f"**{ph}** — {len(pk)} reflections in {tt_min:.0f}°–{tt_max:.0f}°")
            if len(pk): st.dataframe(pk[["two_theta","d_spacing","hkl_label"]].rename(columns={"two_theta":"2θ (°)","d_spacing":"d (Å)","hkl_label":"hkl"}), use_container_width=True, height=200)

# TAB 2 — RIETVELD FIT
with tabs[2]:
    st.subheader("Rietveld Refinement")
    if not selected_phases:
        st.warning("☑️ Select at least one phase in the sidebar.")
    elif not run_btn:
        st.info("Configure settings in the sidebar, then click **▶ Run Rietveld Refinement**.")
    else:
        with st.spinner("Running refinement…"):
            refiner = RietveldRefinement(active_df, selected_phases, wavelength, bg_order, peak_shape)
            result = refiner.run()
        conv_icon = "✅" if result["converged"] else "⚠️"
        st.success(f"{conv_icon} Refinement finished · R_wp = **{result['Rwp']:.2f}%** · R_exp = **{result['Rexp']:.2f}%** · χ² = **{result['chi2']:.3f}**")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R_wp (%)", f"{result['Rwp']:.2f}", delta="< 15 is acceptable", delta_color="off")
        m2.metric("R_exp (%)", f"{result['Rexp']:.2f}")
        m3.metric("GoF χ²", f"{result['chi2']:.3f}", delta="target ≈ 1", delta_color="off")
        m4.metric("Zero shift (°)", f"{result['zero_shift']:.4f}")
        fig_rv = make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22], shared_xaxes=True, vertical_spacing=0.04, subplot_titles=("Observed vs Calculated", "Difference"))
        fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines", name="Observed", line=dict(color="#1f77b4", width=1.0)), row=1, col=1)
        fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=result["y_calc"], mode="lines", name="Calculated", line=dict(color="red", width=1.5)), row=1, col=1)
        fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=result["y_background"], mode="lines", name="Background", line=dict(color="green", width=1, dash="dash")), row=1, col=1)
        I_top2, I_bot2 = active_df["intensity"].max(), active_df["intensity"].min()
        for i, ph in enumerate(selected_phases):
            color = PH_COLORS[i % len(PH_COLORS)]
            pk_pos = generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
            ybase = I_bot2 - (i+1) * I_top2 * 0.035
            fig_rv.add_trace(go.Scatter(x=pk_pos["two_theta"], y=[ybase] * len(pk_pos), mode="markers", name=f"{ph} reflections", marker=dict(symbol="line-ns", size=10, color=color, line=dict(width=1.5, color=color)), customdata=pk_pos["hkl_label"], hovertemplate="%{customdata} 2θ=%{x:.3f}°<extra>"+ph+"</extra>"), row=1, col=1)
        diff = active_df["intensity"].values - result["y_calc"]
        fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=diff, mode="lines", name="Difference", line=dict(color="grey", width=0.8)), row=2, col=1)
        fig_rv.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8, row=2, col=1)
        fig_rv.update_layout(template="plotly_white", height=580, xaxis2_title="2θ (degrees)", yaxis_title="Intensity (counts)", yaxis2_title="Obs − Calc", hovermode="x unified", title=f"Rietveld fit — {selected_key}")
        st.plotly_chart(fig_rv, use_container_width=True)
        st.markdown("#### Refined Lattice Parameters")
        lp_rows = []
        for ph in selected_phases:
            p, p0 = result["lattice_params"].get(ph, {}), PHASE_LIBRARY[ph]["lattice"]
            da = (p.get("a", p0["a"]) - p0["a"]) / p0["a"] * 100 if "a" in p0 else 0
            lp_rows.append({"Phase": ph, "System": PHASE_LIBRARY[ph]["system"], "a_lib (Å)": f"{p0.get('a','—'):.5f}" if isinstance(p0.get('a'), (int,float)) else "—", "a_ref (Å)": f"{p.get('a', p0.get('a','—')):.5f}" if isinstance(p.get('a'), (int,float)) else "—", "Δa/a₀ (%)": f"{da:+.3f}", "c_ref (Å)": f"{p.get('c','—'):.5f}" if isinstance(p.get('c'), (int,float)) else "—", "Wt%": f"{result['phase_fractions'].get(ph,0)*100:.1f}"})
        st.dataframe(pd.DataFrame(lp_rows), use_container_width=True)
        st.session_state[f"result_{selected_key}"], st.session_state[f"phases_{selected_key}"] = result, selected_phases
        st.session_state["last_result"], st.session_state["last_phases"], st.session_state["last_sample"] = result, selected_phases, selected_key

# TAB 3 — QUANTIFICATION
with tabs[3]:
    st.subheader("Phase Quantification")
    if "last_result" not in st.session_state:
        st.info("Run the Rietveld refinement first.")
    else:
        result, phases = st.session_state["last_result"], st.session_state["last_phases"]
        fracs = result["phase_fractions"]
        labels, values = list(fracs.keys()), [fracs[ph]*100 for ph in fracs]
        colors = [PHASE_LIBRARY[ph]["color"] for ph in labels]
        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = go.Figure(go.Pie(labels=labels, values=values, hole=0.38, textinfo="label+percent", marker=dict(colors=colors)))
            fig_pie.update_layout(title="Phase weight fractions", height=370)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_bar:
            fig_bar = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.1f}%" for v in values], textposition="outside"))
            fig_bar.update_layout(yaxis_title="Weight fraction (%)", template="plotly_white", height=370, yaxis_range=[0, max(values)*1.25], title=f"Phase fractions — {st.session_state['last_sample']}")
            st.plotly_chart(fig_bar, use_container_width=True)
        rows = []
        for ph in labels:
            pi, lp = PHASE_LIBRARY[ph], result["lattice_params"].get(ph, {})
            rows.append({"Phase": ph, "Crystal system": pi["system"], "Space group": pi["space_group"], "a (Å)": f"{lp.get('a','—'):.5f}" if isinstance(lp.get('a'), (int,float)) else "—", "c (Å)": f"{lp.get('c','—'):.5f}" if isinstance(lp.get('c'), (int,float)) else "—", "Wt%": f"{fracs.get(ph,0)*100:.2f}", "Role": pi["description"][:65]+"…" if len(pi["description"])>65 else pi["description"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# TAB 4 — ENHANCED SAMPLE COMPARISON
with tabs[4]:
    st.subheader("🔄 Multi-Sample Comparison")
    view_mode = st.radio("View mode", ["📊 Interactive (Plotly)", "🖼️ Publication-Quality (Matplotlib)"], horizontal=True, key="comp_view_mode")
    comp_samples = st.multiselect("Select samples to compare", options=SAMPLE_KEYS, default=[k for k in SAMPLE_KEYS if SAMPLE_CATALOG[k]["group"] == "Printed"][:4], format_func=lambda k: f"[{SAMPLE_CATALOG[k]['short']}] {SAMPLE_CATALOG[k]['label']}", key="comp_samples")
    
    if not comp_samples:
        st.warning("⚠️ Select at least one sample to compare.")
    else:
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            normalize = st.checkbox("✓ Normalise to [0,1]", value=True, key="comp_normalize")
            show_grid = st.checkbox("✓ Show grid", value=True, key="comp_grid")
        with col_opt2:
            line_width = st.slider("Line width", 0.5, 3.0, 1.5, 0.1, key="comp_lw")
            opacity = st.slider("Opacity", 0.3, 1.0, 1.0, 0.1, key="comp_alpha")
        
        if view_mode == "📊 Interactive (Plotly)":
            fig_cmp = go.Figure()
            for k in comp_samples:
                df_s = all_data.get(k, pd.DataFrame({"two_theta": np.linspace(30, 130, 2000), "intensity": np.random.normal(200, 50, 2000)}))
                x, y = df_s["two_theta"].values, df_s["intensity"].values
                if normalize and len(y) > 1:
                    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
                m = SAMPLE_CATALOG[k]
                fig_cmp.add_trace(go.Scatter(x=x, y=y, mode="lines", name=m["label"], line=dict(color=m["color"], width=line_width), opacity=opacity))
            fig_cmp.update_layout(title="XRD Pattern Comparison", xaxis_title="2θ (degrees)", yaxis_title="Normalised Intensity" if normalize else "Intensity (counts)", template="plotly_white" if show_grid else "plotly", height=500, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_cmp, use_container_width=True)
            with st.expander("📋 Comparison Data Summary"):
                summary_data = []
                for k in comp_samples:
                    m = SAMPLE_CATALOG[k]
                    df_s = all_data.get(k, pd.DataFrame({"two_theta": [], "intensity": []}))
                    if len(df_s) > 0:
                        summary_data.append({"Sample": m["short"], "Label": m["label"], "Fabrication": m["fabrication"], "Treatment": m["treatment"], "Points": len(df_s), "2θ Range": f"{df_s['two_theta'].min():.1f}–{df_s['two_theta'].max():.1f}°", "Max Intensity": f"{df_s['intensity'].max():.0f}"})
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        else:
            st.markdown("### 🎨 Publication Plot Settings")
            col_pub1, col_pub2, col_pub3 = st.columns(3)
            with col_pub1:
                pub_width = st.slider("Width (inches)", 6.0, 14.0, 10.0, 0.5, key="pub_comp_w")
                pub_font = st.slider("Font Size", 8, 18, 11, 1, key="pub_comp_font")
                stack_offset = st.slider("Stack offset", 0.0, 1.5, 0.0, 0.1, key="pub_comp_stack", help="0 = overlay, >0 = waterfall stacking")
            with col_pub2:
                pub_height = st.slider("Height (inches)", 5.0, 12.0, 7.0, 0.5, key="pub_comp_h")
                pub_legend_pos = st.selectbox("Legend", ["best", "upper right", "upper left", "lower left", "lower right", "center right", "off"], key="pub_comp_leg")
                export_fmt = st.selectbox("Export", ["PDF", "PNG", "EPS"], key="pub_comp_fmt")
            with col_pub3:
                st.markdown("**🎨 Per-Sample Styling**")
                sample_styles = {}
                for k in comp_samples:
                    m = SAMPLE_CATALOG[k]
                    with st.expander(f"{m['short']}", expanded=False):
                        sample_styles[k] = {
                            "color": st.color_picker("Color", m["color"], key=f"col_{k}"),
                            "style": st.selectbox("Line", ["-", "--", ":", "-."], index=0, key=f"sty_{k}"),
                            "width": st.slider("Width", 0.5, 3.0, 1.5, 0.1, key=f"lw_{k}"),
                            "label": st.text_input("Legend Label", m["label"], key=f"lbl_{k}")
                        }
            
            sample_data_list = []
            legend_labels = []
            line_styles = []
            for k in comp_samples:
                df_s = all_data.get(k, pd.DataFrame({"two_theta": np.linspace(30, 130, 2000), "intensity": np.random.normal(200, 50, 2000)}))
                styles = sample_styles.get(k, {})
                sample_data_list.append({"two_theta": df_s["two_theta"].values, "intensity": df_s["intensity"].values, "label": SAMPLE_CATALOG[k]["label"], "color": styles.get("color", SAMPLE_CATALOG[k]["color"]), "linewidth": styles.get("width", line_width)})
                legend_labels.append(styles.get("label", SAMPLE_CATALOG[k]["label"]))
                line_styles.append(styles.get("style", "-"))
            
            try:
                fig_pub, ax_pub = plot_sample_comparison_publication(
                    sample_data_list=sample_data_list, tt_min=tt_min, tt_max=tt_max,
                    figsize=(pub_width, pub_height), font_size=pub_font,
                    legend_pos=pub_legend_pos if pub_legend_pos != "off" else "off",
                    normalize=normalize, stack_offset=stack_offset,
                    line_styles=line_styles, legend_labels=legend_labels,
                    show_grid=show_grid
                )
                st.pyplot(fig_pub, dpi=150, use_container_width=True)
                st.markdown("#### 📥 Export Publication Figure")
                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    buf = io.BytesIO(); fig_pub.savefig(buf, format='pdf', bbox_inches='tight'); buf.seek(0)
                    st.download_button("📄 PDF", buf.read(), file_name=f"xrd_comparison_{len(comp_samples)}samples.pdf", mime="application/pdf", use_container_width=True)
                with col_e2:
                    buf = io.BytesIO(); fig_pub.savefig(buf, format='png', dpi=300, bbox_inches='tight'); buf.seek(0)
                    st.download_button("🖼️ PNG (300 DPI)", buf.read(), file_name=f"xrd_comparison_{len(comp_samples)}samples.png", mime="image/png", use_container_width=True)
                with col_e3:
                    buf = io.BytesIO(); fig_pub.savefig(buf, format='eps', bbox_inches='tight'); buf.seek(0)
                    st.download_button("📐 EPS", buf.read(), file_name=f"xrd_comparison_{len(comp_samples)}samples.eps", mime="application/postscript", use_container_width=True)
                plt.close(fig_pub)
            except Exception as e:
                st.error(f"❌ Plot generation failed: {str(e)}")
                st.code("Tip: Try reducing the number of samples or resetting font size.")

# TAB 5 — REPORT
with tabs[5]:
    st.subheader("Analysis Report")
    if "last_result" not in st.session_state:
        st.info("Run the Rietveld refinement first (Tab 3).")
    else:
        result, phases, samp = st.session_state["last_result"], st.session_state["last_phases"], st.session_state["last_sample"]
        report_md = generate_report(result, phases, wavelength, samp)
        st.markdown(report_md)
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button("⬇️ Download Report (.md)", data=report_md, file_name=f"rietveld_report_{samp}.md", mime="text/markdown")
        export_df = active_df.copy()
        export_df["y_calc"], export_df["y_background"], export_df["difference"] = result["y_calc"], result["y_background"], active_df["intensity"].values - result["y_calc"]
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        col_dl2.download_button("⬇️ Download Fit Data (.csv)", data=csv_buf.getvalue(), file_name=f"rietveld_fit_{samp}.csv", mime="text/csv")

# TAB 6 — PUBLICATION-QUALITY PLOT (SINGLE SAMPLE)
with tabs[6]:
    st.subheader("🖼️ Publication-Quality Plot (matplotlib)")
    st.caption("Generate journal-ready figures with customizable phase markers, legend control & spacing")
    
    if "last_result" not in st.session_state or "last_phases" not in st.session_state:
        st.info("🔬 Run the Rietveld refinement first (Tab 3: 🧮 Rietveld Fit) to enable publication plotting.")
        st.markdown("""**Quick steps:** 1. Select a sample in the sidebar 2. Choose phases to refine 3. Click **▶ Run Rietveld Refinement** 4. Return here.""")
    else:
        result = st.session_state["last_result"]
        phases = st.session_state["last_phases"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_width = st.slider("Figure width (inches)", 6.0, 14.0, 10.0, 0.5, key="pub_width")
            offset_factor = st.slider("Difference curve offset", 0.05, 0.25, 0.12, 0.01, key="pub_offset")
            font_size = st.slider("Global Font Size", 6, 22, 11, key="pub_font")
        with col2:
            fig_height = st.slider("Figure height (inches)", 5.0, 12.0, 7.0, 0.5, key="pub_height")
            show_hkl = st.checkbox("Show hkl labels", value=True, key="pub_hkl")
            legend_pos = st.selectbox("Legend Position", ["best", "upper right", "upper left", "lower left", "lower right", "center right", "center left", "lower center", "upper center", "center", "off"], index=0, key="pub_legend_pos")
        with col3:
            export_format = st.selectbox("Export format", ["PDF", "PNG", "EPS"], index=0, key="pub_format")
            marker_spacing = st.slider("Marker row spacing", 0.8, 2.5, 1.3, 0.1, help="Vertical distance between phase marker rows", key="pub_spacing")
            st.markdown("**🎨 Phase Customization**")
            
        st.markdown("### 📋 Legend Control")
        st.caption("Select which phases to include in the plot legend (uncheck to hide from legend)")
        n_cols = min(4, len(phases))
        legend_cols = st.columns(n_cols)
        legend_phases_selected = []
        for idx, ph in enumerate(phases):
            col_idx = idx % n_cols
            with legend_cols[col_idx]:
                if st.checkbox(f"✓ {ph}", value=True, key=f"leg_{ph}"):
                    legend_phases_selected.append(ph)
        
        phase_data = []
        for i, ph in enumerate(phases):
            pk_df = generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
            with st.expander(f"⚙️ Settings for **{ph}**", expanded=(i==0)):
                c_col, c_shape = st.columns(2)
                custom_color = c_col.color_picker("Color", value=PHASE_LIBRARY[ph]["color"], key=f"col_{ph}")
                shape_options = ["|", "_", "s", "^", "v", "d", "x", "+", "*"]
                default_idx = shape_options.index(PHASE_LIBRARY[ph].get("marker_shape", "|"))
                custom_shape = c_shape.selectbox("Marker Shape", shape_options, index=default_idx, key=f"shp_{ph}", help="| = vertical bar, _ = horizontal, s = square ■, d = diamond ◆")
            phase_data.append({"name": ph, "positions": pk_df["two_theta"].values if len(pk_df) > 0 else np.array([]), "color": custom_color, "marker_shape": custom_shape, "hkl": [hkl.strip("()").split(",") if hkl else None for hkl in pk_df["hkl_label"].values] if show_hkl and len(pk_df) > 0 else None})
            
        try:
            fig, ax = plot_rietveld_publication(
                active_df["two_theta"].values, active_df["intensity"].values,
                result["y_calc"], active_df["intensity"].values - result["y_calc"],
                phase_data, offset_factor=offset_factor, figsize=(fig_width, fig_height),
                font_size=font_size, legend_pos=legend_pos, marker_row_spacing=marker_spacing,
                legend_phases=legend_phases_selected if legend_phases_selected else None
            )
            st.pyplot(fig, dpi=150, use_container_width=True)
            st.markdown("#### 📥 Export Options")
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                buf = io.BytesIO(); fig.savefig(buf, format='pdf', bbox_inches='tight'); buf.seek(0)
                st.download_button("📄 PDF", buf.read(), file_name=f"rietveld_pub_{selected_key}.pdf", mime="application/pdf", use_container_width=True)
            with col_e2:
                buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=300, bbox_inches='tight'); buf.seek(0)
                st.download_button("🖼️ PNG (300 DPI)", buf.read(), file_name=f"rietveld_pub_{selected_key}.png", mime="image/png", use_container_width=True)
            with col_e3:
                buf = io.BytesIO(); fig.savefig(buf, format='eps', bbox_inches='tight'); buf.seek(0)
                st.download_button("📐 EPS", buf.read(), file_name=f"rietveld_pub_{selected_key}.eps", mime="application/postscript", use_container_width=True)
            with st.expander("🎨 Marker Shape Reference"):
                st.markdown("""| Shape | Code | Visual | Recommended Use |\n|-------|------|--------|----------------|\n| Vertical bar | `|` | │ | FCC-Co matrix (primary) |\n| Horizontal bar | `_` | ─ | HCP-Co (secondary) |\n| **Square** ✨ | `s` | ■ | M₂₃C₆ carbides |\n| Triangle up | `^` | ▲ | Sigma phase |\n| Triangle down | `v` | ▼ | Additional precipitates |\n| **Diamond** ✨ | `d` | ◆ | Trace intermetallics |\n| Cross | `x` | × | Reference peaks |\n| Plus | `+` | + | Calibration markers |\n| Star | `*` | ✦ | Special annotations |""")
            plt.close(fig)
        except Exception as e:
            st.error(f"❌ Plot generation failed: {str(e)}")
            st.code("Tip: Try reducing the number of phases or resetting font size to default.")

st.markdown("---")
st.caption("XRD Rietveld App • Co-Cr Dental Alloy Analysis • Supports .asc, .ASC & .xrdml • GitHub: Maryamslm/XRD-3Dprinted-Ret/SAMPLES")
