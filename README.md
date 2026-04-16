# XRD Rietveld Analysis — Co-Cr Dental Alloy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![CI](https://github.com/YOUR_USERNAME/xrd-rietveld-cocr/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/xrd-rietveld-cocr/actions)

Interactive Streamlit application for **Rietveld refinement** of X-ray diffraction data
from **Co-Cr-Mo-W-Si dental alloys** (Mediloy S Co, BEGO).

Supports **8 samples** in a single app with a dropdown selector:

| ID | Fabrication | Treatment | ψ angle |
|----|-------------|-----------|---------|
| CH0_1   | Cast    | Heat-treated     | 0°  |
| CH45_2  | Cast    | Heat-treated     | 45° |
| CNH0_3  | Cast    | Not heat-treated | 0°  |
| CNH45_4 | Cast    | Not heat-treated | 45° |
| PH0_5   | Printed (SLM) | Heat-treated     | 0°  |
| PH45_6  | Printed (SLM) | Heat-treated     | 45° |
| PNH0_7  | Printed (SLM) | Not heat-treated | 0°  |
| PNH45_8 | Printed (SLM) | Not heat-treated | 45° |

---

## 🔬 Phases Included

| Phase | System | Space Group | Relevance |
|-------|--------|-------------|-----------|
| **γ-Co (FCC)**  | Cubic       | Fm-3m (#225)   | Primary matrix — always dominant |
| **ε-Co (HCP)**  | Hexagonal   | P6₃/mmc (#194) | Martensitic product; more in as-built SLM |
| **σ (Co-Cr)**   | Tetragonal  | P4₂/mnm (#136) | Brittle TCP; grain-boundary precipitate |
| **α-Cr (BCC)**  | Cubic       | Im-3m (#229)   | Cr-rich zone; reduces corrosion resistance |
| **Co₃Mo**       | Hexagonal   | P6₃/mmc (#194) | Mo intermetallic |
| **Co₇Mo₆ (μ)** | Rhombohedral| R-3m (#166)    | TCP μ-phase |
| **Co₃W**        | Hexagonal   | P6₃/mmc (#194) | W intermetallic |
| **Co₂Si**       | Orthorhombic| Pbnm (#62)     | Si precipitate |

---

## 📐 App Features

| Tab | What you get |
|-----|-------------|
| **📈 Raw Pattern** | Interactive Plotly chart of selected sample |
| **🔍 Peak ID** | Automatic peak detection + phase tick-mark overlay |
| **🧮 Rietveld Fit** | Full Rietveld refinement with Obs/Calc/Diff plot |
| **📊 Quantification** | Pie chart + bar chart of phase weight fractions |
| **🔄 Sample Comparison** | Overlay all 8 patterns; ψ=0°/45° stress pairs |
| **📄 Report** | Downloadable Markdown report + CSV export |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/xrd-rietveld-cocr.git
cd xrd-rietveld-cocr

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```
xrd-rietveld-cocr/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── README.md
├── demo_data/                # All 8 XRD .ASC files bundled
│   ├── CH0_1.ASC
│   ├── CH45_2.ASC
│   ├── CNH0_3.ASC
│   ├── CNH45_4.ASC
│   ├── PH0_5.ASC
│   ├── PH45_6.ASC
│   ├── PNH0_7.ASC
│   └── PNH45_8.ASC
├── .streamlit/
│   └── config.toml
├── .github/
│   └── workflows/
│       └── ci.yml
└── utils/
    ├── __init__.py
    ├── sample_catalog.py     # 8-sample metadata + naming convention
    ├── phase_matcher.py      # Phase library + theoretical peak generator
    ├── peak_finder.py        # Savitzky-Golay smoothed peak detection
    ├── rietveld.py           # Rietveld engine (L-BFGS-B optimisation)
    └── report.py             # Markdown report generator
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. **Push** this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select repo · branch `main` · file `app.py`.
4. Click **Deploy** — no server setup needed.

> All 8 demo `.ASC` files are bundled in `demo_data/` so the app works
> out-of-the-box without any uploads.

---

## 🧮 Refinement Method

The engine performs a full-pattern least-squares Rietveld refinement:

| Parameter | Details |
|-----------|---------|
| **Background** | Chebyshev polynomial, order 2–8 |
| **Peak profile** | Pseudo-Voigt · Gaussian · Lorentzian · Pearson VII |
| **FWHM** | Caglioti: FWHM² = U·tan²θ + V·tanθ + W |
| **Lattice** | a (and c for non-cubic) refined per phase |
| **Scale** | Independent scale factor per phase |
| **Corrections** | LP factor, zero-shift |
| **Optimiser** | scipy L-BFGS-B with analytical bounds |
| **R-factors** | R_wp, R_exp, GoF (χ²) |

> For publication, export data and use **GSAS-II**, **FullProf**, or **Maud**
> with CIF structure files for full structure-factor calculations.

---

## 🩺 Residual Stress (sin²ψ method)

The ψ=0°/45° pairs enable residual stress estimation:

```
σ = -E/(1+ν) · 1/sin²ψ · Δd/d₀
```

where `Δd/d₀` is the fractional d-spacing shift between ψ=0° and ψ=45°.
The **Comparison tab** shows peak-shift tables for each sample pair.

---

## 📖 References

1. Rietveld, H. M. (1969). *J. Appl. Cryst.* **2**, 65–71.
2. Young, R. A. (1993). *The Rietveld Method*. IUCr/Oxford.
3. Caglioti, G., Paoletti, A. & Ricci, F. P. (1958). *Nucl. Instrum.* **3**, 223–228.
4. Yamanaka, K. et al. (2014). Microstructure and phase stability of Co-Cr alloys. *Acta Mater.*
5. Takaichi, A. et al. (2013). Microstructures and mechanical properties of Co-29Cr-6Mo alloy
   fabricated by selective laser melting. *Acta Biomater.*
6. BEGO Mediloy S Co alloy technical datasheet.

---

## 📄 License

MIT — free for academic and clinical research use.
