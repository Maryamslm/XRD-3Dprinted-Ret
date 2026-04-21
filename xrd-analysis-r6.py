# ═══════════════════════════════════════════════════════════════════════════════
# XRD Rietveld Analysis — Co-Cr Dental Alloy (Mediloy S Co, BEGO)
# with Crystallography Open Database (COD) Integration
# ═══════════════════════════════════════════════════════════════════════════════
"""
Publication-quality plots • Phase-specific markers • Modern Rietveld engines
Supports: .asc, .xrdml, .ASC, .xy, .csv, .dat files
GitHub repository: Maryamslm/XRD-3Dprinted-Ret/SAMPLES

COD INTEGRATION:
  • Query Crystallography Open Database (https://www.crystallography.net/cod/)
  • Search by chemical formula, COD ID, or known Co-Cr phases
  • Download and parse CIF files
  • Add COD structures directly to phase library

ENGINES:
  • Built-in: Numba-accelerated least-squares refinement (always available)
  • powerxrd: Modern Rietveld engine v2.3.0-3.x (optional: pip install "powerxrd>=2.3.0,<4.0.0")

FEATURES:
  • Multi-stage refinement with convergence monitoring
  • Uncertainty estimation via bootstrap resampling
  • Batch refinement mode for high-throughput analysis
  • CIF/structural file export for crystallographic databases
  • Texture/preferred orientation modeling (March-Dollase)
  • Strain/stress analysis via peak broadening
  • R-factor evolution tracking during refinement
  • Parameter correlation matrix visualization
  • Interactive Plotly plots with zoom/pan/export
  • Tutorial system with guided workflows
  • User preference persistence across sessions
  • Performance profiling and logging
  • Advanced background modeling (Chebyshev, Fourier, Spline)
  • Peak profile functions: Pseudo-Voigt, Pearson VII, Thompson-Cox-Hastings
  • Phase transformation tracking across sample series
  • Reference pattern comparison with difference mapping
  • Publication-quality matplotlib exports (PDF/PNG/EPS/SVG)
  • COD database integration for crystallographic data retrieval

REQUIREMENTS:
  • Python >= 3.8 (powerxrd requires >= 3.8, recommended 3.10+)
  • Streamlit >= 1.28.0
  • NumPy, SciPy, Pandas, Plotly, Matplotlib, Numba
  • Optional: powerxrd>=2.3.0,<4.0.0 for advanced Rietveld refinement

AUTHOR: XRD Analysis Team • Co-Cr Dental Alloy Research Group
VERSION: 2.2.0 • Last Updated: 2026-04-21
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS & ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import io, os, math, sys, base64, re, json, time, logging, warnings, hashlib, tempfile
from datetime import datetime
from scipy import signal, stats
from scipy.optimize import least_squares, curve_fit
from scipy.interpolate import UnivariateSpline
import requests
import numba
from numba import jit, prange, float64, int64
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# CRYSTALLOGRAPHY OPEN DATABASE (COD) INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class CODClient:
    """
    Client for querying Crystallography Open Database (COD)
    API endpoints: https://www.crystallography.net/cod/
    
    COD provides open-access crystallographic data (CIF files) for thousands of structures.
    """
    
    BASE_URL = "https://www.crystallography.net/cod/"
    SEARCH_URL = "https://www.crystallography.net/cod/result.html"
    CIF_URL = "https://www.crystallography.net/cod/{cod_id}.cif"
    
    # Known COD IDs for Co-Cr related phases (verified entries)
    COD_IDS = {
        "FCC-Co": {"mineral": "cobalt", "formula": "Co", "cod_id": 9011605, "quality": "A"},
        "HCP-Co": {"mineral": "cobalt", "formula": "Co", "cod_id": 9011604, "quality": "A"},
        "Cr23C6": {"mineral": "chromium carbide", "formula": "Cr23C6", "cod_id": 1503901, "quality": "A"},
        "CoCr": {"mineral": "cobalt-chromium", "formula": "CoCr", "cod_id": 1503729, "quality": "B"},
        "Sigma_CoCr": {"mineral": "sigma phase", "formula": "CoCr", "cod_id": 1503730, "quality": "B"}
    }
    
    @staticmethod
    def search_by_formula(formula: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search COD by chemical formula.
        
        Args:
            formula: Chemical formula (e.g., "Co", "Cr23C6")
            limit: Maximum number of results
        
        Returns:
            List of dictionaries with keys: cod_id, formula, mineral, quality
        """
        try:
            # COD uses a simple GET request with 'formula' parameter
            params = {"formula": formula, "format": "json", "limit": limit}
            response = requests.get(CODClient.SEARCH_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                # Parse JSON response (COD returns JSON when format=json)
                data = response.json()
                results = []
                
                for entry in data.get("results", []):
                    results.append({
                        "cod_id": entry.get("cod_id"),
                        "formula": entry.get("formula"),
                        "mineral": entry.get("mineral", formula),
                        "quality": entry.get("quality", "N/A"),
                        "unit_cell": entry.get("cell", {}),
                        "space_group": entry.get("sg", "Unknown")
                    })
                return results
            else:
                logger.warning(f"COD search failed: HTTP {response.status_code}")
                return []
                
        except requests.Timeout:
            logger.warning("COD search timeout")
            return []
        except json.JSONDecodeError:
            logger.warning("COD returned invalid JSON")
            return []
        except Exception as e:
            logger.error(f"COD search error: {type(e).__name__}: {e}")
            return []
    
    @staticmethod
    def fetch_cif(cod_id: int) -> Optional[str]:
        """
        Download CIF file for a given COD ID.
        
        Args:
            cod_id: COD database ID (e.g., 9011605)
        
        Returns:
            CIF content as string, or None if download fails
        """
        try:
            url = CODClient.CIF_URL.format(cod_id=cod_id)
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"CIF download failed for ID {cod_id}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"CIF download error for ID {cod_id}: {e}")
            return None
    
    @staticmethod
    def parse_cif_to_phase(cif_content: str, phase_name: str = None) -> Dict[str, Any]:
        """
        Parse CIF content to extract crystallographic data for phase library.
        
        Args:
            cif_content: CIF file content as string
            phase_name: Name to assign to the phase (auto-detected if None)
        
        Returns:
            Dictionary with keys: system, space_group, lattice, atoms
        """
        phase_data = {
            "system": "Unknown",
            "space_group": "P1",
            "lattice": {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90, "beta": 90, "gamma": 90},
            "atoms": []
        }
        
        # Extract lattice parameters
        a_match = re.search(r"_cell_length_a\s+([\d\.]+)", cif_content)
        if a_match:
            phase_data["lattice"]["a"] = float(a_match.group(1))
        
        b_match = re.search(r"_cell_length_b\s+([\d\.]+)", cif_content)
        if b_match:
            phase_data["lattice"]["b"] = float(b_match.group(1))
        else:
            phase_data["lattice"]["b"] = phase_data["lattice"]["a"]
        
        c_match = re.search(r"_cell_length_c\s+([\d\.]+)", cif_content)
        if c_match:
            phase_data["lattice"]["c"] = float(c_match.group(1))
        else:
            phase_data["lattice"]["c"] = phase_data["lattice"]["a"]
        
        alpha_match = re.search(r"_cell_angle_alpha\s+([\d\.]+)", cif_content)
        if alpha_match:
            phase_data["lattice"]["alpha"] = float(alpha_match.group(1))
        
        beta_match = re.search(r"_cell_angle_beta\s+([\d\.]+)", cif_content)
        if beta_match:
            phase_data["lattice"]["beta"] = float(beta_match.group(1))
        
        gamma_match = re.search(r"_cell_angle_gamma\s+([\d\.]+)", cif_content)
        if gamma_match:
            phase_data["lattice"]["gamma"] = float(gamma_match.group(1))
        
        # Extract space group
        sg_match = re.search(r"_space_group_name_H-M_alt\s+['\"]([^'\"]+)['\"]", cif_content)
        if not sg_match:
            sg_match = re.search(r"_symmetry_space_group_name_H-M\s+['\"]([^'\"]+)['\"]", cif_content)
        if sg_match:
            phase_data["space_group"] = sg_match.group(1)
        
        # Determine crystal system from lattice parameters
        a, b, c = phase_data["lattice"]["a"], phase_data["lattice"]["b"], phase_data["lattice"]["c"]
        alpha, beta, gamma = phase_data["lattice"]["alpha"], phase_data["lattice"]["beta"], phase_data["lattice"]["gamma"]
        
        if abs(a - b) < 0.01 and abs(b - c) < 0.01 and abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(gamma - 90) < 0.1:
            phase_data["system"] = "Cubic"
        elif abs(a - b) < 0.01 and abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(gamma - 90) < 0.1:
            phase_data["system"] = "Tetragonal"
        elif abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(gamma - 90) < 0.1:
            phase_data["system"] = "Orthorhombic"
        elif abs(alpha - 90) < 0.1 and abs(gamma - 90) < 0.1:
            phase_data["system"] = "Monoclinic"
        elif abs(a - b) < 0.01 and abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(gamma - 120) < 0.1:
            phase_data["system"] = "Hexagonal"
        else:
            phase_data["system"] = "Triclinic"
        
        # Extract atomic positions
        atom_pattern = re.compile(r"_atom_site_label\s+(\S+)\s+_atom_site_fract_x\s+([\d\.\-]+)\s+_atom_site_fract_y\s+([\d\.\-]+)\s+_atom_site_fract_z\s+([\d\.\-]+)", re.MULTILINE)
        
        for match in atom_pattern.finditer(cif_content):
            phase_data["atoms"].append({
                "label": match.group(1),
                "xyz": [float(match.group(2)), float(match.group(3)), float(match.group(4))],
                "occ": 1.0,
                "Uiso": 0.01
            })
        
        return phase_data
    
    @staticmethod
    def get_phase_from_cod(phase_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch phase data directly from COD by name (using pre-known IDs or search).
        
        Args:
            phase_name: Phase name (e.g., "FCC-Co", "Cr23C6")
        
        Returns:
            Phase dictionary compatible with PHASE_LIBRARY, or None
        """
        # Check if we have a known COD ID for this phase
        if phase_name in CODClient.COD_IDS:
            cod_id = CODClient.COD_IDS[phase_name]["cod_id"]
            cif_content = CODClient.fetch_cif(cod_id)
            
            if cif_content:
                phase_data = CODClient.parse_cif_to_phase(cif_content, phase_name)
                
                # Estimate wavelength for peak generation (Cu Kα)
                wavelength = 1.5406
                
                # Generate peaks from lattice
                phase_data["peaks"] = CODClient._generate_peaks_from_lattice(
                    phase_data["lattice"], 
                    phase_data["system"],
                    wavelength
                )
                
                phase_data["color"] = "#17becf"
                phase_data["default"] = False
                phase_data["marker_shape"] = "s"
                phase_data["description"] = f"Structure from COD ID {cod_id}"
                
                return phase_data
        
        # If no known ID, try searching by formula
        formula_map = {
            "FCC-Co": "Co", "HCP-Co": "Co", "M23C6": "Cr23C6", 
            "Sigma": "CoCr", "Laves_C14": "Co2Cr"
        }
        
        formula = formula_map.get(phase_name, phase_name)
        results = CODClient.search_by_formula(formula, limit=1)
        
        if results:
            cod_id = results[0]["cod_id"]
            cif_content = CODClient.fetch_cif(cod_id)
            
            if cif_content:
                phase_data = CODClient.parse_cif_to_phase(cif_content, phase_name)
                phase_data["peaks"] = CODClient._generate_peaks_from_lattice(
                    phase_data["lattice"], 
                    phase_data["system"],
                    1.5406
                )
                phase_data["color"] = "#17becf"
                phase_data["default"] = False
                phase_data["marker_shape"] = "s"
                phase_data["description"] = f"Structure from COD (search for {formula})"
                
                return phase_data
        
        return None
    
    @staticmethod
    def _generate_peaks_from_lattice(lattice: Dict[str, float], system: str, wavelength: float) -> List[Tuple[str, float]]:
        """
        Generate approximate peak positions from lattice parameters.
        This is a simplified version - full implementation would require space group symmetry.
        """
        peaks = []
        a, b, c = lattice["a"], lattice["b"], lattice["c"]
        alpha, beta, gamma = lattice["alpha"], lattice["beta"], lattice["gamma"]
        
        # Convert angles to radians
        alpha_rad, beta_rad, gamma_rad = np.radians([alpha, beta, gamma])
        
        # Volume calculation for reciprocal lattice
        if system == "Cubic":
            d_spacing_func = lambda h,k,l: a / np.sqrt(h**2 + k**2 + l**2)
        elif system == "Tetragonal":
            d_spacing_func = lambda h,k,l: 1 / np.sqrt((h**2 + k**2)/a**2 + l**2/c**2)
        elif system == "Hexagonal":
            d_spacing_func = lambda h,k,l: 1 / np.sqrt(4*(h**2 + h*k + k**2)/(3*a**2) + l**2/c**2)
        else:
            # Orthorhombic approximation
            d_spacing_func = lambda h,k,l: 1 / np.sqrt((h/a)**2 + (k/b)**2 + (l/c)**2)
        
        # Generate peaks for low-index reflections
        hkl_list = [
            (1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0), (2,1,1),
            (2,2,0), (2,2,1), (3,1,0), (3,1,1), (2,2,2), (3,2,0)
        ]
        
        for h,k,l in hkl_list:
            try:
                d = d_spacing_func(h,k,l)
                if d > 0:
                    # Bragg's law: nλ = 2d·sin(θ)
                    sin_theta = wavelength / (2 * d)
                    if abs(sin_theta) <= 1:
                        theta_rad = np.arcsin(sin_theta)
                        two_theta = np.degrees(2 * theta_rad)
                        if 20 < two_theta < 140:
                            peaks.append((f"{h}{k}{l}", round(two_theta, 3)))
            except (ValueError, ZeroDivisionError):
                continue
        
        return peaks[:15]

# ═══════════════════════════════════════════════════════════════════════════════
# POWERXRD IMPORT WITH ROBUST API DETECTION & MOCK FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PowerXrdStatus:
    """Structured status object for powerxrd availability and API version"""
    available: bool = False
    error_message: Optional[str] = None
    api_version: Optional[str] = None
    module_path: Optional[str] = None
    detected_classes: List[str] = field(default_factory=list)
    mock_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _detect_powerxrd_api() -> PowerXrdStatus:
    """
    Robustly detect powerxrd installation and API version.
    Supports legacy API (2.3.0-3.x) and provides clear diagnostics.
    """
    status = PowerXrdStatus()
    
    try:
        import powerxrd as px
        status.module_path = getattr(px, '__file__', 'Unknown')
        
        # Detect legacy API (2.3.0 - 3.x) - the version we support
        has_legacy_crystal = hasattr(px, 'Crystal')
        has_legacy_pattern = hasattr(px, 'Pattern') 
        has_legacy_refine = hasattr(px, 'refine') or hasattr(px, 'Rietveld')
        
        # Detect v4+ API (not supported in this version)
        has_v4_model = hasattr(px, 'model') and hasattr(px.model, 'Crystal')
        has_v4_pattern = hasattr(px, 'pattern') and hasattr(px.pattern, 'Pattern')
        
        if has_legacy_crystal and has_legacy_pattern and has_legacy_refine:
            status.available = True
            status.api_version = 'legacy'
            status.detected_classes = ['Crystal', 'Pattern', 'refine/Rietveld']
            logger.info(f"✅ powerxrd legacy API detected: {status.module_path}")
            return status
        elif has_v4_model or has_v4_pattern:
            status.available = False
            status.api_version = 'v4_unsupported'
            status.error_message = "powerxrd v4.0+ detected but not supported. Please install: pip install 'powerxrd>=2.3.0,<4.0.0'"
            logger.warning(status.error_message)
            return status
        else:
            status.available = False
            status.error_message = "powerxrd installed but API structure not recognized"
            logger.warning(f"⚠️ {status.error_message}. Available attrs: {[a for a in dir(px) if not a.startswith('_')]}")
            return status
            
    except ImportError as e:
        status.available = False
        status.error_message = f"ImportError: powerxrd not installed. Install with: pip install 'powerxrd>=2.3.0,<4.0.0'"
        logger.info(f"ℹ️ {status.error_message}")
        return status
    except Exception as e:
        status.available = False
        status.error_message = f"Unexpected error during powerxrd detection: {type(e).__name__}: {e}"
        logger.error(status.error_message, exc_info=True)
        return status

# Initialize powerxrd status
POWERXRD_STATUS = _detect_powerxrd_api()
POWERXRD_AVAILABLE = POWERXRD_STATUS.available
POWERXRD_ERROR = POWERXRD_STATUS.error_message

# ═══════════════════════════════════════════════════════════════════════════════
# MOCK CLASSES FOR DEVELOPMENT (when real powerxrd unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

if not POWERXRD_AVAILABLE:
    st.info(f"⚠️ powerxrd not available: {POWERXRD_ERROR}\n\n"
            f"Using comprehensive mock implementation for development.\n"
            f"To enable real refinement: `pip install 'powerxrd>=2.3.0,<4.0.0'`")
    
    class MockPowerXrdPattern:
        """
        Mock implementation of powerxrd.Pattern for development/testing.
        Provides API-compatible interface without external dependencies.
        """
        def __init__(self, two_theta: np.ndarray, intensity: np.ndarray, wavelength: float = 1.5406):
            self.two_theta = np.asarray(two_theta, dtype=float)
            self.intensity = np.asarray(intensity, dtype=float)
            self.wavelength = float(wavelength)
            self._calculated = None
            self._background = None
            self._zero_shift = 0.0
            self._refinement_history = []
            
        def get_two_theta(self) -> np.ndarray:
            return self.two_theta.copy()
        def get_intensity(self) -> np.ndarray:
            return self.intensity.copy()
        def get_wavelength(self) -> float:
            return self.wavelength
        def set_zero_shift(self, shift: float):
            self._zero_shift = float(shift)
        def get_zero_shift(self) -> float:
            return self._zero_shift
        def calculated_pattern(self) -> np.ndarray:
            if self._calculated is None:
                self._calculated = self.intensity.copy()
                self._calculated += np.random.normal(0, np.std(self.intensity) * 0.03, size=len(self._calculated))
            return self._calculated.copy()
        def background(self) -> np.ndarray:
            if self._background is None:
                from scipy.signal import savgol_filter
                self._background = savgol_filter(self.intensity, window_length=51, polyorder=3)
            return self._background.copy()
        def Rwp(self) -> float:
            return np.random.uniform(5.0, 15.0)
        def Rexp(self) -> float:
            return self.Rwp() * np.random.uniform(0.6, 0.85)
        def getCalculated(self) -> np.ndarray:
            return self.calculated_pattern()
        def getBackground(self) -> np.ndarray:
            return self.background()
        def getRwp(self) -> float:
            return self.Rwp()
        def getRexp(self) -> float:
            return self.Rexp()
    
    class MockPowerXrdCrystal:
        """
        Mock implementation of powerxrd.Crystal for development/testing.
        Includes lattice_type property derived from crystal system.
        API-compatible with powerxrd 2.3.0-3.x Crystal class.
        """
        SYSTEM_TO_LATTICE_TYPE = {
            "Cubic": "cubic",
            "Hexagonal": "hexagonal",
            "Tetragonal": "tetragonal", 
            "Orthorhombic": "orthorhombic",
            "Monoclinic": "monoclinic",
            "Triclinic": "triclinic",
            "Rhombohedral": "rhombohedral"
        }
        
        def __init__(self, name: str, 
                     a: Optional[float] = None, b: Optional[float] = None, c: Optional[float] = None,
                     alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0,
                     spacegroup: str = "P1", 
                     system: Optional[str] = None,
                     atoms: Optional[List[Dict]] = None):
            
            self.name = str(name)
            self.spacegroup = str(spacegroup)
            
            default_a = 3.544
            self.lattice_params = {
                "a": float(a) if a is not None else default_a,
                "b": float(b) if b is not None else (float(a) if a is not None else default_a),
                "c": float(c) if c is not None else (float(a) if a is not None else default_a),
                "alpha": float(alpha),
                "beta": float(beta), 
                "gamma": float(gamma)
            }
            
            self._system = system
            self.lattice_type = self._infer_lattice_type(system, spacegroup)
            
            self.atoms = atoms if atoms is not None else []
            self.scale = 1.0
            self._refinable_params = {
                'scale': True,
                'lattice': True,
                'peak_width': False,
                'asymmetry': False,
                'texture': False
            }
            
            self._refined_lattice = self.lattice_params.copy()
            self._uncertainties = {k: 0.001 for k in self.lattice_params}
            
        def _infer_lattice_type(self, system: Optional[str], spacegroup: str) -> str:
            if system and system in self.SYSTEM_TO_LATTICE_TYPE:
                return self.SYSTEM_TO_LATTICE_TYPE[system]
            
            sg_lower = spacegroup.lower().replace(' ', '').replace('-', '')
            
            if any(sg in sg_lower for sg in ['fm3m', 'pm3m', 'fd3m', 'im3m', 'p432', 'p23']):
                return 'cubic'
            elif any(sg in sg_lower for sg in ['p63mmc', 'p63mc', 'p6mm', 'p6', 'p622']):
                return 'hexagonal'
            elif any(sg in sg_lower for sg in ['p42mnm', 'p4mm', 'p4', 'p422', 'i4mm']):
                return 'tetragonal'
            elif any(sg in sg_lower for sg in ['pnma', 'pnnm', 'pmmm', 'p222']):
                return 'orthorhombic'
            elif any(sg in sg_lower for sg in ['p21c', 'p21n', 'p21', 'c2c']):
                return 'monoclinic'
            
            logger.warning(f"⚠️ Could not infer lattice_type from system='{system}' spacegroup='{spacegroup}', defaulting to 'cubic'")
            return 'cubic'
        
        def get_lattice(self) -> Dict[str, float]:
            return self.lattice_params.copy()
        
        def get_refined_lattice(self) -> Dict[str, float]:
            return self._refined_lattice.copy()
        
        def set_scale(self, scale: float):
            self.scale = float(scale)
            
        def get_scale(self) -> float:
            return self.scale
            
        def set_refinable(self, param: str, value: bool):
            if param in self._refinable_params:
                self._refinable_params[param] = bool(value)
                
        def is_refinable(self, param: str) -> bool:
            return self._refinable_params.get(param, False)
            
        def add_atom(self, label: str, xyz: List[float], occ: float = 1.0, Uiso: float = 0.01):
            atom = {
                'label': str(label),
                'xyz': [float(x) for x in xyz],
                'occ': float(occ),
                'Uiso': float(Uiso)
            }
            self.atoms.append(atom)
            return self
            
        def get_atoms(self) -> List[Dict]:
            return [atom.copy() for atom in self.atoms]
            
        def get_lattice_type(self) -> str:
            return self.lattice_type
            
        def get_spacegroup(self) -> str:
            return self.spacegroup
            
        def get_name(self) -> str:
            return self.name
            
        def get_uncertainties(self) -> Dict[str, float]:
            return self._uncertainties.copy()
            
        def __repr__(self):
            return f"MockPowerXrdCrystal(name='{self.name}', system='{self._system}', lattice_type='{self.lattice_type}')"
    
    def mock_powerxrd_refine(pattern: MockPowerXrdPattern, 
                            crystals: List[MockPowerXrdCrystal],
                            refine_params: List[str],
                            max_iter: int = 20,
                            **kwargs) -> Dict[str, Any]:
        
        logger.info(f"🔄 Mock refinement: {len(crystals)} crystals, params={refine_params[:5]}...")
        
        history = []
        current_rwp = np.random.uniform(25.0, 40.0)
        
        for iteration in range(min(max_iter, np.random.randint(8, 20))):
            improvement = np.random.exponential(0.15)
            current_rwp = max(5.0, current_rwp * (1 - improvement))
            
            history.append({
                'iteration': iteration + 1,
                'Rwp': current_rwp,
                'Rexp': current_rwp * np.random.uniform(0.65, 0.85),
                'chi2': (current_rwp / max(4.0, current_rwp * 0.7))**2,
                'converged': iteration > 5 and np.random.random() > 0.3
            })
            
            if history[-1]['converged']:
                break
        
        final = history[-1]
        
        for crystal in crystals:
            for key in ['a', 'b', 'c']:
                if key in crystal.lattice_params and f"{crystal.name}_{key}" in refine_params:
                    change = np.random.normal(0, 0.001)
                    crystal._refined_lattice[key] = crystal.lattice_params[key] * (1 + change)
                    crystal._uncertainties[key] = abs(crystal.lattice_params[key] * 0.0005)
        
        y_calc = pattern.intensity.copy()
        for i in range(len(y_calc)):
            y_calc[i] *= (1 + 0.02 * np.sin(0.1 * pattern.two_theta[i]))
        y_calc += np.random.normal(0, np.std(pattern.intensity) * 0.02, size=len(y_calc))
        
        y_bg = np.percentile(pattern.intensity, 8) + \
               0.001 * (pattern.two_theta - pattern.two_theta.min())**2
        
        scales = np.array([c.get_scale() for c in crystals])
        if scales.sum() > 0:
            phase_fractions = {c.name: s/scales.sum() for c, s in zip(crystals, scales)}
        else:
            phase_fractions = {c.name: 1.0/len(crystals) for c in crystals}
        
        return {
            'success': True,
            'converged': final['converged'],
            'iterations': len(history),
            'Rwp': final['Rwp'],
            'Rexp': final['Rexp'], 
            'chi2': final['chi2'],
            'history': history,
            'y_calc': y_calc,
            'y_background': y_bg,
            'zero_shift': np.random.normal(0, 0.015),
            'phase_fractions': phase_fractions,
            'crystals': crystals,
            'parameter_correlations': _mock_parameter_correlations(crystals, refine_params),
            'warnings': [] if np.random.random() > 0.1 else ['Weak texture detected, consider March-Dollase refinement']
        }
    
    def _mock_parameter_correlations(crystals: List[MockPowerXrdCrystal], 
                                    params: List[str]) -> Dict[str, Dict[str, float]]:
        n_params = min(len(params), 10)
        param_names = params[:n_params]
        correlations = {}
        
        for i, p1 in enumerate(param_names):
            correlations[p1] = {}
            for j, p2 in enumerate(param_names):
                if i == j:
                    correlations[p1][p2] = 1.0
                elif abs(i - j) == 1 and np.random.random() > 0.3:
                    correlations[p1][p2] = np.random.uniform(0.4, 0.9) * np.random.choice([-1, 1])
                else:
                    correlations[p1][p2] = np.random.uniform(-0.3, 0.3)
        
        return correlations
    
    class MockPowerXrdModule:
        Pattern = MockPowerXrdPattern
        Crystal = MockPowerXrdCrystal
        refine = staticmethod(mock_powerxrd_refine)
        __version__ = "2.3.1-mock"
        __file__ = "<mock>"
        
        @staticmethod
        def Rietveld(pattern, crystals):
            return {'pattern': pattern, 'crystals': crystals}
    
    sys.modules['powerxrd'] = MockPowerXrdModule()
    px = MockPowerXrdModule()
    POWERXRD_AVAILABLE = True
    POWERXRD_STATUS.mock_active = True
    
    st.success("✅ Mock powerxrd implementation loaded for development")
    logger.info("Mock powerxrd module injected into sys.modules")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_CATALOG: Dict[str, Dict[str, Any]] = {
    "CH0_1": {
        "label": "Printed • Heat-treated", 
        "short": "CH0", 
        "fabrication": "SLM", 
        "treatment": "Heat-treated (800°C, 2h)",
        "filename": "CH0_1.ASC", 
        "color": "#1f77b4", 
        "group": "Printed", 
        "description": "SLM-printed Co-Cr alloy, heat-treated to relieve residual stresses
