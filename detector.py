"""
detector.py

Detector sub-system and readout simulation engine for ExoETC (ExoWorlds mission).
Implements MULTIACCUM ramp modeling, saturation checks, effective group truncation,
and dual-track noise evaluation following the JWST/NIRSpec NEETC framework
(Nielsen et al. 2016) and pyEDITH modularity conventions (Alei et al. 2026).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from astropy import units as u


class BaseDetector(ABC):
    """Abstract Base Class for focal plane detector arrays in ExoETC."""

    @abstractmethod
    def evaluate_saturation_and_groups(
        self,
        rate_peak_pix: Union[np.ndarray, float],
        n_g_commanded: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates usable effective groups (n_eff) and identifies hard-saturated pixels.
        """
        pass

    @abstractmethod
    def compute_cycle_timing(
        self,
        n_eff: Union[np.ndarray, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates integration time (t_int), total cycle period (t_cycle),
        and duty cycle efficiency (eta_duty).
        """
        pass

    @abstractmethod
    def evaluate_noise_variance(
        self,
        n_e_signal: Union[np.ndarray, float],
        t_int: Union[np.ndarray, float],
        n_eff: Union[np.ndarray, int],
        n_extracted_pix: float = 1.0,
        rate_bg_e_per_s: float = 0.0
    ) -> np.ndarray:
        """
        Evaluates single-integration variance using the dual-track noise matrix.
        """
        pass


class ExoWorldsDetector(BaseDetector):
    """
    Simulates infrared Focal Plane Arrays (e.g., Teledyne H2RG / HAWAII arrays)
    under MULTIACCUM up-the-ramp readout for exoplanet transit spectroscopy.

    Parameters
    ----------
    full_well_electrons : float, optional
        Conservative pixel full-well capacity in photo-electrons (default: 77,000 e-).
    read_noise : float, optional
        Correlated double sampling (CDS) single-frame read noise (default: 18.0 e- rms).
    ktc_reset_noise : float, optional
        Uncorrelated kTC reset noise per pixel (default: 60.0 e- rms).
    dark_current : float, optional
        Thermal dark current per pixel per second (default: 0.01 e-/pix/s).
    gain : float, optional
        Electronic conversion gain (default: 2.0 e-/ADU).
    t_frame : float, optional
        Frame readout time for the designated subarray geometry in seconds (default: 0.0737 s).
    name : str, optional
        Detector identifier string (default: "ExoWorlds_IR_H2RG").
    """

    def __init__(
        self,
        full_well_electrons: float = 77000.0,
        read_noise: float = 18.0,
        ktc_reset_noise: float = 60.0,
        dark_current: float = 0.01,
        gain: float = 2.0,
        t_frame: float = 0.0737,
        name: str = "ExoWorlds_IR_H2RG"
    ):
        self.name = name
        self.full_well = float(full_well_electrons)
        self.read_noise = float(read_noise)
        self.ktc_noise = float(ktc_reset_noise)
        self.dark_current = float(dark_current)
        self.gain = float(gain)
        self.t_frame = float(t_frame)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ExoWorldsDetector":
        """Factory constructor instantiating from standardized configuration dictionaries."""
        return cls(
            full_well_electrons=config.get("full_well_electrons", 77000.0),
            read_noise=config.get("read_noise", 18.0),
            ktc_reset_noise=config.get("ktc_reset_noise", 60.0),
            dark_current=config.get("dark_current", 0.01),
            gain=config.get("gain", 2.0),
            t_frame=config.get("t_frame", 0.0737),
            name=config.get("name", "ExoWorlds_IR_H2RG")
        )

    def evaluate_saturation_and_groups(
        self,
        rate_peak_pix: Union[np.ndarray, float],
        n_g_commanded: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates peak pixel accumulation against the detector full well.
        Computes effective usable groups before saturation (n_eff).

        Parameters
        ----------
        rate_peak_pix : np.ndarray or float
            Electron arrival rate on the brightest central pixel (e-/s).
        n_g_commanded : int
            Commanded groups per integration (MULTIACCUM ramp length).

        Returns
        -------
        n_eff : np.ndarray (dtype=int)
            Array of usable groups per spectral channel before pixel overflow.
        is_hard_saturated : np.ndarray (dtype=bool)
            True where a pixel exceeds full well within the first readout frame.
        """
        rate = np.asarray(rate_peak_pix, dtype=np.float64)

        # Total ramp accumulation on the brightest pixel if unclipped
        accumulated_peak = rate * (n_g_commanded * self.t_frame)

        # Theoretical groups before reaching well depth (floor division)
        with np.errstate(divide='ignore', invalid='ignore'):
            groups_to_sat = np.floor(self.full_well / (rate * self.t_frame))
            groups_to_sat = np.nan_to_num(groups_to_sat, nan=n_g_commanded, posinf=n_g_commanded)

        # Pixels exceeding full well in the very first frame are hard saturated
        is_hard_saturated = groups_to_sat < 1.0

        # Vectorized group clamping: capped at n_g_commanded, floor at 1
        n_eff = np.where(
            accumulated_peak > self.full_well,
            np.clip(groups_to_sat, 1, n_g_commanded),
            n_g_commanded
        ).astype(int)

        return n_eff, is_hard_saturated

    def compute_cycle_timing(
        self,
        n_eff: Union[np.ndarray, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates integration timing and observational duty cycle.

        Following Nielsen et al. (2016):
        - For multi-group CDS (n_eff >= 2): t_int = (n_eff - 1) * t_frame
        - For single-group reset-read (n_eff == 1): t_int = 1 * t_frame
        - Total cycle time includes one reset frame: t_cycle = (n_eff + 1) * t_frame

        Parameters
        ----------
        n_eff : np.ndarray or int
            Usable groups per integration.

        Returns
        -------
        t_int : np.ndarray
            Active photon integration duration per ramp (seconds).
        t_cycle : np.ndarray
            Total integration cycle period including reset dead-time (seconds).
        duty_cycle : np.ndarray
            Fraction of total cycle time spent collecting photons (t_int / t_cycle).
        """
        ng = np.asarray(n_eff, dtype=int)

        # Integration time formulation: CDS differencing uses (n_eff - 1) frames
        t_int = np.where(ng > 1, (ng - 1) * self.t_frame, 1.0 * self.t_frame)

        # Total cycle time includes one non-integrating reset frame[cite: 294, 303]
        t_cycle = (ng + 1.0) * self.t_frame

        duty_cycle = t_int / t_cycle
        return t_int, t_cycle, duty_cycle

    def evaluate_noise_variance(
        self,
        n_e_signal: Union[np.ndarray, float],
        t_int: Union[np.ndarray, float],
        n_eff: Union[np.ndarray, int],
        n_extracted_pix: float = 1.0,
        rate_bg_e_per_s: float = 0.0
    ) -> np.ndarray:
        """
        Evaluates single-integration statistical variance sigma_int^2 via the
        dual-track noise matrix from Nielsen et al. (2016) Section 3.1[cite: 294, 303].

        Track A: CDS Multi-Group (n_eff >= 2)[cite: 294, 303]
            sigma^2 = S_int + 2 * sigma_read^2 + N_pix * (I_dark + I_bg) * t_int
            (First-frame differencing cancels kTC reset noise)[cite: 294, 303]

        Track B: Reset-Read Single-Group (n_eff == 1)[cite: 294, 303]
            sigma^2 = S_int + sigma_read^2 + sigma_kTC^2 + N_pix * (I_dark + I_bg) * t_frame
            (Single read must reference zero level, incurring kTC noise)[cite: 294, 303]

        Parameters
        ----------
        n_e_signal : np.ndarray or float
            Accumulated stellar photo-electrons inside extraction aperture per integration.
        t_int : np.ndarray or float
            Active integration duration (seconds).
        n_eff : np.ndarray or int
            Usable groups per integration.
        n_extracted_pix : float, optional
            Number of detector pixels in the spatial photometric extraction aperture.
        rate_bg_e_per_s : float, optional
            Background count rate per pixel (zodiacal, thermal, straylight) in e-/pix/s.

        Returns
        -------
        var_per_int : np.ndarray
            Variance per integration (e-^2).
        """
        signal = np.asarray(n_e_signal, dtype=np.float64)
        t_i = np.asarray(t_int, dtype=np.float64)
        ng = np.asarray(n_eff, dtype=int)

        # Dark current and background Poisson noise
        var_dark_bg = n_extracted_pix * (self.dark_current + rate_bg_e_per_s) * t_i

        # Dual-track detector read noise matrix[cite: 294, 303]
        var_read = np.where(
            ng == 1,
            self.read_noise**2 + self.ktc_noise**2,  # Track B (Reset-Read)[cite: 294, 303]
            2.0 * (self.read_noise**2)               # Track A (CDS Multi-Group)[cite: 294, 303]
        )

        # Total variance: Source Poisson shot noise + Dark/BG + Detector Read Noise
        var_per_int = signal + var_read + var_dark_bg
        return var_per_int

    def format_status_summary(
        self,
        wavelengths: np.ndarray,
        n_eff: np.ndarray,
        hard_sat: np.ndarray
    ) -> Dict[str, Any]:
        """Provides diagnostic metrics on array-wide saturation and efficiency."""
        total_channels = len(wavelengths)
        n_sat = int(np.sum(hard_sat))
        n_track_b = int(np.sum(n_eff == 1)) - n_sat
        n_track_a = int(np.sum(n_eff >= 2))

        return {
            "total_channels": total_channels,
            "hard_saturated_channels": n_sat,
            "track_b_channels (n_eff=1)": n_track_b,
            "track_a_channels (n_eff>=2)": n_track_a,
            "min_effective_groups": int(np.min(n_eff)),
            "max_effective_groups": int(np.max(n_eff)),
            "wavelength_saturation_mask": hard_sat
        }

    import numpy as np
from detector import ExoWorldsDetector

# 1. Initialize detector with NIRSpec SUB512 parameters
detector = ExoWorldsDetector(
    full_well_electrons=77000.0,
    read_noise=18.0,
    ktc_reset_noise=60.0,
    dark_current=0.01,
    gain=2.0,
    t_frame=0.0737,
    name="ExoWorlds_IR_H2RG"
)

# 2. Define three test wavelength channels representing GJ 1214:
# Channel 0 (1.25 um, J-band peak): ~500,000 e-/s on peak pixel (very bright)
# Channel 1 (2.20 um, continuum):   ~250,000 e-/s on peak pixel (moderate)
# Channel 2 (4.50 um, faint wing):   ~80,000 e-/s on peak pixel (low flux)
wavelengths = np.array([1.25, 2.20, 4.50])
rate_peak_pix = np.array([500000.0, 250000.0, 80000.0])  # e-/s on central peak pixel

# Assume 85% aperture extraction enclosed signal
rate_extracted = rate_peak_pix / 0.30 * 0.85

print("=====================================================================")
print("          ExoETC: Detector Module Diagnostic Verification            ")
print("=====================================================================\n")

for ng_cmd in [1, 2, 3]:
    print(f"--- RUNNING WITH COMMANDED GROUPS: n_g = {ng_cmd} ---")
    
    # Evaluate saturation truncation
    n_eff, hard_sat = detector.evaluate_saturation_and_groups(rate_peak_pix, ng_cmd)
    
    # Compute timing and duty cycle
    t_int, t_cycle, duty_cycle = detector.compute_cycle_timing(n_eff)
    
    # Extract electrons per integration
    signal_per_int = rate_extracted * t_int
    
    # Compute single-ramp variance (using 1 spatial pixel, negligible background)
    var_int = detector.evaluate_noise_variance(
        n_e_signal=signal_per_int,
        t_int=t_int,
        n_eff=n_eff,
        n_extracted_pix=1.0,
        rate_bg_e_per_s=0.0
    )
    
    snr_per_int = signal_per_int / np.sqrt(var_int)
    
    for i, w in enumerate(wavelengths):
        regime = "Track B (kTC)" if n_eff[i] == 1 else "Track A (CDS)"
        print(f"  λ = {w:.2f} µm | n_eff: {n_eff[i]} | t_int: {t_int[i]:.4f}s | "
              f"Duty: {duty_cycle[i]*100:.1f}% | Mode: {regime:<13} | "
              f"Signal: {signal_per_int[i]:8.1f} e- | SNR/int: {snr_per_int[i]:6.1f}")
    
    summary = detector.format_status_summary(wavelengths, n_eff, hard_sat)
    print(f"  Summary -> Hard Saturated: {summary['hard_saturated_channels']}, "
          f"Track B: {summary['track_b_channels (n_eff=1)']}, "
          f"Track A: {summary['track_a_channels (n_eff>=2)']}\n")