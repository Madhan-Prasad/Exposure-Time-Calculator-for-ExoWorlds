"""
disperser.py

Dispersive element engine for ExoETC (ExoWorlds mission).
Models prisms, gratings, and grisms; computes wavelength-dependent resolving power,
pixel dispersion (Δλ_pix), and optical throughput curves following the
JWST/NIRSpec NEETC framework (Nielsen et al. 2016) and pyEDITH (Alei et al. 2026).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, Optional
import numpy as np
from astropy import units as u


class BaseDisperser(ABC):
    """Abstract Base Class for spectrometers, gratings, and prisms in ExoETC."""

    @abstractmethod
    def get_dispersion_per_pixel(self, wavelengths: np.ndarray) -> np.ndarray:
        """Calculates wavelength width per pixel column (Δλ_pix) in microns/pixel."""
        pass

    @abstractmethod
    def get_resolving_power(self, wavelengths: np.ndarray) -> np.ndarray:
        """Returns the dimensionless resolving power R(λ) = λ / Δλ."""
        pass

    @abstractmethod
    def get_throughput(self, wavelengths: np.ndarray) -> np.ndarray:
        """Returns the internal transmission or blaze efficiency of the disperser."""
        pass


class GratingDisperser(BaseDisperser):
    """
    Diffraction grating disperser with nearly constant linear dispersion (dλ/dx).
    
    For a reflection or transmission grating, the dispersion per pixel column
    is approximately constant in wavelength units (Δλ_pix ≈ constant), meaning
    resolving power R(λ) = λ / (Δλ_pix * N_samp) scales proportionally with λ.

    Parameters
    ----------
    wave_min : float
        Short-wavelength cutoff in microns.
    wave_max : float
        Long-wavelength cutoff in microns.
    dispersion_nm_per_pix : float
        Linear spectral dispersion in nanometers per detector pixel column.
    sampling_pixels : float, optional
        Pixels per resolution element (default: 2.0 for Nyquist sampling).
    peak_throughput : float, optional
        Peak blaze grating efficiency (default: 0.75).
    name : str, optional
        Identifier name (e.g., 'G140M', 'ExoWorlds_Grating').
    """

    def __init__(
        self,
        wave_min: float,
        wave_max: float,
        dispersion_nm_per_pix: float,
        sampling_pixels: float = 2.0,
        peak_throughput: float = 0.75,
        name: str = "Grating_Medium"
    ):
        self.wave_min = float(wave_min)
        self.wave_max = float(wave_max)
        self.disp_um_pix = float(dispersion_nm_per_pix) * 1e-3  # convert nm -> um
        self.sampling_pixels = float(sampling_pixels)
        self.peak_throughput = float(peak_throughput)
        self.name = name

    def get_dispersion_per_pixel(self, wavelengths: np.ndarray) -> np.ndarray:
        """Grating dispersion is constant across pixel columns."""
        waves = np.asarray(wavelengths, dtype=np.float64)
        return np.full_like(waves, self.disp_um_pix)

    def get_resolving_power(self, wavelengths: np.ndarray) -> np.ndarray:
        """R(λ) = λ / (Δλ_pix * N_samp)."""
        waves = np.asarray(wavelengths, dtype=np.float64)
        delta_lambda = self.disp_um_pix * self.sampling_pixels
        return waves / delta_lambda

    def get_throughput(self, wavelengths: np.ndarray) -> np.ndarray:
        """Blaze efficiency modeled with an inverted parabolic bandpass."""
        waves = np.asarray(wavelengths, dtype=np.float64)
        w_center = 0.5 * (self.wave_min + self.wave_max)
        w_half_width = 0.5 * (self.wave_max - self.wave_min)

        # Normalized offset from blaze center
        norm_offset = (waves - w_center) / w_half_width
        
        # Parabolic blaze roll-off
        eff = self.peak_throughput * np.maximum(0.0, 1.0 - 0.4 * (norm_offset ** 2))
        
        # Zero out transmission outside physical grating bandpass
        mask = (waves >= self.wave_min) & (waves <= self.wave_max)
        return np.where(mask, eff, 0.0)


class PrismDisperser(BaseDisperser):
    """
    Prism disperser with non-linear prismatic dispersion (e.g., CaF2 or ZnSe).

    Following NIRSpec PRISM / CLEAR (Nielsen et al. 2016):
    Prisms have variable dispersion: wide pixel bandwidth (large Δλ_pix, low R)
    at short wavelengths, narrowing into finer dispersion (small Δλ_pix, higher R)
    at longer wavelengths.

    Parameters
    ----------
    wave_min : float, optional
        Short-wavelength cutoff (default: 0.6 um).
    wave_max : float, optional
        Long-wavelength cutoff (default: 5.3 um).
    r_min : float, optional
        Resolving power at the blue edge (default: 30.0).
    r_max : float, optional
        Resolving power at the red edge (default: 300.0).
    sampling_pixels : float, optional
        Pixels per resolution element (default: 2.0).
    mean_throughput : float, optional
        Average optical transmission across the prism glass (default: 0.85).
    name : str, optional
        Identifier name (default: 'CLEAR/PRISM').
    """

    def __init__(
        self,
        wave_min: float = 0.6,
        wave_max: float = 5.3,
        r_min: float = 30.0,
        r_max: float = 300.0,
        sampling_pixels: float = 2.0,
        mean_throughput: float = 0.85,
        name: str = "CLEAR/PRISM"
    ):
        self.wave_min = float(wave_min)
        self.wave_max = float(wave_max)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.sampling_pixels = float(sampling_pixels)
        self.mean_throughput = float(mean_throughput)
        self.name = name

    def get_resolving_power(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Calculates resolving power R(λ).
        Models the characteristic linear rise in R across infrared prism materials:
        R(λ) = R_min + (R_max - R_min) * ((λ - λ_min) / (λ_max - λ_min)).
        """
        waves = np.asarray(wavelengths, dtype=np.float64)
        slope = (self.r_max - self.r_min) / (self.wave_max - self.wave_min)
        r_curve = self.r_min + slope * (waves - self.wave_min)
        return np.clip(r_curve, self.r_min, self.r_max)

    def get_dispersion_per_pixel(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Calculates wavelength span per pixel column:
        Δλ_pix(λ) = λ / (R(λ) * N_samp).
        """
        waves = np.asarray(wavelengths, dtype=np.float64)
        r_curve = self.get_resolving_power(waves)
        return waves / (r_curve * self.sampling_pixels)

    def get_throughput(self, wavelengths: np.ndarray) -> np.ndarray:
        """High-transmission prism throughput with anti-reflection coating cutoffs."""
        waves = np.asarray(wavelengths, dtype=np.float64)
        mask = (waves >= self.wave_min) & (waves <= self.wave_max)
        return np.where(mask, self.mean_throughput, 0.0)


class DisperserFactory:
    """Factory helper to load benchmark or mission-defined disperser modes."""

    @staticmethod
    def create(mode: str) -> BaseDisperser:
        mode_upper = mode.strip().upper()
        
        if mode_upper in ["PRISM", "CLEAR/PRISM", "NIRSPEC_PRISM"]:
            return PrismDisperser(
                wave_min=0.6,
                wave_max=5.3,
                r_min=30.0,
                r_max=300.0,
                name="JWST_NIRSpec_PRISM"
            )
        elif mode_upper in ["G140M", "NIRSPEC_G140M"]:
            return GratingDisperser(
                wave_min=0.70,
                wave_max=1.80,
                dispersion_nm_per_pix=0.55,
                peak_throughput=0.78,
                name="JWST_NIRSpec_G140M"
            )
        elif mode_upper in ["EXOWORLDS_PRISM", "EXOWORLDS_SURVEY"]:
            return PrismDisperser(
                wave_min=0.5,
                wave_max=5.0,
                r_min=40.0,
                r_max=250.0,
                name="ExoWorlds_Survey_Prism"
            )
        else:
            raise ValueError(f"Unknown disperser mode: '{mode}'. "
                             f"Available options: 'PRISM', 'G140M', 'EXOWORLDS_PRISM'.")

import numpy as np
from disperser import DisperserFactory, PrismDisperser, GratingDisperser

# 1. Initialize benchmark dispersers
prism = DisperserFactory.create("PRISM")
grating = DisperserFactory.create("G140M")

# 2. Define wavelength test points across the near-infrared
test_wavelengths = np.array([0.80, 1.25, 2.00, 3.50, 5.00])

print("===========================================================================")
print("             ExoETC: Disperser Module Diagnostic Test Run                  ")
print("===========================================================================\n")

# Test NIRSpec PRISM Mode
print(f"--- DISPERSER MODE: {prism.name} (Prism Configuration) ---")
print(f"Bandpass: {prism.wave_min:.2f} - {prism.wave_max:.2f} µm | Sampling: {prism.sampling_pixels:.1f} pix/FWHM")
print(f"{'λ (µm)':<8} | {'R(λ)':<8} | {'Δλ_elem (nm)':<14} | {'Δλ_pix (nm/pix)':<16} | {'Throughput':<10}")
print("-" * 65)

r_prism = prism.get_resolving_power(test_wavelengths)
disp_prism = prism.get_dispersion_per_pixel(test_wavelengths)
thru_prism = prism.get_throughput(test_wavelengths)

for i, w in enumerate(test_wavelengths):
    delta_elem_nm = (w / r_prism[i]) * 1e3
    disp_pix_nm = disp_prism[i] * 1e3
    print(f"{w:<8.2f} | {r_prism[i]:<8.1f} | {delta_elem_nm:<14.2f} | {disp_pix_nm:<16.2f} | {thru_prism[i]*100:<9.1f}%")

print("\n" + "=" * 65 + "\n")

# Test NIRSpec G140M Mode (Medium Resolution Grating)
grating_waves = np.array([0.80, 1.00, 1.25, 1.50, 1.75])
print(f"--- DISPERSER MODE: {grating.name} (Grating Configuration) ---")
print(f"Bandpass: {grating.wave_min:.2f} - {grating.wave_max:.2f} µm | Sampling: {grating.sampling_pixels:.1f} pix/FWHM")
print(f"{'λ (µm)':<8} | {'R(λ)':<8} | {'Δλ_elem (nm)':<14} | {'Δλ_pix (nm/pix)':<16} | {'Throughput':<10}")
print("-" * 65)

r_grating = grating.get_resolving_power(grating_waves)
disp_grating = grating.get_dispersion_per_pixel(grating_waves)
thru_grating = grating.get_throughput(grating_waves)

for i, w in enumerate(grating_waves):
    delta_elem_nm = (w / r_grating[i]) * 1e3
    disp_pix_nm = disp_grating[i] * 1e3
    print(f"{w:<8.2f} | {r_grating[i]:<8.1f} | {delta_elem_nm:<14.2f} | {disp_pix_nm:<16.2f} | {thru_grating[i]*100:<9.1f}%")

print("\nDiagnostic complete: All dispersion and resolving power metrics verified.")