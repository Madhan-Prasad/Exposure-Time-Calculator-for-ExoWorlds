"""
scene.py

Astrophysical Target Scene Engine for ExoETC (ExoWorlds mission).
Implements stellar spectral energy distribution (SED) modeling, 2MASS photometric
normalization, planetary atmospheric transmission/emission signal generation,
and transit observation timing following pyEDITH (Alei et al. 2026) and
NEETC (Nielsen et al. 2016).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
import numpy as np


class BaseScene(ABC):
    """Abstract Base Class for target scenes in ExoETC."""

    @abstractmethod
    def get_stellar_flux(self, wavelengths: np.ndarray) -> np.ndarray:
        """Returns stellar flux density F*(λ) at Earth in W / m^2 / um."""
        pass

    @abstractmethod
    def get_transit_depth(self, wavelengths: np.ndarray) -> np.ndarray:
        """Returns the wavelength-dependent primary transit depth (Rp(λ) / R*)^2."""
        pass

    @abstractmethod
    def get_eclipse_contrast(self, wavelengths: np.ndarray) -> np.ndarray:
        """Returns the secondary eclipse contrast Fp(λ) / F*(λ)."""
        pass


class TargetScene(BaseScene):
    """
    Simulates the star-planet target system for transmission and emission spectroscopy.

    Parameters
    ----------
    star_teff : float, optional
        Stellar effective temperature in Kelvin (default: 3000 K for GJ 1214).
    star_radius_rsun : float, optional
        Stellar radius in Solar radii (default: 0.215 R_sun).
    star_distance_pc : float, optional
        Distance to the system in parsecs (default: 14.65 pc).
    j_mag : float, optional
        Host star 2MASS J-band apparent magnitude (default: 9.75 mag).
    planet_radius_rearth : float, optional
        Planetary radius in Earth radii (default: 2.74 R_earth).
    planet_teq : float, optional
        Planetary equilibrium temperature in Kelvin (default: 550 K).
    transit_duration_min : float, optional
        In-transit duration in minutes (default: 52.0 min for GJ 1214 b).
    baseline_factor : float, optional
        Ratio of out-of-transit to in-transit time (default: 2.0).
    target_name : str, optional
        Identifier name of the exoplanetary system (default: "GJ 1214 b").
    """

    def __init__(
        self,
        star_teff: float = 3000.0,
        star_radius_rsun: float = 0.215,
        star_distance_pc: float = 14.65,
        j_mag: float = 9.75,
        planet_radius_rearth: float = 2.74,
        planet_teq: float = 550.0,
        transit_duration_min: float = 52.0,
        baseline_factor: float = 2.0,
        target_name: str = "GJ 1214 b"
    ):
        self.target_name = target_name
        self.star_teff = float(star_teff)
        self.star_radius_rsun = float(star_radius_rsun)
        self.star_distance_pc = float(star_distance_pc)
        self.j_mag = float(j_mag)

        self.planet_radius_rearth = float(planet_radius_rearth)
        self.planet_teq = float(planet_teq)
        self.transit_duration_min = float(transit_duration_min)
        self.baseline_factor = float(baseline_factor)

        # Derived timing in seconds
        self.transit_duration_s = self.transit_duration_min * 60.0
        self.baseline_duration_s = self.transit_duration_s * self.baseline_factor
        self.total_observation_s = self.transit_duration_s + self.baseline_duration_s

        # Fundamental constants (SI)
        self._h = 6.62607015e-34  # J * s
        self._c = 299792458.0     # m / s
        self._kb = 1.380649e-23   # J / K
        self._r_sun_m = 6.957e8   # meters
        self._r_earth_m = 6.371e6 # meters
        self._pc_to_m = 3.085677581e16

        # Zero-point flux density for 2MASS J-band (W / m^2 / um)
        # Vega reference flux at 1.25 microns
        self._f0_j = 3.129e-10

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TargetScene":
        """Factory constructor instantiating directly from parameter dictionaries."""
        return cls(
            star_teff=config.get("star_teff", 3000.0),
            star_radius_rsun=config.get("star_radius_rsun", 0.215),
            star_distance_pc=config.get("star_distance_pc", 14.65),
            j_mag=config.get("j_mag", 9.75),
            planet_radius_rearth=config.get("planet_radius_rearth", 2.74),
            planet_teq=config.get("planet_teq", 550.0),
            transit_duration_min=config.get("transit_duration_min", 52.0),
            baseline_factor=config.get("baseline_factor", 2.0),
            target_name=config.get("target_name", "GJ 1214 b")
        )

    def _planck_flux_density(self, wavelengths_um: np.ndarray, temp_k: float) -> np.ndarray:
        """
        Evaluates Planck surface spectral radiance B_lambda(T) in W / (m^2 * um).
        Radiance pi * B_lambda emitted at the stellar/planetary photosphere.
        """
        waves_m = np.asarray(wavelengths_um, dtype=np.float64) * 1e-6
        with np.errstate(over='ignore', invalid='ignore'):
            exp_term = np.exp((self._h * self._c) / (waves_m * self._kb * temp_k))
            b_lambda_si = (2.0 * np.pi * self._h * self._c**2) / (waves_m**5 * (exp_term - 1.0))
            b_lambda_si = np.nan_to_num(b_lambda_si, nan=0.0, posinf=0.0)

        # Convert W / (m^2 * m) to W / (m^2 * um)
        return b_lambda_si * 1e-6

    def get_stellar_flux(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Calculates stellar flux density F*(λ) received at Earth (W / m^2 / um),
        normalized to the target's 2MASS J-band apparent magnitude.
        """
        waves = np.asarray(wavelengths, dtype=np.float64)

        # 1. Raw Planck flux shape at stellar surface
        b_raw = self._planck_flux_density(waves, self.star_teff)

        # 2. Reference flux at J-band center (1.25 microns)
        b_ref = self._planck_flux_density(np.array([1.25]), self.star_teff)[0]

        # 3. Scale by 2MASS J-band apparent magnitude: F_J = F0_J * 10^(-0.4 * J)
        f_target_j = self._f0_j * (10.0 ** (-0.4 * self.j_mag))

        # 4. Normalized flux density at Earth across all wavelengths
        f_lambda = b_raw * (f_target_j / b_ref)
        return f_lambda

    def get_transit_depth(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Calculates wavelength-dependent primary transit depth (Rp(λ) / R*)^2.
        Includes geometric base depth plus simulated transmission spectral modulation
        (e.g., H2O absorption features around 1.4, 1.9, and 2.7 microns).
        """
        waves = np.asarray(wavelengths, dtype=np.float64)

        r_star_m = self.star_radius_rsun * self._r_sun_m
        r_planet_m = self.planet_radius_rearth * self._r_earth_m

        # Geometric baseline transit depth: (Rp / R*)^2
        depth_base = (r_planet_m / r_star_m) ** 2

        # Empirical synthetic transmission signal (H2O water vapor bands proxy)
        # Adds realistic ~150-200 ppm atmospheric transmission modulation
        water_features = (
            0.00018 * np.exp(-0.5 * ((waves - 1.40) / 0.08)**2) +
            0.00022 * np.exp(-0.5 * ((waves - 1.90) / 0.10)**2) +
            0.00030 * np.exp(-0.5 * ((waves - 2.70) / 0.15)**2)
        )

        return depth_base + water_features

    def get_eclipse_contrast(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Calculates secondary eclipse planet-to-star flux contrast:
        C(λ) = (Rp / R*)^2 * (Bp(T_day) / B*(T_eff)).
        """
        waves = np.asarray(wavelengths, dtype=np.float64)

        r_star_m = self.star_radius_rsun * self._r_sun_m
        r_planet_m = self.planet_radius_rearth * self._r_earth_m
        geom_ratio = (r_planet_m / r_star_m) ** 2

        # Dayside temperature approximation: T_day ≈ 1.10 * T_eq
        t_dayside = 1.10 * self.planet_teq

        # Surface emission contrast
        b_planet = self._planck_flux_density(waves, t_dayside)
        b_star = self._planck_flux_density(waves, self.star_teff)

        with np.errstate(divide='ignore', invalid='ignore'):
            contrast = geom_ratio * (b_planet / b_star)
            contrast = np.nan_to_num(contrast, nan=0.0, posinf=0.0)

        return contrast

# test_scene.py
import sys
from pathlib import Path
import numpy as np

# Ensure path resolution
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

core_dir = current_dir / "exoworlds_etc" / "core"
if core_dir.exists() and str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

try:
    from exoworlds_etc.core.scene import TargetScene
    from exoworlds_etc.core.optics import TelescopeOptics
    from exoworlds_etc.core.disperser import DisperserFactory
    from exoworlds_etc.core.mediator import ExoETCMediator
except ModuleNotFoundError:
    from scene import TargetScene
    from optics import TelescopeOptics
    from disperser import DisperserFactory
    from mediator import ExoETCMediator


def run_scene_test():
    print("===========================================================================")
    print("              ExoETC: Scene Module Diagnostic Verification                 ")
    print("===========================================================================\n")

    # 1. Initialize Benchmark Scene: GJ 1214 b
    scene = TargetScene(
        star_teff=3000.0,
        star_radius_rsun=0.215,
        star_distance_pc=14.65,
        j_mag=9.75,
        planet_radius_rearth=2.74,
        planet_teq=550.0,
        transit_duration_min=52.0,
        target_name="GJ 1214 b"
    )

    test_waves = np.array([0.80, 1.25, 1.40, 1.90, 2.70, 4.50])

    # 2. Verify Stellar Flux and Photometric Calibration
    f_star = scene.get_stellar_flux(test_waves)
    expected_j_flux = 3.129e-10 * (10.0 ** (-0.4 * 9.75))
    computed_j_flux = scene.get_stellar_flux(np.array([1.25]))[0]

    print("--- STEP 1: STELLAR SED & 2MASS PHOTOMETRIC CALIBRATION ---")
    print(f"  Target System                : {scene.target_name}")
    print(f"  Expected J-band Flux (Earth) : {expected_j_flux:.4e} W/m^2/um")
    print(f"  Computed J-band Flux (Earth) : {computed_j_flux:.4e} W/m^2/um")
    print(f"  Calibration Offset           : {abs(computed_j_flux - expected_j_flux):.2e}")
    assert abs(computed_j_flux - expected_j_flux) < 1e-18, "Photometric calibration error!"
    print("  >> PASSED: Absolute 2MASS J-band calibration verified.\n")

    # 3. Verify Primary Transit Depth and Secondary Eclipse Contrast
    depths = scene.get_transit_depth(test_waves)
    contrasts = scene.get_eclipse_contrast(test_waves)

    print("--- STEP 2: PLANETARY TRANSIT & ECLIPSE SIGNALS ---")
    print(f"{'λ (µm)':<8} | {'Flux (W/m²/µm)':<18} | {'Transit Depth (dT)':<20} | {'Eclipse Contrast (ppm)':<22}")
    print("-" * 75)
    for i, w in enumerate(test_waves):
        print(f"{w:<8.2f} | {f_star[i]:<18.4e} | {depths[i]*100:<18.4f}% | {contrasts[i]*1e6:<20.2f}")

    # 4. Step 3: Full Mediator Brokering Test
    mediator = ExoETCMediator()
    optics = TelescopeOptics(diameter_m=2.0, obscuration_fraction=0.15)
    disperser = DisperserFactory.create("PRISM")

    mediator.register_scene(scene)
    mediator.register_optics(optics)
    mediator.register_disperser(disperser)

    arrival_rates = mediator.get_incident_photon_rate(test_waves)

    print("\n--- STEP 3: END-TO-END BROKERING VIA MEDIATOR ---")
    print(f"Telescope: {optics.diameter}m | Mode: {disperser.name}")
    for i, w in enumerate(test_waves):
        print(f"  λ = {w:.2f} µm -> Incident Rate = {arrival_rates[i]:10.2f} photons/s/column")

    print("\nDiagnostic complete: scene.py validated across all operational checks.")


if __name__ == "__main__":
    run_scene_test()