"""
mediator.py

Centralized Parameter Broker and Component Mediator for ExoETC (ExoWorlds mission).
Implements the decoupled Mediator behavioral pattern following the architecture
of pyEDITH (Alei et al. 2026), preventing circular imports and rigid subsystem coupling.
"""

from typing import Dict, Any, Optional
import numpy as np


class ExoETCMediator:
    """
    Central mediator managing decoupled parameter exchange across the
    ExoWorlds ETC subsystems (Scene, Optics, Disperser, Detector, Observation).
    """

    def __init__(self):
        self._scene: Optional[Any] = None
        self._optics: Optional[Any] = None
        self._disperser: Optional[Any] = None
        self._detector: Optional[Any] = None
        self._observation: Optional[Any] = None
        self._custom_registry: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Subsystem Registration Interface
    # -------------------------------------------------------------------------

    def register_scene(self, scene: Any) -> None:
        """Registers the astrophysical target scene component (stellar & planetary)."""
        self._scene = scene

    def register_optics(self, optics: Any) -> None:
        """Registers the telescope and optical train component."""
        self._optics = optics

    def register_disperser(self, disperser: Any) -> None:
        """Registers the spectrometer disperser component (prism or grating)."""
        self._disperser = disperser

    def register_detector(self, detector: Any) -> None:
        """Registers the focal plane detector component."""
        self._detector = detector

    def register_observation(self, observation: Any) -> None:
        """Registers the observational strategy component (timing and geometries)."""
        self._observation = observation

    def set_custom_parameter(self, key: str, value: Any) -> None:
        """Stores arbitrary mission or override parameters in the shared registry."""
        self._custom_registry[key] = value

    # -------------------------------------------------------------------------
    # Scene Parameter Accessors
    # -------------------------------------------------------------------------

    def get_scene_parameter(self, param_name: str) -> Any:
        """Retrieves parameters from the registered Scene module."""
        if self._scene is None:
            raise RuntimeError("Cannot request scene parameter: 'Scene' is not registered with Mediator.")
        if hasattr(self._scene, param_name):
            return getattr(self._scene, param_name)
        elif isinstance(self._scene, dict) and param_name in self._scene:
            return self._scene[param_name]
        else:
            raise KeyError(f"Parameter '{param_name}' not found on registered Scene.")

    # -------------------------------------------------------------------------
    # Optics Parameter Accessors
    # -------------------------------------------------------------------------

    def get_optics_parameter(self, param_name: str) -> Any:
        """Retrieves parameters from the registered Optics module."""
        if self._optics is None:
            raise RuntimeError("Cannot request optics parameter: 'Optics' is not registered with Mediator.")
        if hasattr(self._optics, param_name):
            return getattr(self._optics, param_name)
        elif isinstance(self._optics, dict) and param_name in self._optics:
            return self._optics[param_name]
        else:
            raise KeyError(f"Parameter '{param_name}' not found on registered Optics.")

    # -------------------------------------------------------------------------
    # Disperser Parameter Accessors
    # -------------------------------------------------------------------------

    def get_disperser_parameter(self, param_name: str) -> Any:
        """Retrieves parameters from the registered Disperser module."""
        if self._disperser is None:
            raise RuntimeError("Cannot request disperser parameter: 'Disperser' is not registered with Mediator.")
        if hasattr(self._disperser, param_name):
            return getattr(self._disperser, param_name)
        elif isinstance(self._disperser, dict) and param_name in self._disperser:
            return self._disperser[param_name]
        else:
            raise KeyError(f"Parameter '{param_name}' not found on registered Disperser.")

    # -------------------------------------------------------------------------
    # Detector Parameter Accessors
    # -------------------------------------------------------------------------

    def get_detector_parameter(self, param_name: str) -> Any:
        """Retrieves parameters from the registered Detector module."""
        if self._detector is None:
            raise RuntimeError("Cannot request detector parameter: 'Detector' is not registered with Mediator.")
        if hasattr(self._detector, param_name):
            return getattr(self._detector, param_name)
        elif isinstance(self._detector, dict) and param_name in self._detector:
            return self._detector[param_name]
        else:
            raise KeyError(f"Parameter '{param_name}' not found on registered Detector.")

    # -------------------------------------------------------------------------
    # Observation Parameter Accessors
    # -------------------------------------------------------------------------

    def get_observation_parameter(self, param_name: str) -> Any:
        """Retrieves parameters from the registered Observation module."""
        if self._observation is None:
            raise RuntimeError("Cannot request observation parameter: 'Observation' is not registered with Mediator.")
        if hasattr(self._observation, param_name):
            return getattr(self._observation, param_name)
        elif isinstance(self._observation, dict) and param_name in self._observation:
            return self._observation[param_name]
        else:
            raise KeyError(f"Parameter '{param_name}' not found on registered Observation.")

    # -------------------------------------------------------------------------
    # System-Level Derived Calculations
    # -------------------------------------------------------------------------

    def get_incident_photon_rate(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Calculates the monochromatic photon arrival rate Φ_pix(λ) incident
        on a detector resolution element by negotiating parameters across modules:
          - Flux density F*(λ) from Scene
          - Aperture A_eff & Telescope throughput from Optics
          - Wavelength dispersion Δλ_pix(λ) and Disperser throughput from Disperser
        """
        waves = np.asarray(wavelengths, dtype=np.float64)

        # 1. Fetch physical and optical parameters via mediator
        f_lambda = self.get_scene_parameter("get_stellar_flux")(waves)  # W / m^2 / um
        a_eff = self.get_optics_parameter("collecting_area")            # m^2
        t_optics = self.get_optics_parameter("get_throughput")(waves)    # 0 to 1

        # 2. Fetch dispersion and element efficiency
        dlambda_pix = self.get_disperser_parameter("get_dispersion_per_pixel")(waves) # um / pix
        t_disp = self.get_disperser_parameter("get_throughput")(waves)                # 0 to 1

        # 3. Fundamental constants
        h = 6.62607015e-34  # J * s
        c = 299792458.0     # m / s
        energy_photon = (h * c) / (waves * 1e-6)  # Joules

        # 4. Total throughput excluding detector QE
        pce_optical = t_optics * t_disp

        # 5. Incident photon arrival rate (photons/s per pixel column)
        rate_photons_pix = (f_lambda * a_eff * dlambda_pix * pce_optical) / energy_photon
        return rate_photons_pix

    def get_status_overview(self) -> Dict[str, Any]:
        """Provides an inspection summary of registered components."""
        return {
            "Scene_Registered": self._scene is not None,
            "Optics_Registered": self._optics is not None,
            "Disperser_Registered": self._disperser is not None,
            "Detector_Registered": self._detector is not None,
            "Observation_Registered": self._observation is not None,
            "Custom_Parameters_Count": len(self._custom_registry)
        }

import numpy as np
from mediator import ExoETCMediator
from disperser import DisperserFactory
from detector import ExoWorldsDetector

# 1. Create Mock Subsystems for Scene and Optics to test mediator brokering
class MockTargetScene:
    """Mock Scene providing stellar flux for GJ 1214 (M-dwarf, J=9.75)."""
    def __init__(self, target_name="GJ 1214 b"):
        self.target_name = target_name
        self.transit_duration_s = 52.0 * 60.0  # 3120 s

    def get_stellar_flux(self, waves: np.ndarray) -> np.ndarray:
        # Approximate blackbody-like SED for Teff=3000K normalized at J-band (W/m^2/um)
        return 1.2e-11 * (waves / 1.25)**(-2.5)

class MockTelescopeOptics:
    """Mock Optics representing the ExoWorlds 2.0-meter space telescope."""
    def __init__(self, diameter_m=2.0, obscuration=0.15):
        self.diameter = diameter_m
        self.collecting_area = (np.pi * (diameter_m / 2.0)**2) * (1.0 - obscuration)

    def get_throughput(self, waves: np.ndarray) -> np.ndarray:
        # Optical train throughput (primary + secondary + fold mirrors)
        return np.full_like(waves, 0.88)

# 2. Instantiate all real and mock modules
mediator = ExoETCMediator()
scene = MockTargetScene()
optics = MockTelescopeOptics(diameter_m=2.0)
disperser = DisperserFactory.create("PRISM")
detector = ExoWorldsDetector(full_well_electrons=77000.0, t_frame=0.0737)

# 3. Register components with the mediator
mediator.register_scene(scene)
mediator.register_optics(optics)
mediator.register_disperser(disperser)
mediator.register_detector(detector)

print("===========================================================================")
print("             ExoETC: Mediator Module Verification Test Run                 ")
print("===========================================================================\n")

# Check registration status
status = mediator.get_status_overview()
print("Subsystem Registration Status:")
for comp, registered in status.items():
    print(f"  - {comp:<25}: {registered}")
print("\n" + "-" * 75)

# Test cross-component parameter retrieval
print("Testing Decoupled Parameter Retrieval via Mediator:")
t_name = mediator.get_scene_parameter("target_name")
t_dur = mediator.get_scene_parameter("transit_duration_s")
a_eff = mediator.get_optics_parameter("collecting_area")
d_name = mediator.get_disperser_parameter("name")
t_frame = mediator.get_detector_parameter("t_frame")

print(f"  Target: {t_name} (Transit Duration: {t_dur:.0f} s)")
print(f"  ExoWorlds Effective Aperture: {a_eff:.3f} m^2")
print(f"  Disperser Mode: {d_name}")
print(f"  Detector Frame Readout: {t_frame:.4f} s")
print("\n" + "-" * 75)

# 4. Test multi-subsystem mediated calculation: Incident Photon Rate
test_waves = np.array([0.80, 1.25, 2.20, 3.50, 5.00])
incident_photons = mediator.get_incident_photon_rate(test_waves)

print("Incident Photon Arrival Rate on Focal Plane (Brokered across Scene/Optics/Disperser):")
print(f"{'λ (µm)':<8} | {'Disp (nm/pix)':<14} | {'Flux (W/m²/µm)':<18} | {'Arrival Rate (e-/s/col)':<22}")
print("-" * 75)

dlambda_pix = disperser.get_dispersion_per_pixel(test_waves) * 1e3
f_star = scene.get_stellar_flux(test_waves)

for i, w in enumerate(test_waves):
    print(f"{w:<8.2f} | {dlambda_pix[i]:<14.2f} | {f_star[i]:<18.4e} | {incident_photons[i]:<22.2f}")

print("\nDiagnostic complete: Mediator successfully brokered all cross-module transactions.")