"""Central configuration of the EnergyMind Tunisia data-preparation pipeline."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- inputs -----------------------------------------------------------
    ampds2_dir: Path = ROOT / "dataverse_files"
    wpuq_dir: Path = ROOT / "wpuq_csv"
    simbench_code: str = "1-LV-rural1--0-no_sw"

    # ---- outputs ----------------------------------------------------------
    out_dir: Path = ROOT / "output"

    # ---- simulation -------------------------------------------------------
    year: int = 2018                 # same calendar as WPuQ (weekday alignment)
    days: int = 365                  # use a smaller value for quick tests
    resolution_min: int = 15         # native WPuQ resolution
    tz: str = "Africa/Tunis"         # UTC+1, no DST
    seed: int = 42

    # ---- Tunisian adaptation ---------------------------------------------
    latitude: float = 36.8           # Tunis
    longitude: float = 10.18
    evening_shift_h: float = 1.0     # Tunisian meals/evening peak ~1 h later than Vancouver
    cooling_season_shift_days: int = 182  # AMPds2 heat pump (heating, Canada) -> A/C (cooling, Tunisia)
    pv_tilt_deg: float = 30.0
    pv_azimuth_deg: float = 180.0    # south
    pv_perf_ratio: float = 0.82
    pv_temp_coeff: float = -0.004    # 1/K
    # monthly mean clearness index and air temperature for Tunis (climatology)
    kt_month: tuple = (0.50, 0.53, 0.57, 0.60, 0.65, 0.70, 0.74, 0.71, 0.64, 0.58, 0.52, 0.48)
    tamb_month: tuple = (11.5, 12.0, 14.0, 16.5, 20.5, 24.5, 27.5, 28.0, 25.5, 21.0, 16.0, 12.5)

    # ---- AMPds2 sub-meters used as house equipment -------------------------
    # code: (label, probability a house owns it)
    appliances: dict = field(default_factory=lambda: {
        "FGE": ("fridge", 1.00),
        "UNE": ("lighting_and_misc", 1.00),   # AMPds2 un-metered remainder: lighting + small loads
        "TVE": ("tv", 0.95),
        "EQE": ("network_equipment", 1.00),
        "B1E": ("bedroom_1_plugs", 1.00),
        "B2E": ("bedroom_2_plugs", 0.80),
        "DNE": ("dining_room_plugs", 0.85),
        "OFE": ("home_office", 0.55),
        "UTE": ("utility_room", 0.60),
        "CWE": ("washing_machine", 0.90),
        "WOE": ("oven", 0.70),
        "HTE": ("water_heater", 0.55),
        "HPE": ("air_conditioner", 0.85),
        "DWE": ("dishwasher", 0.30),
        "CDE": ("tumble_dryer", 0.10),
    })
    # appliances that follow the season shift (heating in Canada == cooling in Tunisia)
    seasonal_inverted: tuple = ("HPE",)
    # appliances whose timing is not tied to occupant habits (no evening shift)
    no_time_shift: tuple = ("FGE", "EQE")

    # ---- validation limits -------------------------------------------------
    v_min_pu: float = 0.90           # EN 50160: 230 V +/-10 %
    v_max_pu: float = 1.10
    max_trafo_loading_pct: float = 100.0
    max_line_loading_pct: float = 100.0

    @property
    def n_steps(self) -> int:
        return self.days * 24 * 60 // self.resolution_min

    @property
    def clean_dir(self) -> Path:
        return self.out_dir / "clean"

    @property
    def fig_dir(self) -> Path:
        return self.out_dir / "figures"
