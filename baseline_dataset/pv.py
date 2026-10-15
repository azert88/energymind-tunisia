"""Tunisian PV generation: clear-sky + stochastic (but smooth) clearness, plane-of-array transposition."""
import numpy as np
import pandas as pd

from .config import Config

I0 = 1361.0  # solar constant W/m2


def solar_geometry(index: pd.DatetimeIndex, cfg: Config):
    """Zenith & incidence cosines for interval-centred times (local standard time UTC+1)."""
    t = index + pd.Timedelta(minutes=cfg.resolution_min / 2)
    n = t.dayofyear.to_numpy()
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60
    b = 2 * np.pi * (n - 1) / 365
    decl = (0.006918 - 0.399912 * np.cos(b) + 0.070257 * np.sin(b) - 0.006758 * np.cos(2 * b)
            + 0.000907 * np.sin(2 * b) - 0.002697 * np.cos(3 * b) + 0.00148 * np.sin(3 * b))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(b) - 0.032077 * np.sin(b)
                    - 0.014615 * np.cos(2 * b) - 0.04089 * np.sin(2 * b))          # minutes
    solar_h = hour + (4 * (cfg.longitude - 15.0) + eot) / 60                        # UTC+1 meridian = 15E
    w = np.radians(15 * (solar_h - 12))
    phi, beta, gam = np.radians(cfg.latitude), np.radians(cfg.pv_tilt_deg), np.radians(cfg.pv_azimuth_deg - 180)
    cosz = np.sin(decl) * np.sin(phi) + np.cos(decl) * np.cos(phi) * np.cos(w)
    cos_inc = (np.sin(decl) * np.sin(phi) * np.cos(beta) - np.sin(decl) * np.cos(phi) * np.sin(beta) * np.cos(gam)
               + np.cos(decl) * np.cos(phi) * np.cos(beta) * np.cos(w)
               + np.cos(decl) * np.sin(phi) * np.sin(beta) * np.cos(gam) * np.cos(w)
               + np.cos(decl) * np.sin(beta) * np.sin(gam) * np.sin(w))
    e0 = 1 + 0.033 * np.cos(2 * np.pi * n / 365)
    return cosz, np.clip(cos_inc, 0, None), e0


def pv_profile(index: pd.DatetimeIndex, cfg: Config, rng: np.random.Generator) -> dict:
    """Shared neighbourhood weather -> PV output per kWp (kW/kWp), POA irradiance, ambient/cell temperature."""
    cosz, cos_inc, e0 = solar_geometry(index, cfg)
    day = index.normalize()
    day_id = (day - day[0]).days.to_numpy()
    month = index.month.to_numpy() - 1
    sun_up = cosz > 0.02
    cz = np.where(sun_up, cosz, 1.0)

    ghi_cs = np.where(sun_up, 1098 * cosz * np.exp(-0.057 / cz), 0.0)                # Haurwitz clear sky
    # daily attenuation ~ Beta with monthly mean = kt_month / kt_clear
    kt = np.array(cfg.kt_month)
    mean_att = np.clip(kt / 0.76, 0.3, 0.97)
    n_days = day_id.max() + 1
    m_of_day = pd.Series(month).groupby(day_id).first().to_numpy()
    conc = 3.0
    a_d = rng.beta(mean_att[m_of_day] * conc, (1 - mean_att[m_of_day]) * conc)
    a_d = np.clip(a_d, 0.18, 1.0)
    # smooth intraday cloud passages (AR(1)), amplitude grows on non-clear days
    n = len(index)
    eps = rng.standard_normal(n)
    ar = np.zeros(n)
    rho = 0.85
    for i in range(1, n):
        ar[i] = rho * ar[i - 1] + np.sqrt(1 - rho ** 2) * eps[i]
    att = np.clip(a_d[day_id] * (1 + 0.12 * (1 - a_d[day_id]) * ar), 0.12, 1.0)
    ghi = ghi_cs * att

    # Erbs decomposition -> diffuse fraction, then isotropic transposition
    ghi_ext = I0 * e0 * cz
    kt_i = np.where(sun_up, np.clip(ghi / ghi_ext, 0, 1), 0)
    df = np.where(kt_i <= 0.22, 1 - 0.09 * kt_i,
                  np.where(kt_i <= 0.8, 0.9511 - 0.1604 * kt_i + 4.388 * kt_i ** 2 - 16.638 * kt_i ** 3 + 12.336 * kt_i ** 4, 0.165))
    dhi = ghi * df
    dni = np.where(cosz > 0.087, (ghi - dhi) / np.maximum(cosz, 0.087), 0.0)        # avoid horizon blow-up
    beta = np.radians(cfg.pv_tilt_deg)
    poa = dni * cos_inc + dhi * (1 + np.cos(beta)) / 2 + ghi * 0.2 * (1 - np.cos(beta)) / 2
    poa = np.where(sun_up, poa, 0.0)

    # ambient temperature: monthly mean + diurnal wave + slow daily anomaly
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    anomaly = rng.normal(0, 2.0, n_days)[day_id]
    tamb = np.array(cfg.tamb_month)[month] + 5.0 * np.sin(2 * np.pi * (hour - 9) / 24) + anomaly
    tcell = tamb + 0.03 * poa                                                         # NOCT-like
    kw_per_kwp = poa / 1000 * (1 + cfg.pv_temp_coeff * (tcell - 25)) * cfg.pv_perf_ratio
    return dict(kw_per_kwp=np.clip(kw_per_kwp, 0, 1.0), poa=poa, ghi=ghi, tamb=tamb, tcell=tcell, att_day=a_d)
