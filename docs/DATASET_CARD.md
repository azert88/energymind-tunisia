# EnergyMind Tunisia: clean smart-metered LV neighbourhood baseline

Dataset: https://www.kaggle.com/datasets/azertyqsdfghqsdfgh/pes-test

TSYP14 Technical Challenge, Phase 1. This is the clean, anomaly-free baseline for the EnergyMind trust layer. **It contains no injected anomalies.** Anomalies are injected on top of it by the project code, so every corrupted sample has a known label.

## Real-data foundation
Consumption behaviour comes from two real, measured datasets. Only the grid physics, the PV and the meter model are simulated.

- **WPuQ:** real three-phase smart-meter measurements from German single-family houses. Sets the load shapes, phase split, and meter noise and gain (`meter_calibration.json`).
- **AMPds2:** real appliance sub-metering from one Canadian house (2012-2014). Reused as the equipment of each simulated house.
- **SimBench** `1-LV-rural1--0-no_sw`: LV grid topology. Solved with pandapower AC power flow.
- **Simulated:** rooftop PV from Tunis climatology, and the Tunisian adaptation (evening peak +1 h, heating turned into summer air-conditioning, Tunis location).

Real STEG data is not publicly available at this resolution, so a Tunisian-adapted real-data baseline was built instead.

## Scope
13 households on one 160 kVA transformer, 14 LV buses, one year (2018), 35,040 steps at 15 minutes, timezone Africa/Tunis.

## Two views of the same data
- `ground_truth/`: true physical values from power flow.
- `measured/`: what a healthy smart meter reports (normal gain error and noise calibrated on real WPuQ meters).

## Files
```
output/
  clean/
    all_houses_measured.parquet      all 13 houses, measured view
    measured/house_01..13.csv        smart-meter view
    ground_truth/house_01..13.csv    true values (+ P_load_W)
    house_inventory.csv              bus, rated kW, PV kWp, WPuQ id, appliances, phase map, kWh/day
    grid_bus_voltage_pu.csv          bus voltages (per-unit)
    grid_line_loading_percent_of_rated_current.csv
    grid_transformer.csv             loading, P/Q, losses
  figures/                           10 diagnostic PNGs
  meter_calibration.json             meter noise and gain model fitted on WPuQ
  validation_report.json             14 physical checks, all passed
```

## Columns (measured and ground_truth)
`timestamp` (15-min steps, 2018-01-01 to 2018-12-31); `U_1..3` (V); `I_1..3` (A); `P_1..3`, `Q_1..3`, `S_1..3` (W, var, VA); `PF_1..3`; `P_TOT`, `Q_TOT`, `S_TOT`, `PF_TOT`; `P_PV_W` (PV generation); `app_<name>_W` (per-appliance power); `P_load_W` (true load, ground truth only).

## Validation
All 14 checks pass: no NaN or Inf, uniform time axis, all power flows converged, voltage within EN 50160 +/-10 % (100 % of samples within +/-5 %), max voltage unbalance 1.03 %, transformer peak loading 66.6 %, line peak loading 35.5 %, PV zero at night, transformer energy balance exact, no frozen sensors, daily energy 6.7-41.6 kWh per house (each house's mean daily energy over the year, from the mean of `P_load_W`; the range is across the 13 houses, not the min-max of individual days).

## Anomaly injection (project code, not in this dataset)
The seeded injector (`energymind/inject.py`, 16 subtypes, see `docs/INJECTION.md`) adds measurement problems (bias, drift, scale, noise, frozen, dropout, delay, replay, falsified), real events (voltage sag, outage, demand step, duty cycle, phase imbalance) and benign cases (new load, PV ramp), with labels.

## Adaptation to Tunisian conditions
| Tunisian condition | How the dataset reproduces it |
|---|---|
| Summer peak driven by air conditioning | Canadian heat-pump profiles shifted by 182 days |
| Daily rhythm | Evening peak shifted by one hour |
| Rooftop solar in Tunis | PV 30 deg, facing south, Tunis monthly clearness index and temperature |
| Grid | 230 V, 50 Hz, EN 50160 +/-10 %, SimBench rural network |
| Place and clock | Tunis latitude/longitude, time zone Africa/Tunis |

## Limitations
Limit: consumption levels (6.7-41.6 kWh/day per house) are higher than the Tunisian household average (about 1.6 MWh/year, STEG survey); calibration is planned for Phase 2. Grid-level electrical values and PV are simulated. German and Canadian behaviour is adapted to Tunisia by documented rules. One network topology, one year (2018). The Phase 1 detection results were measured on external recordings (ECO, AMPds2, WPuQ), not yet on this dataset.

## Sources
- SimBench (grid topology, ODbL), Meinecke et al., 2020.
- WPuQ, Schlemminger et al., Scientific Data, 2022 (CC BY 4.0).
- AMPds2, Makonin et al., Scientific Data, 2016 (CC BY 4.0).
- pandapower; EN 50160.

## License
Open Database License (ODbL). Rebuild with `python scripts/build_baseline_dataset.py --seed 42`.
