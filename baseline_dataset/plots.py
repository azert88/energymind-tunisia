"""Figures for the clean dataset. Muted categorical palette, thin marks, recessive grid."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import wpuq

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
CAT = ["#2a6fbb", "#d9822b", "#3a9d6b", "#8e5bb5", "#c4453c", "#17a2b8", "#7a7a7a"]
GROUPS = {  # appliance group -> AMPds2 codes
    "Base load (fridge, network, lighting/misc)": ["FGE", "EQE", "UNE"],
    "Air conditioning": ["HPE"],
    "Cooking": ["WOE"],
    "Laundry & dishes": ["CWE", "DWE", "CDE"],
    "Water heater": ["HTE"],
    "Plugs & electronics": ["TVE", "OFE", "B1E", "B2E", "DNE", "UTE"],
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9, "legend.frameon": False,
        "lines.linewidth": 1.6,
    })


def _save(fig, ctx, name):
    fig.savefig(ctx["cfg"].fig_dir / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _day(ctx, month, weekday=True):
    """first date in `month` that exists in the simulated period (falls back to the first full weekday)."""
    idx = ctx["index"]
    days = idx.normalize().unique()
    for d in days:
        if d.month == month and (d.dayofweek < 5) == weekday:
            return d
    for d in days:
        if (d.dayofweek < 5) == weekday:
            return d
    return days[0]


def _grouped(ctx, k, day):
    """appliance power of house k on `day`, summed by group (kW)."""
    x = ctx["hl"][k]
    idx = ctx["index"]
    m = (idx.normalize() == day)
    out = {}
    for g, codes in GROUPS.items():
        cols = [j for j, c in enumerate(x.appliances) if c in codes]
        out[g] = x.P_app[m][:, cols].sum(1) / 1000 if cols else np.zeros(m.sum())
    return idx[m], out


def fig_network(ctx):
    net, houses, res = ctx["net"], ctx["houses"], ctx["res"]
    xy = np.array([[float(v) for v in __import__("json").loads(g)["coordinates"]] for g in net.bus.geo])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.grid(False)
    lmax = res["line_loading"].max(0)
    for li, l in net.line.iterrows():
        a, b = xy[l.from_bus], xy[l.to_bus]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=CAT[0], lw=1.2 + lmax[li] / 8, alpha=.8, solid_capstyle="round")
        ax.annotate(f"{lmax[li]:.0f}%", ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), fontsize=6.5, color=MUTED,
                    ha="center", va="bottom")
    ax.scatter(*xy.T, s=14, color=INK, zorder=3)
    t = net.trafo.iloc[0]
    ax.scatter(*xy[t.lv_bus], s=160, marker="s", color=CAT[4], zorder=4, label="MV/LV transformer 160 kVA")
    for _, h in houses.iterrows():
        p = xy[h.bus]
        ax.scatter(*p, s=120 if not h.has_pv else 220, marker="o" if not h.has_pv else "D",
                   facecolor="white" if not h.has_pv else CAT[1], edgecolor=INK, lw=1, zorder=5)
        ax.annotate(f"{h.house_id[-2:]}" + (f"\n{h.pv_kwp:.0f} kWp" if h.has_pv else ""), p, xytext=(7, 7),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.scatter([], [], marker="o", facecolor="white", edgecolor=INK, label="house (no PV)")
    ax.scatter([], [], marker="D", facecolor=CAT[1], edgecolor=INK, label="house with PV")
    ax.set_title(f"SimBench {ctx['cfg'].simbench_code}: 13 houses, 4 with PV\n(line label = peak loading over the period)")
    ax.legend(loc="best"); ax.set_xticks([]); ax.set_yticks([])
    _save(fig, ctx, "01_network_topology.png")


def fig_ampds2(ctx):
    pool, cfg = ctx["pool"], ctx["cfg"]
    spd = pool.P.shape[1]
    t = np.arange(spd) * cfg.resolution_min / 60
    fig, axes = plt.subplots(3, 5, figsize=(14, 7), sharex=True)
    for ax, (j, c) in zip(axes.ravel(), enumerate(pool.codes)):
        v = pool.P[pool.valid][:, :, j]
        wd = v[~pool.weekend[pool.valid]].mean(0); we = v[pool.weekend[pool.valid]].mean(0)
        ax.plot(t, wd, color=CAT[0], label="weekday"); ax.plot(t, we, color=CAT[1], label="weekend")
        ax.set_title(f"{cfg.appliances[c][0]} ({c})", fontsize=9)
        ax.set_xticks([0, 6, 12, 18, 24])
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("hour of day (AMPds2 local time)"); fig.supylabel("mean active power (W)")
    fig.suptitle("AMPds2 equipment library: average daily profile per appliance (source behaviour, before Tunisian adaptation)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, ctx, "02_ampds2_appliance_profiles.png")


def fig_house_days(ctx):
    houses = ctx["houses"]
    pv_k = int(np.flatnonzero(houses.has_pv)[0]); no_k = int(np.flatnonzero(~houses.has_pv)[0])
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey="row")
    for r, k in enumerate([no_k, pv_k]):
        for c, (month, lab) in enumerate([(1, "winter"), (7, "summer")]):
            day = _day(ctx, month)
            if c == 1 and day.month != 7:
                day = _day(ctx, ctx["index"][-1].month)
            idx, g = _grouped(ctx, k, day)
            ax = axes[r, c]
            ax.stackplot(idx, *g.values(), labels=list(g), colors=CAT[:len(g)], linewidth=0.4, edgecolor="white")
            if houses.has_pv[k]:
                m = ctx["index"].normalize() == day
                ax.plot(idx, ctx["pv_w"][m, k] / 1000, color=INK, lw=1.4, ls="--", label="PV generation")
            ax.set_title(f"{houses.house_id[k]} ({'PV ' + format(houses.pv_kwp[k], '.0f') + ' kWp' if houses.has_pv[k] else 'no PV'})"
                         f" - {day:%d %b} ({lab})")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            ax.set_ylabel("kW")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[1, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Equipment (AMPds2) behind each house's load, by appliance group", fontweight="bold")
    fig.tight_layout()
    _save(fig, ctx, "03_house_equipment_stack.png")


def fig_heatmaps(ctx):
    houses, truth, cfg = ctx["houses"], ctx["truth"], ctx["cfg"]
    spd = 24 * 60 // cfg.resolution_min
    fig, axes = plt.subplots(3, 5, figsize=(15, 8), sharex=True, sharey=True)
    vmax = max(truth[h].P_load_W.quantile(.995) for h in houses.house_id) / 1000
    for ax, h in zip(axes.ravel(), houses.house_id):
        a = truth[h].P_load_W.to_numpy().reshape(-1, spd) / 1000
        im = ax.imshow(a, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                       extent=[0, 24, 0, a.shape[0]])
        ax.set_title(h.replace("house_", "house "), fontsize=9); ax.grid(False)
    for ax in axes.ravel()[len(houses):]:
        ax.axis("off")
    fig.supxlabel("hour of day"); fig.supylabel("day of simulation")
    fig.colorbar(im, ax=axes, label="household load (kW)", shrink=.6)
    fig.suptitle("Household load, day x time-of-day (clean reference)", fontweight="bold")
    _save(fig, ctx, "04_load_heatmaps.png")


def fig_voltage(ctx):
    houses, truth, meas, res, cfg = ctx["houses"], ctx["truth"], ctx["meas"], ctx["res"], ctx["cfg"]
    idx = ctx["index"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    vm = pd.DataFrame(res["vm"][:, houses.bus.to_numpy()] * 400 / np.sqrt(3), index=idx)
    hod = idx.hour + idx.minute / 60
    g = vm.groupby(hod)
    ax.fill_between(g.min().index, g.min().min(1), g.max().max(1), color=CAT[0], alpha=.18, label="min-max, all houses")
    ax.plot(g.median().index, g.median().median(1), color=CAT[0], label="median")
    ax.axhline(230 * 1.1, color=CAT[4], ls=":", lw=1); ax.axhline(230 * .9, color=CAT[4], ls=":", lw=1, label="EN 50160 +/-10 %")
    ax.set_ylim(200, 260); ax.set_xlabel("hour of day"); ax.set_ylabel("V (phase-neutral)")
    ax.set_title("Voltage at house connection points"); ax.legend(fontsize=7, loc="upper right")
    pv_k = int(np.flatnonzero(houses.has_pv)[0]); h = houses.house_id[pv_k]
    day = _day(ctx, 7)
    m = idx.normalize() == day
    ax = axes[1]
    for i, c in enumerate(CAT[:3]):
        ax.plot(idx[m], truth[h][f"U_{i + 1}"][m], color=c, label=f"L{i + 1}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H")); ax.set_ylabel("V")
    ax.set_title(f"{h}: per-phase voltage, {day:%d %b} (PV house)"); ax.legend(fontsize=7, ncol=3)
    ax = axes[2]
    allu = np.concatenate([truth[x][["U_1", "U_2", "U_3"]].to_numpy().ravel() for x in houses.house_id])
    ax.hist(allu, bins=60, color=CAT[0], alpha=.85)
    ax.axvline(230, color=INK, lw=1, ls="--"); ax.set_xlabel("V"); ax.set_ylabel("samples")
    ax.set_title("Distribution of all phase voltages")
    fig.tight_layout()
    _save(fig, ctx, "05_voltage_quality.png")


def fig_pv(ctx):
    houses, truth, cfg = ctx["houses"], ctx["truth"], ctx["cfg"]
    pvh = houses[houses.has_pv]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for lab, mo, ls in (("winter", 1, "-"), ("summer", 7, "--")):
        day = _day(ctx, mo)
        m = ctx["index"].normalize() == day
        t = ctx["index"][m]
        ax.plot(t.hour + t.minute / 60, ctx["prof"]["kw_per_kwp"][m], ls=ls, color=CAT[1], label=f"{day:%d %b} ({lab})")
    ax.set_xlabel("hour"); ax.set_ylabel("kW per kWp"); ax.set_title("PV yield, Tunis climate"); ax.legend(fontsize=7)
    ax = axes[1]
    day = _day(ctx, 7)
    m = ctx["index"].normalize() == day
    for k, c in zip(pvh.index, CAT):
        h = houses.house_id[k]
        ax.plot(ctx["index"][m], truth[h].P_TOT[m] / 1000, color=c, label=f"{h[-2:]} ({houses.pv_kwp[k]:.0f} kWp)")
    ax.axhline(0, color=MUTED, lw=.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H")); ax.set_ylabel("kW (import +, export -)")
    ax.set_title(f"Net power at PV houses, {day:%d %b}"); ax.legend(fontsize=7, title="house", title_fontsize=7)
    ax = axes[2]
    e = pd.DataFrame({houses.house_id[k][-2:]: truth[houses.house_id[k]].P_PV_W for k in pvh.index})
    mon = e.resample("MS").sum() * cfg.resolution_min / 60 / 1000
    mon.index = mon.index.strftime("%b")
    mon.plot.bar(ax=ax, color=CAT[:len(pvh)], width=.8, legend=True)
    ax.set_ylabel("kWh"); ax.set_xlabel(""); ax.set_title("Monthly PV energy"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, ctx, "06_pv_generation.png")


def fig_calibration(ctx):
    wst, pairing, houses, meas = ctx["wstats"], ctx["pairing"], ctx["houses"], ctx["meas"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    k = 0
    real = wpuq.load_household(wst.loc[pairing[k], "path"])
    ok = wpuq.valid_mask(real)
    u = real["U_1"].where(ok).to_numpy()
    hf_r = (u[1:-1] - .5 * (u[:-2] + u[2:])) / np.nanmean(u) * 100
    um = meas[houses.house_id[k]]["U_1"].to_numpy()
    hf_s = (um[1:-1] - .5 * (um[:-2] + um[2:])) / um.mean() * 100
    ax = axes[0]
    b = np.linspace(-1, 1, 80)
    ax.hist(hf_r[np.isfinite(hf_r)], bins=b, density=True, alpha=.6, color=CAT[0], label=f"WPuQ {pairing[k]} (real)")
    ax.hist(hf_s, bins=b, density=True, alpha=.6, color=CAT[1], label=f"{houses.house_id[k]} (synthetic)")
    ax.set_xlabel("voltage second-difference (% of mean)"); ax.set_title("Voltage short-term noise: real vs synthetic"); ax.legend(fontsize=7)
    ax = axes[1]
    s = wst.loc[pairing]
    x = np.arange(len(s))
    ax.bar(x - .2, s.sigma_u * 100, .4, color=CAT[0], label="sigma U"); ax.bar(x + .2, s.sigma_i * 100, .4, color=CAT[2], label="sigma I")
    ax.set_xticks(x); ax.set_xticklabels([h[-2:] for h in houses.house_id]); ax.set_xlabel("house")
    ax.set_ylabel("relative noise (%)"); ax.set_title("Meter noise calibrated per paired WPuQ household"); ax.legend(fontsize=7)
    ax = axes[2]
    ps = np.array(s.phase_share.tolist()) * 100
    bottom = np.zeros(len(s))
    for i in range(3):
        ax.bar(x, ps[:, i], bottom=bottom, color=CAT[i], label=f"L{i + 1}", width=.8, edgecolor="white", linewidth=1)
        bottom += ps[:, i]
    ax.set_xticks(x); ax.set_xticklabels([h[-2:] for h in houses.house_id]); ax.set_xlabel("house")
    ax.set_ylabel("share of active power (%)"); ax.set_title("Phase split taken from WPuQ"); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    _save(fig, ctx, "07_wpuq_meter_calibration.png")


def fig_truth_vs_measured(ctx):
    houses, truth, meas = ctx["houses"], ctx["truth"], ctx["meas"]
    h = houses.house_id[int(np.flatnonzero(houses.has_pv)[0])]
    day = _day(ctx, 7)
    m = ctx["index"].normalize() == day
    t = ctx["index"][m]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, u in zip(axes, ["U_1", "I_1", "P_TOT"], ["V", "A", "W"]):
        ax.plot(t, truth[h][col][m], color=CAT[0], lw=2.2, alpha=.5, label="physical (power flow)")
        ax.plot(t, meas[h][col][m], color=CAT[1], lw=.9, label="measured (meter model)")
        ax.set_title(f"{h} {col} - {day:%d %b}"); ax.set_ylabel(u); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, ctx, "08_physical_vs_measured.png")


def fig_transformer(ctx):
    res, idx, cfg = ctx["res"], ctx["index"], ctx["cfg"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    s = pd.Series(res["trafo_loading"], index=idx)
    axes[0].plot(s.resample("D").max(), color=CAT[0], label="daily max"); axes[0].plot(s.resample("D").mean(), color=CAT[2], label="daily mean")
    axes[0].axhline(100, color=CAT[4], ls=":"); axes[0].set_ylim(0, 110); axes[0].set_ylabel("% of 160 kVA"); axes[0].legend(fontsize=7)
    axes[0].set_title("Transformer loading")
    p = pd.Series(-res["trafo_p_lv"] * 1000, index=idx)
    hod = idx.hour + idx.minute / 60
    g = p.groupby(hod)
    axes[1].fill_between(g.min().index, g.min(), g.max(), color=CAT[0], alpha=.18); axes[1].plot(g.mean(), color=CAT[0])
    axes[1].axhline(0, color=MUTED, lw=.8); axes[1].set_xlabel("hour"); axes[1].set_ylabel("kW (+ = towards LV)")
    axes[1].set_title("Neighbourhood power at transformer (mean, min-max)")
    loss = pd.Series(res["losses"] * 1000, index=idx)
    axes[2].plot(loss.resample("D").mean(), color=CAT[3]); axes[2].set_ylabel("kW"); axes[2].set_title("Mean daily grid losses (lines+transformer)")
    fig.tight_layout()
    _save(fig, ctx, "09_transformer_and_losses.png")


def fig_validation(ctx):
    rep = ctx["report"]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rep["checks"]) + 1.4))
    ax.axis("off"); ax.grid(False)
    for i, c in enumerate(rep["checks"]):
        y = len(rep["checks"]) - i
        ax.text(0, y, "PASS" if c["passed"] else "FAIL", color="white", fontsize=8, fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=.3", fc=CAT[2] if c["passed"] else CAT[4], ec="none"))
        ax.text(.07, y, c["check"], va="center", fontsize=9); ax.text(.45, y, c["detail"], va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, len(rep["checks"]) + 1)
    ax.set_title("Data-quality gate - " + ("all checks passed: dataset certified clean" if rep["all_passed"] else "CHECKS FAILED"))
    _save(fig, ctx, "10_validation_report.png")


def make_all(ctx):
    _style()
    for f in (fig_network, fig_ampds2, fig_house_days, fig_heatmaps, fig_voltage, fig_pv, fig_calibration,
              fig_truth_vs_measured, fig_transformer, fig_validation):
        try:
            f(ctx)
        except Exception as e:  # a figure must never lose the dataset
            print(f"    figure {f.__name__} failed: {type(e).__name__}: {e}")
