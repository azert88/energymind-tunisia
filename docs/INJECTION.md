# Incident injection (`energymind/inject.py`)

`inject(frame, seed, n_incidents=None, severities=(0.3,0.6,1.0), families=None, rate_per_day=2.0, block_edges=None) -> (frame_out, labels, manifest)`

Seeded, deterministic, input never mutated. Incidents are placed at random on stretches where `p`,`v` are finite and `valid_frac>=0.5`
(never on NaN stretches), never overlap (10 min guard), and, if `block_edges` (see `split_edges`) is given, stay >=60 min away
from every block edge so they lie wholly in one `split_by_time` block. Default count = `rate_per_day * days`. Family weights:
measurement 0.45, physical 0.35, benign 0.20, uniform over eligible subtypes inside a family. `families` accepts family names
and/or subtypes. Manifest `end` is the last affected minute (inclusive); `params.rows=[a,b)` gives row positions.
Rows = minutes (canonical 1-min frame). `X` below is a channel; `med(X)` its median over the frame; sev in {0.3,0.6,1.0}.

## Measurement (cls 1) - only reported channels change, `e` untouched, invariants break

| subtype | duration | formula | severity mapping |
|---|---|---|---|
| bias | 30-180 | `X += s*0.08*sev*med(X)`, X random in v,i,p, s=+-1 | 2.4% / 4.8% / 8% of median |
| drift | 60-180 | `X += ramp(0 -> s*0.08*sev*med(X))` | same final size |
| scale | 30-120 | `X *= 1 + s*0.5*sev`, X in i,p | +-15% / 30% / 50% |
| noise | 20-120 | `X += N(0, 0.04*sev*med(X))`, X in v,i,p,q,s | sigma 1.2% / 2.4% / 4% |
| frozen | 10-90 | hold value at t=start-1 | channels: sev .3 one of v,i,p,q,s,pf; .6 p,q,s; 1.0 all six |
| dropout | 10-60 | NaN run, `valid_frac=0` | same channel mapping as frozen |
| delay | 30-120 | `X[t] = X[t-k]` for i,p,q,s,pf (v not delayed) | k = 3 / 8 / 20 min |
| replay | 30-120 | copy same time-of-day segment 1-3 days earlier from the pristine frame (skipped if no history/NaN) | channels as frozen (p only / p,q,s / all six) |
| falsified | 60-180 | coordinated ramp `g: 1 -> 1+-0.4*sev` on i,p,q,s (+per-phase), `pf`,`v` unchanged, `e` re-integrated: all invariants hold | final scale +-12% / 24% / 40% |

## Physical (cls 2) - state changed consistently, invariants hold, `e` re-integrated (increments of `p` change added to `e`; later `e` shifted)

| subtype | duration | formula | severity |
|---|---|---|---|
| voltage_sag | 10-60 | `d=0.35*sev`; v*=(1-d). Per-incident `load_model` (in params): const_power -> i/=(1-d), p,q,s unchanged; const_impedance -> i*=(1-d), p,q,s*=(1-d)^2 | d = 10.5% / 21% / 35% |
| outage | 15-120 | v,i *= k; p,q,s *= k^2, `k=0.17-0.16*sev` | residual voltage 12% / 7% / 1% |
| demand_step | 30-180 | i,p,q,s (+per-phase) *= f, v,pf kept; f=1+0.6sev (up) or max(1-0.6sev,0.2) (down), random direction | step 18/36/60% |
| duty_cycle | 60-180 | square wave period T in 4-30 min, on half the cycle. If a `plug_*`/`sub_*` column with max>20 W in the segment exists it is forced to `x*(0.2+0.8*on)` and the delta propagated into i,p,q,s; else added load `max(0.3*sev*med(p),50) W` | amplitude scales with sev |
| phase_imbalance | 30-120 | only if p_l1,p_l3,i_l1,i_l3 exist: l1 *= 1+k, l3 *= 1-k, `k=0.4*sev`; totals unchanged (locally invisible in totals) | k = 12% / 24% / 40% |

## Benign look-alikes (cls 0, subtype kept for analysis; same physics as demand_step)

| subtype | duration | formula | severity |
|---|---|---|---|
| benign_new_load | 60-180 | demand step up, f = 1+0.6*sev | as demand_step |
| benign_pv_ramp | 60-180 | smooth dip `g = 1 - min(0.6sev,0.8)*sin(pi*t/L)` on i,p,q,s (PV reducing net import), `e` re-integrated | depth 18% / 36% / 60% |

## Derived channels (v2, default in `pipeline`)

`inject(..., derived=DERIVED["eco"])` (ECO: s, q, pf, e) / `DERIVED["ampds2"]` (pf). These are the channels the
loader computes from other channels. A measurement fault picks only real registers. `s` and the per-phase registers
follow a `v` / `i` edit, and `q`, `pf`, `e` are re-derived with the loader formulas. `derived=None` is the legacy
behaviour of the first run (bit-identical). That behaviour produced impossible residuals and a float-rounding
fingerprint in every edited window, i.e. a label shortcut: see `results/leakage_audit.md`. The tables above describe
edits on registers; in v2 the derived channels then follow.

## Limitations

- A label records **how the data was generated, not provable intent**. `falsified` is physically indistinguishable
  from `demand_step` by invariants alone; only context (time-of-day, history, cross-meter) can separate them. Likewise
  benign_* vs demand_step: both are legitimate-looking state changes by design.
- Benign and physical events are bounded (revert at segment end); a real new load would persist.
- Events are relative scalings, so incidents on low-p minutes have small absolute effect. `delay` and `replay` need history
  (rows before start); otherwise the placement is silently dropped, so fewer than `n_incidents` may be produced (see manifest).
- Non-total extra columns (`plug_*`, `sub_*`, `v_l*` for non-sag events) are left untouched except where stated.
- Synthetic injections on real baselines: results measure detection of these generators, not field performance.
