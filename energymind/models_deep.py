"""Deep models: sequence builder, conv autoencoder detector, causal TCN classifier."""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = ["v", "i", "p", "q", "s", "pf", "valid_frac"]
CLIP = 20.0


def _residuals(f):
    """Physics residuals from canonical columns (only those computable)."""
    r, ref = {}, {}
    if {"pf", "p", "s"} <= set(f):
        r["res_pf"] = f.pf - f.p / f.s.replace(0, np.nan); ref["res_pf"] = 1.0
    if {"s", "p", "q"} <= set(f):
        r["res_s"] = f.s - np.sqrt(f.p ** 2 + f.q ** 2); ref["res_s"] = f.s.median()
    if {"s", "v", "i"} <= set(f):
        r["res_vi"] = f.s - f.v * f.i; ref["res_vi"] = f.s.median()
    if {"e", "p"} <= set(f):
        r["res_e"] = f.e.diff() - f.p / 60.0; ref["res_e"] = f.p.median()
    return pd.DataFrame(r, index=f.index), ref


def _channel_frame(frame, channels):
    base = [c for c in (channels or BASE) if c in frame]
    res, ref = _residuals(frame)
    return pd.concat([frame[base], res], axis=1).astype("float64"), ref


def make_sequences(frame, labels=None, win=60, stride=10, channels=None, scaler=None):
    """Returns X float32 [n,win,c], y (int or None), meta, scaler.
    Channels = base + physics residuals (robust-normalised, clipped) + one NaN-mask channel per base channel.
    If scaler is None it is fit on `frame` (pass the TRAIN frame); otherwise it is reused untouched."""
    df, ref = _channel_frame(frame, channels)
    if scaler is None:
        med = df.median()
        iqr = (df.quantile(.75) - df.quantile(.25)) / 1.349
        floor = pd.Series({c: 0.005 * abs(ref[c]) if c in ref else 1e-6 for c in df}) + 1e-9
        scaler = {"cols": list(df.columns), "med": med.fillna(0).to_dict(),
                  "scale": np.maximum(iqr.fillna(0), floor).to_dict()}
    cols = scaler["cols"]
    df = df.reindex(columns=cols)  # missing channel -> NaN -> 0 + mask
    z = (df - pd.Series(scaler["med"])) / pd.Series(scaler["scale"])
    mask = df.isna().astype("float32").values
    arr = np.concatenate([z.clip(-CLIP, CLIP).fillna(0).values.astype("float32"), mask], axis=1)
    n = (len(arr) - win) // stride + 1
    starts = np.arange(max(n, 0)) * stride
    if n > 0:
        X = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(arr, win, axis=0)[::stride][:n].transpose(0, 2, 1))
    else:
        X = np.zeros((0, win, arr.shape[1]), "float32")
    idx = frame.index
    meta = pd.DataFrame({"start": idx[starts] if n > 0 else [], "end": idx[starts + win - 1] if n > 0 else [],
                         "incident_id": -1, "subtype": "", "severity": 0.0})
    y = None
    if labels is not None:  # vectorised: incident class covering >= 50% of the window (class 2 wins a 50/50 tie)
        cls, inc, sub = labels["cls"].values, labels["incident_id"].values, labels["subtype"].values
        sev = labels["severity"].values if "severity" in labels else np.zeros(len(cls))
        y = np.zeros(len(starts), int)
        iid, isub, isev = np.full(len(starts), -1), np.full(len(starts), "", dtype=object), np.zeros(len(starts))
        r = np.arange(len(cls))
        for k_cls in (1, 2):
            hit = cls == k_cls
            cnt = np.r_[0, np.cumsum(hit)]
            ok = (cnt[starts + win] - cnt[starts]) >= win / 2
            nxt = np.minimum.accumulate(np.where(hit, r, len(cls))[::-1])[::-1]  # first row >= i of this class
            j = nxt[starts[ok]]
            y[ok] = k_cls
            iid[ok], isub[ok], isev[ok] = inc[j], sub[j], sev[j]
        meta["incident_id"], meta["subtype"], meta["severity"] = iid, isub, isev
    return X, y, meta, scaler


def pick_device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_loop(model, dev, loss_fn, X, y, Xv, yv, epochs, bs, lr, patience, seed, amp):
    """Generic trainer. loss_fn(model, xb, yb) -> scalar. Early stopping on val loss. Returns history."""
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(np.asarray(X, "float32"))
    yt = None if y is None else torch.from_numpy(np.asarray(y)).long()
    use_amp = amp and dev.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist, best, bad, best_state = [], np.inf, 0, None

    def val_loss():
        model.eval()
        if Xv is None:
            return None
        with torch.no_grad():
            xb = torch.from_numpy(np.asarray(Xv, "float32")).to(dev)
            yb = None if yv is None else torch.from_numpy(np.asarray(yv)).long().to(dev)
            return float(loss_fn(model, xb, yb))

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for k in range(0, len(Xt), bs):
            ix = perm[k:k + bs]
            xb = Xt[ix].to(dev); yb = None if yt is None else yt[ix].to(dev)
            opt.zero_grad()
            with torch.autocast(dev.type, enabled=use_amp):
                loss = loss_fn(model, xb, yb)
            if not torch.isfinite(loss):  # unstable -> skip batch
                continue
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss) * len(ix)
        tr = tot / len(Xt)
        vl = val_loss()
        hist.append(tr if vl is None else vl)
        hist_tr = tr
        cur = hist[-1]
        if cur < best - 1e-5:
            best, bad = cur, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return hist


class _Base:
    def __init__(self, seed=0, device=None):
        self.seed, self.dev = seed, pick_device(device)

    def to(self, device):
        self.dev = pick_device(device); self.net.to(self.dev); return self

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())

    def _batches(self, X, fn, bs=1024):
        self.net.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(X), bs):
                xb = torch.from_numpy(np.asarray(X[k:k + bs], "float32")).to(self.dev)
                out.append(fn(xb).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0,))

    def save(self, path):
        torch.save({"cfg": self.cfg, "state": self.net.state_dict(), "extra": self._extra(), "seed": self.seed}, path)

    @classmethod
    def load(cls, path, device=None):
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(**d["cfg"], seed=d["seed"], device=device)
        m.net.load_state_dict(d["state"]); m.net.to(m.dev); m.net.eval(); m._set_extra(d["extra"])
        return m


# ---------------------------------------------------------------- autoencoder
class _AE(nn.Module):
    def __init__(self, c, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(c, h, 5, 2, 2), nn.ReLU(), nn.Conv1d(h, h, 5, 2, 2), nn.ReLU(),
                                 nn.Conv1d(h, h // 2, 3, 1, 1))
        self.dec = nn.Sequential(nn.Conv1d(h // 2, h, 3, 1, 1), nn.ReLU(),
                                 nn.ConvTranspose1d(h, h, 4, 2, 1), nn.ReLU(), nn.ConvTranspose1d(h, c, 4, 2, 1))

    def forward(self, x):  # x [b,t,c]
        t = x.shape[1]
        y = self.dec(self.enc(x.transpose(1, 2)))
        y = F.interpolate(y, size=t) if y.shape[-1] != t else y
        return y.transpose(1, 2)


class AEDetector(_Base):
    def __init__(self, n_channels=1, hidden=32, seed=0, device=None):
        super().__init__(seed, device)
        self.cfg = {"n_channels": n_channels, "hidden": hidden}
        _seed(seed)
        self.net = _AE(n_channels, hidden).to(self.dev)
        self.threshold = None

    def fit(self, X, y=None, Xval=None, yval=None, epochs=30, bs=128, lr=2e-3, patience=5, amp=True):
        """Train on normal windows (if y given, only y==0 rows are used)."""
        if y is not None:
            X = X[np.asarray(y) == 0]
        if Xval is not None and yval is not None:
            Xval = Xval[np.asarray(yval) == 0]
        _seed(self.seed)
        loss = lambda m, xb, yb: F.mse_loss(m(xb), xb)
        self.history = _fit_loop(self.net, self.dev, loss, X, None, Xval, None, epochs, bs, lr, patience, self.seed, amp)
        return self

    def anomaly_score(self, X):
        return self._batches(X, lambda xb: ((self.net(xb) - xb) ** 2).mean((1, 2)))

    def fit_threshold(self, Xcalib_normal, q=0.99):
        self.threshold = float(np.quantile(self.anomaly_score(Xcalib_normal), q))
        return self.threshold

    def predict(self, X):
        return (self.anomaly_score(X) > self.threshold).astype(int)

    def _extra(self): return {"threshold": self.threshold}
    def _set_extra(self, e): self.threshold = e["threshold"]


# ---------------------------------------------------------------- TCN
class _Block(nn.Module):
    def __init__(self, h, d, drop):
        super().__init__()
        self.pad = 2 * d  # (k-1)*d, left-only => causal
        self.c1, self.c2 = nn.Conv1d(h, h, 3, dilation=d), nn.Conv1d(h, h, 3, dilation=d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.c2(F.pad(y, (self.pad, 0)))))
        return F.relu(x + y)


class _TCN(nn.Module):
    def __init__(self, c, h, dilations, drop, n_cls=3):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 1)
        self.blocks = nn.Sequential(*[_Block(h, d, drop) for d in dilations])
        self.head = nn.Conv1d(h, n_cls, 1)

    def step_logits(self, x):  # x [b,t,c] -> [b,t,3]; output at t depends only on x[:t+1]
        return self.head(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)

    def forward(self, x):  # window logits = last timestep
        return self.step_logits(x)[:, -1]


class TCNClassifier(_Base):
    def __init__(self, n_channels=1, hidden=48, dilations=(1, 2, 4, 8, 16), dropout=0.1, seed=0, device=None):
        super().__init__(seed, device)
        self.cfg = {"n_channels": n_channels, "hidden": hidden, "dilations": tuple(dilations), "dropout": dropout}
        _seed(seed)
        self.net = _TCN(n_channels, hidden, dilations, dropout).to(self.dev)
        self.T = 1.0
        # Class-weighted CE learns logits shifted by log(w_c) (prior shift); subtracting it at inference recovers
        # the unweighted posterior, so temperature scaling no longer has to absorb the shift (T ~0.35 before).
        self.log_w = np.zeros(3, "float32")
        self.prior_correct = True

    def fit(self, X, y, Xval=None, yval=None, epochs=40, bs=128, lr=2e-3, patience=6, amp=True):
        cnt = np.bincount(np.asarray(y), minlength=3).astype("float32")
        wn = cnt.sum() / (3 * np.maximum(cnt, 1))
        self.log_w = np.log(wn).astype("float32")
        w = torch.tensor(wn, device=self.dev)  # inverse-frequency
        loss = lambda m, xb, yb: F.cross_entropy(m(xb).float(), yb, weight=w)
        _seed(self.seed)
        self.history = _fit_loop(self.net, self.dev, loss, X, y, Xval, yval, epochs, bs, lr, patience, self.seed, amp)
        return self

    def logits(self, X):
        z = self._batches(X, self.net)
        return z - self.log_w if self.prior_correct and len(z) else z

    def predict_proba(self, X):
        z = self.logits(X) / self.T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def calibrate(self, X, y):
        """Temperature scaling on calib block (LBFGS on log T)."""
        z = torch.from_numpy(self.logits(X)); t = torch.from_numpy(np.asarray(y)).long()
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad(); l = F.cross_entropy(z / logT.exp(), t); l.backward(); return l
        opt.step(closure)
        self.T = float(logT.exp().clamp(0.05, 20))
        return self.T

    def _extra(self): return {"T": self.T, "log_w": self.log_w.tolist(), "prior_correct": self.prior_correct}

    def _set_extra(self, e):
        self.T = e["T"]
        self.log_w = np.asarray(e.get("log_w", [0, 0, 0]), "float32")  # models saved before the fix: no shift
        self.prior_correct = e.get("prior_correct", True)


def latency(model, X, n=200, device=None):
    """Per-window (batch of 1) inference latency in ms: {'p50','p95','device','n_params'}."""
    old = model.dev
    if device is not None:
        model.to(device)
    fn = model.predict_proba if hasattr(model, "predict_proba") else model.anomaly_score
    for k in range(10):
        fn(X[k % len(X):k % len(X) + 1])  # warmup
    ts = []
    for k in range(n):
        j = k % len(X)
        if model.dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(X[j:j + 1])
        if model.dev.type == "cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    out = {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95)),
           "device": str(model.dev), "n_params": model.n_params()}
    model.to(old)
    return out
