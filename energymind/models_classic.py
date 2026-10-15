"""Classic models. All: fit(X, y=None), predict_proba(X)->(n,3) [normal, measurement, physical].

Latency: inference is a few ms per batch of windows on CPU (see `latency_ms`); one window
(15 min stride 5) is produced every 5 minutes, so real-time budget is not a concern.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM

MODEL_DIR = Path("outputs/models")
SEED = 0


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _full3(proba, classes):
    out = np.zeros((len(proba), 3))
    out[:, list(classes)] = proba
    return out


class _Base:
    def _prep_fit(self, X, y):
        self.feature_names_ = list(X.columns)
        X = X[(np.asarray(y) == 0)] if y is not None else X  # normal-only for unsupervised
        return X

    def _align(self, X):
        return X.reindex(columns=self.feature_names_)

    def save(self, path=None):
        path = Path(path or MODEL_DIR / f"{type(self).__name__}.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path):
        return joblib.load(path)


class _Robust(_Base):
    """Robust z-scores against NORMAL training windows (median/MAD), no test statistics."""

    def _fit_z(self, X):
        A = X.to_numpy(float)
        self.med_ = np.nanmedian(A, 0)
        mad = 1.4826 * np.nanmedian(np.abs(A - self.med_), 0)
        self.sc_ = np.maximum(mad, np.maximum(0.1 * np.nanstd(A, 0), 1e-9))
        names = np.array(self.feature_names_)
        # measurement-type features: physics residuals, frozen runs, gaps; the rest = state/dynamics
        self.meas_ = np.array([n.startswith("res_") or n.endswith("_frozen") or n.startswith(("missing", "valid_frac"))
                               for n in names])
        self.med_ = np.nan_to_num(self.med_)
        self.sc_ = np.nan_to_num(self.sc_, nan=1.0)

    def _z(self, X):
        A = self._align(X).to_numpy(float)
        return np.nan_to_num(np.abs(A - self.med_) / self.sc_, nan=0.0)  # NaN features carry no evidence

    def group_scores(self, X):
        z = self._z(X)
        m = z[:, self.meas_].max(1) if self.meas_.any() else np.zeros(len(z))
        ph = z[:, ~self.meas_].max(1) if (~self.meas_).any() else np.zeros(len(z))
        return m, ph


class StatBaseline(_Robust):
    """Rule-based: measurement if a physics/frozen/gap feature exceeds thr; physical if only
    state features do (physical events keep the invariants intact)."""

    def __init__(self, thr=6.0):
        self.thr = thr

    def fit(self, X, y=None):
        X = self._prep_fit(X, y)
        self._fit_z(X)
        return self

    def anomaly_score(self, X):
        return np.maximum(*self.group_scores(X))

    def predict_proba(self, X):
        m, ph = self.group_scores(X)
        ph = np.where(m > self.thr, ph - m, ph)  # invariants broken -> not a plain physical event
        return _softmax(np.stack([np.full(len(m), self.thr), m, ph], 1))


class _Unsup(_Robust):
    """Unsupervised score + small supervised head (fit_head on calib) -> 3 classes."""

    head_ = None

    def _impute(self, X):
        A = self._align(X).to_numpy(float)
        A = np.where(np.isnan(A), self.fill_, A)
        return np.clip((A - self.fill_) / self.sc_, -20, 20)

    def fit(self, X, y=None):
        X = self._prep_fit(X, y)
        self._fit_z(X)
        self.fill_ = self.med_
        self._fit_core(self._impute(X))
        tr = self.anomaly_score(X)
        self.thr_ = float(np.quantile(tr, 0.99))
        self.scale_ = float(np.std(tr)) + 1e-9
        return self

    def _head_x(self, X):
        m, ph = self.group_scores(X)
        return np.stack([(self.anomaly_score(X) - self.thr_) / self.scale_, np.log1p(m), np.log1p(ph)], 1)

    def fit_head(self, Xcal, ycal):
        ycal = np.asarray(ycal)
        if len(np.unique(ycal)) > 1:
            self.head_ = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
            self.head_.fit(self._head_x(Xcal), ycal)
        return self

    def predict_proba(self, X):
        if self.head_ is not None:
            return _full3(self.head_.predict_proba(self._head_x(X)), self.head_.classes_)
        pa = 1 / (1 + np.exp(-(self.anomaly_score(X) - self.thr_) / self.scale_))
        return np.stack([1 - pa, pa / 2, pa / 2], 1)


class IsoForestModel(_Unsup):
    def __init__(self, n_estimators=200, seed=SEED):
        self.n_estimators, self.seed = n_estimators, seed

    def _fit_core(self, A):
        self.m_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=1).fit(A)

    def anomaly_score(self, X):
        return -self.m_.score_samples(self._impute(X))


class OCSVMModel(_Unsup):
    def __init__(self, nu=0.02, max_train=4000, seed=SEED):
        self.nu, self.max_train, self.seed = nu, max_train, seed

    def _fit_core(self, A):
        if len(A) > self.max_train:  # ponytail: random subsample keeps O(n^2) SVM fit bounded
            A = A[np.random.RandomState(self.seed).choice(len(A), self.max_train, replace=False)]
        self.m_ = OneClassSVM(nu=self.nu, gamma="scale").fit(A)

    def anomaly_score(self, X):
        return -self.m_.score_samples(self._impute(X))


class LGBMModel(_Base):
    def __init__(self, n_estimators=400, seed=SEED, **kw):
        self.n_estimators, self.seed, self.kw = n_estimators, seed, kw

    def fit(self, X, y, Xval=None, yval=None):
        self.feature_names_ = list(X.columns)
        self.m_ = LGBMClassifier(objective="multiclass", n_estimators=self.n_estimators, learning_rate=0.05,
                                 num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, class_weight="balanced", random_state=self.seed,
                                 deterministic=True, force_row_wise=True, n_jobs=2, verbose=-1, **self.kw)
        kw = {}
        if Xval is not None and len(np.unique(yval)) > 0:
            kw = dict(eval_set=[(self._align(Xval), np.asarray(yval))], callbacks=[early_stopping(30, verbose=False)])
        self.m_.fit(X, np.asarray(y), **kw)
        return self

    def predict_proba(self, X):
        return _full3(self.m_.predict_proba(self._align(X)), self.m_.classes_)


def fit_all(Xtr, ytr, Xcal, ycal, val_frac=0.2, purge=5):
    """Fit every model. Train windows are time-ordered: the last `val_frac` (after `purge`
    overlapping windows) is the LGBM early-stopping set; calib is used only for the heads."""
    ytr = np.asarray(ytr)
    k = int(len(Xtr) * (1 - val_frac))
    out = {"stat": StatBaseline().fit(Xtr, ytr),
           "iforest": IsoForestModel().fit(Xtr, ytr).fit_head(Xcal, ycal),
           "ocsvm": OCSVMModel().fit(Xtr, ytr).fit_head(Xcal, ycal),
           "lgbm": LGBMModel().fit(Xtr.iloc[:k], ytr[:k], Xtr.iloc[k + purge:], ytr[k + purge:])}
    return out


def latency_ms(model, X, n=200):
    """Median ms per window for single-window inference."""
    ts = []
    for j in range(min(n, len(X))):
        t = time.perf_counter()
        model.predict_proba(X.iloc[j:j + 1])
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))
