"""Automated noise sweep for WSINDy scattering-data model identification.

Sweeps the i.i.d. Gaussian noise ratio sigma_NR = ||eta||_F / ||U||_F over many
levels and RNG seeds, and computes the Section 4.1 validation metrics for each of
the four manuscript cases (unperturbed/perturbed x single/two-soliton):

  * TPR                  -- true positive ratio vs. the reference scattering ODEs
  * E_inf                -- normalized l-infinity coefficient error
  * R^2                  -- per equation, only when its BIC test did NOT trigger
  * RMSE_IST             -- pointwise RMSE of the field reconstructed via the
                            reflectionless IST from the forecast scattering data
                            (scalar only; the reconstructed U_IST is discarded)

The heavy lifting (scattering extraction + WSINDy fits + IST reconstruction) reuses
the exact functions that recreate_paper_results.ipynb uses, so the numerics match
the notebook.  Two things are added here:

  * spatial subsampling of U (every `stride`-th point in x, never in time) to speed
    up the scattering computation, which is the dominant cost; and
  * a crash-safe, resumable append-only JSONL result log under outputs/.

Run from the command line, e.g.

    python noise_sweep.py                         # full sweep (defaults below)
    python noise_sweep.py --cases single_soliton --noise-step 0.5 --n-seeds 1

or import and call run_sweep(make_config(...)) from a notebook.
"""

import os
import io
import json
import time
import socket
import argparse
import platform
import subprocess
from contextlib import redirect_stdout, redirect_stderr

import matplotlib
import numpy as np
import torch

from netCDF4 import Dataset  # noqa: F401  (kept for parity with helpers' env)
from helper_fcns import add_noise, compute_residuals
from compute_scattering_data import scattering_data_sequence
from recreate_results_helpers import (
    load_kdv_field,
    compute_tpr,
    fit_wsindy_bic_stacked,
    fit_wsindy_bic_stacked_eq,
    RMSE_of_forecast,
    RMSE_of_forecast_pooled,
)

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------------------
# Case registry: one source of truth for everything that differs between the four cases.
# --------------------------------------------------------------------------------------
NAMES = [r"\kappa", r"\log(c)"]

# Shared WSINDy hyperparameters (identical to recreate_paper_results.ipynb).
SWEEP_HYPERPARAMS = dict(m=[20], p=[10], s=[1], rescale=False, trigger_BIC="poor_fit")
EPS = 0.2          # perturbation strength for the perturbed cases
LOGC_LAMBDA = 4e-2  # fixed MSTLS lambda for the perturbed log(c) equation (eq=1)

BETA_UNPERT = [[1, 0], [2, 0], [3, 0],           # k, k^2, k^3
               [0, 1], [0, 2], [0, 3], [1, 1]]   # log c, (log c)^2, (log c)^3, k*log c
BETA_PERT = [[1, 0], [2, 0], [3, 0],             # k, k^2, k^3
             [0, 1], [0, 2], [0, 3]]             # log c, (log c)^2, (log c)^3

W_TRUTH_UNPERT = [torch.zeros(len(BETA_UNPERT)),
                  torch.tensor([0., 0., 0., 0., 0., 8., 0.])]        # log(c)_t = 8 k^3
W_TRUTH_PERT = [torch.tensor([EPS / 3, 0., 0., 0., 0., 0.]),         # k_t = (eps/3) k
                torch.tensor([-2 * EPS / 3, 0., 0., 0., 0., 8.])]    # log(c)_t = -(2eps/3) log c + 8 k^3


def _case(key, label, perturbed, n_solitons, config_files):
    beta = BETA_PERT if perturbed else BETA_UNPERT
    w_truth = W_TRUTH_PERT if perturbed else W_TRUTH_UNPERT
    return dict(key=key, label=label, perturbed=perturbed, n_solitons=n_solitons,
                config_files=config_files, beta=beta, w_truth=w_truth)


CASES = [
    _case("single_soliton", "unperturbed, single soliton", False, 1,
          ["outputs/kdv_single_soliton_v0.nc",
           "outputs/kdv_single_soliton_v1.nc",
           "outputs/kdv_single_soliton_v2.nc"]),
    _case("two_soliton", "unperturbed, two-soliton collision", False, 2,
          ["outputs/kdv_soliton_collision_v0.nc",
           "outputs/kdv_soliton_collision_v1.nc",
           "outputs/kdv_soliton_collision_v2.nc"]),
    _case("single_soliton_perturbed", "perturbed, single soliton", True, 1,
          ["outputs/kdv_single_soliton_perturbed_v0.nc",
           "outputs/kdv_single_soliton_perturbed_v1.nc",
           "outputs/kdv_single_soliton_perturbed_v2.nc"]),
    _case("two_soliton_perturbed", "perturbed, two-soliton collision", True, 2,
          ["outputs/kdv_soliton_collision_perturbed_v0.nc",
           "outputs/kdv_soliton_collision_perturbed_v1.nc",
           "outputs/kdv_soliton_collision_perturbed_v2.nc"]),
]
CASES_BY_KEY = {c["key"]: c for c in CASES}


class SolitonCountError(RuntimeError):
    """Raised when a snapshot yields fewer bound states than the case requires."""


# --------------------------------------------------------------------------------------
# Scattering extraction (mirrors scattering_series / scattering_series_n, plus spatial
# subsampling and per-snapshot soliton-count diagnostics).
# --------------------------------------------------------------------------------------
def extract_states(config_files, n_solitons, noise, seed, stride, min_kappa):
    """Return (per_sim_states, t_tensor, n_diagnostics) for the 3 simulations of a case.

    Noise is added to the FULL field (the paper's sigma_NR is defined on full U) and the
    RNG is seeded ONCE before the three sims so a given seed is reproducible regardless of
    execution order.  Space is subsampled by `stride`; time is never subsampled.
    """
    torch.manual_seed(seed)
    per_sim_states, n_diagnostics, t_ref = [], [], None

    for path in config_files:
        x, t_np, U_star = load_kdv_field(path)
        U = torch.tensor(U_star, dtype=torch.float64)
        if noise > 0:
            U = add_noise(U, noise)

        xs = x[::stride]
        Us = U.numpy()[:, ::stride]
        seq = scattering_data_sequence(xs, Us, t_grid=t_np, min_kappa=min_kappa)

        n_arr = np.asarray(seq.n)
        n_diagnostics.append(n_arr)
        if int(n_arr.min()) < n_solitons:
            raise SolitonCountError(
                f"{os.path.basename(path)}: snapshot with n={int(n_arr.min())} < "
                f"required {n_solitons} (noise={noise}, seed={seed}, min_kappa={min_kappa})")

        kappas, log_cs = [], []
        for k, lc in zip(seq.kappas, seq.log_cs):
            order = np.argsort(k)[::-1][:n_solitons]   # n_solitons largest kappas
            kappas.append(np.asarray(k)[order])
            log_cs.append(np.asarray(lc)[order])
        kappas, log_cs = np.array(kappas), np.array(log_cs)   # (n_t, n_solitons)

        sim_states = [torch.tensor(kappas[:, i], dtype=torch.float64) for i in range(n_solitons)] \
                   + [torch.tensor(log_cs[:, i], dtype=torch.float64) for i in range(n_solitons)]
        per_sim_states.append(sim_states)
        t_ref = t_np

    return per_sim_states, torch.tensor(t_ref, dtype=torch.float64), n_diagnostics


def build_fit_states(per_sim_states, n_solitons):
    """Shape extracted states into the structure each fit routine expects.

    Single soliton: list of [kappa, log c] (one per sim).
    Two soliton: each soliton pooled as its own [kappa_i, log c_i] 2-state system.
    """
    if n_solitons == 1:
        return per_sim_states
    pooled = []
    for sim in per_sim_states:
        for i in range(n_solitons):
            pooled.append([sim[i], sim[n_solitons + i]])
    return pooled


# --------------------------------------------------------------------------------------
# Fit dispatch + metric extraction (output suppressed; helpers are notebook-oriented).
# --------------------------------------------------------------------------------------
def run_fit(case, fit_states, t):
    """Return (models, coeffs) -- models ordered [kappa-eq, log(c)-eq]."""
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        if not case["perturbed"]:
            models, _odes, coeffs = fit_wsindy_bic_stacked(
                fit_states, t, NAMES, case["beta"], **SWEEP_HYPERPARAMS)
        else:
            m_k, _ode_k, w_k = fit_wsindy_bic_stacked_eq(
                fit_states, t, NAMES, case["beta"], eq=0, **SWEEP_HYPERPARAMS)
            m_c, _ode_c, w_c = fit_wsindy_bic_stacked_eq(
                fit_states, t, NAMES, case["beta"], eq=1, Lambda=LOGC_LAMBDA,
                **SWEEP_HYPERPARAMS)
            models, coeffs = [m_k, m_c], [w_k, w_c]
    return models, coeffs


def compute_metrics(models, coeffs, w_truth):
    """TPR, E_inf, per-equation R^2 (None if BIC triggered), and BIC flags."""
    w_found = torch.cat([w for w in coeffs])
    w_truth_cat = torch.cat(w_truth)

    TPR = compute_tpr(w_found, w_truth_cat)
    nonzero = w_truth_cat != 0
    E_inf = ((w_found - w_truth_cat).abs()[nonzero] / w_truth_cat.abs()[nonzero]).max().item()

    R2, bic, Lambda = {}, {}, {}
    for name, m in zip(["kappa", "logc"], models):
        bic[name] = bool(m.bic_triggered)
        Lambda[name] = float(m.Lambda)
        # rescale=False throughout, so coeffs are already in the raw frame (matches print_report)
        R2[name] = None if m.bic_triggered else float(compute_residuals(m.library, m.coeffs, m.lhs)[1])
    return TPR, E_inf, R2, bic, Lambda


def compute_rmse(case, per_sim_states, coeffs, t):
    """Scalar IST-RMSE for the v0 trajectory on the FULL clean grid (U_IST discarded)."""
    x_full, _t, U_star = load_kdv_field(case["config_files"][0])  # v0, full grid, clean
    states_v0 = per_sim_states[0]
    t_np = t.numpy()
    if case["n_solitons"] == 1:
        rmse, _series, _U = RMSE_of_forecast(
            states_v0, U_star, x_full, t_np, coeffs, case["beta"], plot=False)
    else:
        rmse, _series, _U = RMSE_of_forecast_pooled(
            states_v0, U_star, x_full, t_np, coeffs, case["beta"], case["n_solitons"], plot=False)
    return float(rmse)


# --------------------------------------------------------------------------------------
# Crash-safe append-only JSONL log + resume bookkeeping.
# --------------------------------------------------------------------------------------
def _key(case_key, noise, seed):
    return (case_key, round(float(noise), 6), int(seed))


def load_completed(path):
    """Set of (case, noise, seed) keys already in the log; tolerant of a torn last line."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a power-cut may truncate the final line; skip it
            done.add(_key(r["case"], r["noise"], r["seed"]))
    return done


def append_result(path, record):
    """Append one JSON line and force it to disk so a later crash cannot lose it."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_results(path):
    """Read the JSONL log into a list of dicts (skips a torn final line)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a power-cut-truncated final line
    return rows


def aggregate_metric(rows, case, value_fn):
    """(noise, mean, std) over seeds for the ok rows of one case; None values dropped.

    `std` is the sample standard deviation (ddof=1); it is 0 at noise levels with a single
    sample (e.g. sigma=0, which collapses to one seed) so the band pinches to the mean there.
    `value_fn(record) -> float | None`; e.g. lambda r: r['E_inf'] or r['R2']['logc'].
    Dependency-free (no pandas) so it runs in the bare wsindy env.
    """
    by_noise = {}
    for r in rows:
        if r.get("case") != case or r.get("status") != "ok":
            continue
        v = value_fn(r)
        if v is None:
            continue
        by_noise.setdefault(round(float(r["noise"]), 6), []).append(float(v))
    noise = np.array(sorted(by_noise))
    if len(noise) == 0:
        empty = np.array([])
        return noise, empty, empty
    mean = np.array([np.mean(by_noise[n]) for n in noise])
    std = np.array([np.std(by_noise[n], ddof=1) if len(by_noise[n]) > 1 else 0.0
                    for n in noise])
    return noise, mean, std


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def config_signature(cfg):
    """The subset of config that must stay fixed across resumes for results to be comparable."""
    return dict(stride=cfg["stride"], min_kappa=cfg["min_kappa"],
                noise_step=cfg["noise_step"], noise_max=cfg["noise_max"],
                hyperparams=SWEEP_HYPERPARAMS, eps=EPS, logc_Lambda=LOGC_LAMBDA)


def _runs_path(out):
    base, _ext = os.path.splitext(out)
    return base + ".runs.jsonl"


def record_launch(out, cfg):
    """Append a per-launch metadata line; warn if config drifts from an earlier launch."""
    runs = _runs_path(out)
    sig = config_signature(cfg)

    prev_sig = None
    if os.path.exists(runs):
        with open(runs, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prev_sig = json.loads(line).get("signature")
                except json.JSONDecodeError:
                    pass
    if prev_sig is not None and prev_sig != sig:
        print("WARNING: sweep config differs from an earlier launch of this results file.\n"
              f"  previous: {prev_sig}\n  current:  {sig}\n"
              "  Mixing these into one file makes the metrics inconsistent. Consider a new --out.")

    entry = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                 host=socket.gethostname(),
                 python=platform.python_version(),
                 numpy=np.__version__, torch=torch.__version__,
                 git_commit=_git_commit(), signature=sig, config=cfg)
    with open(runs, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------------------
# Sweep driver.
# --------------------------------------------------------------------------------------
def make_config(out="outputs/noise_sweep_results.jsonl", noise_step=0.025, noise_max=0.5,
                noise_offset=0.0, n_seeds=5, seed_start=0, stride=2, min_kappa=0.6,
                cases=None, force=False):
    return dict(out=out, noise_step=noise_step, noise_max=noise_max, noise_offset=noise_offset,
                n_seeds=n_seeds, seed_start=seed_start,
                stride=stride, min_kappa=min_kappa,
                cases=list(cases) if cases else [c["key"] for c in CASES], force=bool(force))


def noise_grid(cfg):
    start = cfg.get("noise_offset", 0.0)
    return [round(float(v), 6)
            for v in np.arange(start, cfg["noise_max"] + cfg["noise_step"] / 2, cfg["noise_step"])]


def _base_record(case, noise, seed, cfg):
    return dict(case=case["key"], label=case["label"], perturbed=case["perturbed"],
                n_solitons=case["n_solitons"], noise=round(float(noise), 6), seed=int(seed),
                stride=cfg["stride"], min_kappa=cfg["min_kappa"],
                m=SWEEP_HYPERPARAMS["m"], p=SWEEP_HYPERPARAMS["p"], s=SWEEP_HYPERPARAMS["s"],
                rescale=SWEEP_HYPERPARAMS["rescale"], trigger_BIC=SWEEP_HYPERPARAMS["trigger_BIC"],
                beta=case["beta"], eps=(EPS if case["perturbed"] else None),
                logc_Lambda=(LOGC_LAMBDA if case["perturbed"] else None),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))


def run_one(case, noise, seed, cfg):
    """Compute every metric for a single (case, noise, seed); never raises."""
    record = _base_record(case, noise, seed, cfg)
    t0 = time.time()
    try:
        per_sim_states, t, n_diag = extract_states(
            case["config_files"], case["n_solitons"], noise, seed, cfg["stride"], cfg["min_kappa"])
        fit_states = build_fit_states(per_sim_states, case["n_solitons"])
        models, coeffs = run_fit(case, fit_states, t)
        TPR, E_inf, R2, bic, Lambda = compute_metrics(models, coeffs, case["w_truth"])
        rmse = compute_rmse(case, per_sim_states, coeffs, t)
        record.update(
            status="ok", TPR=TPR, E_inf=E_inf, R2=R2, bic_triggered=bic, RMSE_IST=rmse,
            Lambda=Lambda,
            soliton_count_ok=bool(all(np.all(a == case["n_solitons"]) for a in n_diag)),
            n_min=int(min(int(a.min()) for a in n_diag)),
            n_max=int(max(int(a.max()) for a in n_diag)),
            runtime_s=round(time.time() - t0, 2))
    except Exception as exc:  # log failures instead of aborting the sweep
        record.update(status="error", error=repr(exc), runtime_s=round(time.time() - t0, 2))
    return record


def run_sweep(cfg):
    """Run (or resume) the full sweep, appending one crash-safe JSONL line per combo."""
    out = cfg["out"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    record_launch(out, cfg)

    done = set() if cfg["force"] else load_completed(out)
    grid = noise_grid(cfg)
    seeds = list(range(cfg["seed_start"], cfg["seed_start"] + cfg["n_seeds"]))

    # Count the work up front so progress is meaningful.
    todo = []
    for case_key in cfg["cases"]:
        case = CASES_BY_KEY[case_key]
        for noise in grid:
            run_seeds = [0] if noise == 0 else seeds   # noise=0 is seed-independent
            for seed in run_seeds:
                if _key(case_key, noise, seed) not in done:
                    todo.append((case, noise, seed))

    print(f"Sweep: {len(cfg['cases'])} cases x {len(grid)} noise x {len(seeds)} seeds "
          f"({seeds[0]}..{seeds[-1]}) -> {len(todo)} combos to run "
          f"({len(done)} already done). out={out}")

    for i, (case, noise, seed) in enumerate(todo, start=1):
        rec = run_one(case, noise, seed, cfg)
        append_result(out, rec)
        extra = (f"TPR={rec['TPR']:.3f} E_inf={rec['E_inf']:.2e} RMSE={rec['RMSE_IST']:.2e} "
                 f"count_ok={rec['soliton_count_ok']}" if rec["status"] == "ok"
                 else f"ERROR {rec.get('error', '')}")
        print(f"[{i}/{len(todo)}] {case['key']:<26} noise={noise:.3f} seed={seed} "
              f"{rec['status']} ({rec['runtime_s']:.1f}s) {extra}")

    print(f"Done. Results in {out}")
    return out


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="WSINDy scattering-data noise sweep.")
    p.add_argument("--out", default="outputs/noise_sweep_results.jsonl")
    p.add_argument("--noise-step", type=float, default=0.025)
    p.add_argument("--noise-max", type=float, default=0.5)
    p.add_argument("--noise-offset", type=float, default=0.0,
                   help="First noise level; grid is arange(offset, max+step/2, step). "
                        "Use offset=step/2 to fill the half-points of an existing grid.")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0,
                   help="First RNG seed; seeds run are seed_start..seed_start+n_seeds-1. "
                        "Use to add seeds distinct from a prior run (e.g. --seed-start 5).")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--min-kappa", type=float, default=0.6)
    p.add_argument("--cases", nargs="+", choices=[c["key"] for c in CASES], default=None,
                   help="Subset of cases to run (default: all).")
    p.add_argument("--force", action="store_true", help="Recompute even completed combos.")
    return p.parse_args(argv)


def main(argv=None):
    matplotlib.use("Agg")  # headless standalone run: never require a display
    a = _parse_args(argv)
    cfg = make_config(out=a.out, noise_step=a.noise_step, noise_max=a.noise_max,
                      noise_offset=a.noise_offset, n_seeds=a.n_seeds, seed_start=a.seed_start,
                      stride=a.stride, min_kappa=a.min_kappa, cases=a.cases, force=a.force)
    run_sweep(cfg)


if __name__ == "__main__":
    main()
