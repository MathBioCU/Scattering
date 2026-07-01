"""Local extension of sangkuriang_ideal for perturbed KdV simulations.

The supported equation is

    u_t + eps*u*u_x + mu*u_xxx = eta*(x*u)_x,

where eta is the small perturbation parameter.  This module keeps the
initial-condition and output conventions used by sangkuriang_ideal while
adding only the perturbation term to the solver RHS.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from sangkuriang_ideal.cli import create_initial_condition, normalize_scenario_name
from sangkuriang_ideal.core.solver import KdVSolver
from sangkuriang_ideal.io.config_manager import ConfigManager
from sangkuriang_ideal.io.data_handler import DataHandler
from sangkuriang_ideal.visualization.animator import Animator


VALIDATION_ABS_TOL = 1e-5
VALIDATION_REL_L2_TOL = 1e-6

# ``sangkuriang_ideal`` was written against NumPy versions with ``np.trapz``.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


class PerturbedKdVSolver(KdVSolver):
    """KdV solver with the perturbation ``eta*(x*u)_x`` on the RHS."""

    def __init__(self, *args: Any, eta: float = 0.0, **kwargs: Any):
        self.eta = float(eta)
        super().__init__(*args, **kwargs)

    def kdv_rhs(self, t: float, u: np.ndarray, mu: float, eps: float) -> np.ndarray:
        """Evaluate ``u_t`` for perturbed KdV.

        The unperturbed ``eta=0`` path delegates exactly to the upstream solver.
        """
        base_rhs = super().kdv_rhs(t, u, mu, eps)

        if self.eta == 0.0:
            return base_rhs

        u_x = self.spatial_derivative(u, order=1)
        return base_rhs + self.eta*(u + self.x*u_x)

    def solve(
        self,
        u0: np.ndarray,
        mu: float,
        eps: float,
        t_final: float = 50.0,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        n_snapshots: int = 200,
        eta: float | None = None,
    ) -> dict[str, Any]:
        """Solve the perturbed KdV equation and attach ``eta`` to metadata."""
        if eta is not None:
            self.eta = float(eta)

        result = super().solve(
            u0=u0,
            mu=mu,
            eps=eps,
            t_final=t_final,
            rtol=rtol,
            atol=atol,
            n_snapshots=n_snapshots,
        )
        result["params"]["eta"] = self.eta
        return result


def load_config(config_path: str | Path, eta: float | None = None) -> dict[str, Any]:
    """Load a sangkuriang-style config and optionally override ``eta``."""
    config = ConfigManager.load(str(config_path))
    config.setdefault("eta", 0.0)
    if eta is not None:
        config["eta"] = float(eta)
    return config


def solve_from_config(
    config: dict[str, Any],
    verbose: bool = True,
    n_cores: int | None = None,
) -> dict[str, Any]:
    """Create a perturbed-KdV solution dictionary from a config."""
    solver = PerturbedKdVSolver(
        nx=config.get("nx", 512),
        x_min=config.get("x_min", -30.0),
        x_max=config.get("x_max", 30.0),
        verbose=verbose,
        n_cores=n_cores,
        eta=config.get("eta", 0.0),
    )
    u0 = create_initial_condition(config, solver.x)
    return solver.solve(
        u0=u0,
        mu=config.get("mu", 1.0),
        eps=config.get("eps", 6.0),
        eta=config.get("eta", 0.0),
        t_final=config.get("t_final", 1.5),
        rtol=config.get("rtol", 1e-10),
        atol=config.get("atol", 1e-12),
        n_snapshots=config.get("n_frames", 200),
    )


def run_perturbed_scenario(
    config: dict[str, Any],
    output_dir: str = "outputs",
    verbose: bool = True,
    n_cores: int | None = None,
    suffix: str | None = None,
) -> dict[str, Any]:
    """Run a perturbed-KdV scenario and optionally save NetCDF/GIF outputs."""
    result = solve_from_config(config, verbose=verbose, n_cores=n_cores)
    clean_name = normalize_scenario_name(config.get("scenario_name", "simulation"))
    if suffix:
        clean_name = f"{clean_name}_{suffix}"

    if config.get("save_netcdf", True):
        netcdf_name = f"{clean_name}.nc"
        DataHandler.save_netcdf(netcdf_name, result, config, output_dir)
        _annotate_perturbed_netcdf(Path(output_dir)/netcdf_name, config, result)

    if config.get("save_animation", False):
        Animator.create_gif(
            result,
            f"{clean_name}.gif",
            output_dir,
            config.get("scenario_name", "simulation"),
            fps=config.get("fps", 30),
            dpi=config.get("dpi", 150),
            view_3d=config.get("view_3d", True),
            colormap=config.get("colormap", "plasma"),
            line_width=config.get("line_width", 2.5),
            alpha=config.get("alpha", 0.9),
        )

    return result


def _annotate_perturbed_netcdf(
    filepath: str | Path,
    config: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Append perturbation metadata not known to upstream DataHandler."""
    with Dataset(filepath, "a") as nc:
        nc.eta = float(config.get("eta", result["params"].get("eta", 0.0)))
        nc.eta_description = "Perturbation coefficient in eta*(x*u)_x"
        nc.equation = "u_t + eps*u*u_x + mu*u_xxx = eta*(x*u)_x"


@dataclass
class ValidationResult:
    label: str
    reference_path: str
    shape: tuple[int, ...]
    max_abs_u_error: float
    max_relative_l2_error: float
    mass_error: float
    momentum_error: float
    energy_error: float

    @property
    def passed(self) -> bool:
        return (
            self.max_abs_u_error <= VALIDATION_ABS_TOL
            and self.max_relative_l2_error <= VALIDATION_REL_L2_TOL
        )


def validate_against_netcdf(
    label: str,
    config_path: str | Path,
    reference_path: str | Path,
    verbose: bool = False,
    n_cores: int | None = None,
) -> ValidationResult:
    """Run with ``eta=0`` and compare the solution to an existing NetCDF file."""
    config = load_config(config_path, eta=0.0)
    result = solve_from_config(config, verbose=verbose, n_cores=n_cores)

    with Dataset(reference_path) as nc:
        u_ref = nc["u"][:].data

    u = result["u"]
    diff = u - u_ref
    max_abs_u_error = float(np.max(np.abs(diff)))
    max_relative_l2_error = float(np.linalg.norm(diff.ravel())/np.linalg.norm(u_ref.ravel()))

    return ValidationResult(
        label=label,
        reference_path=str(reference_path),
        shape=tuple(u.shape),
        max_abs_u_error=max_abs_u_error,
        max_relative_l2_error=max_relative_l2_error,
        mass_error=float(result["mass_error"]),
        momentum_error=float(result["momentum_error"]),
        energy_error=float(result["energy_error"]),
    )


def validate_zero_eta(verbose: bool = False, n_cores: int | None = None) -> list[ValidationResult]:
    """Validate that ``eta=0`` reproduces the existing one- and two-soliton runs."""
    cases = [
        (
            "single_soliton",
            "configs/kdv_single_soliton_v0.txt",
            "outputs/kdv_single_soliton_v0.nc",
        ),
        (
            "two_soliton_collision",
            "configs/kdv_soliton_collision_v0.txt",
            "outputs/kdv_soliton_collision_v0.nc",
        ),
    ]
    return [
        validate_against_netcdf(label, config_path, reference_path, verbose, n_cores)
        for label, config_path, reference_path in cases
    ]


def _print_validation(results: list[ValidationResult]) -> None:
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.label}: {status}")
        print(f"  reference: {result.reference_path}")
        print(f"  shape: {result.shape}")
        print(f"  max_abs_u_error: {result.max_abs_u_error:.3e}")
        print(f"  max_relative_l2_error: {result.max_relative_l2_error:.3e}")
        print(f"  mass_error: {result.mass_error:.3e}")
        print(f"  momentum_error: {result.momentum_error:.3e}")
        print(f"  energy_error: {result.energy_error:.3e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Run a single scenario from this config file.")
    parser.add_argument("--eta", type=float, help="Override eta from the config.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--suffix", help="Optional suffix for saved output filenames.")
    parser.add_argument("--validate-zero-eta", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n-cores", type=int, default=None)
    args = parser.parse_args()

    if args.validate_zero_eta:
        results = validate_zero_eta(verbose=not args.quiet, n_cores=args.n_cores)
        _print_validation(results)
        return 0 if all(result.passed for result in results) else 1

    if not args.config:
        parser.error("Provide --config or --validate-zero-eta.")

    config = load_config(args.config, eta=args.eta)
    run_perturbed_scenario(
        config,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        n_cores=args.n_cores,
        suffix=args.suffix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
