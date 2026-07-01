# Core
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, trapezoid
from numpy.polynomial.legendre import leggauss


def laplacian_1D(n, dx):
    Dxx = np.zeros((n,n))
    i = np.arange(1, n-1)
    Dxx[i,i-1] =  1 / dx**2
    Dxx[i,i] = -2 / dx**2
    Dxx[i,i+1] =  1 / dx**2
    return Dxx


def solve_jost_bvp(x_grid, u_grid, kappa, tol=1e-5, max_nodes=25000, x_peak=None):
    """Solve -psi'' + 2*kappa*psi' - u(x)*psi = 0 with psi(b)=1, psi'(b)=0."""
    x_grid = np.asarray(x_grid)
    u_grid = np.asarray(u_grid)
    kappa = float(np.real_if_close(kappa, tol=1000))

    def potential(x_eval):
        return np.interp(x_eval, x_grid, u_grid)

    def ode(x_eval, y):
        psi = y[0]
        psi_x = y[1]
        return np.vstack((psi_x, 2.0*kappa*psi_x - potential(x_eval)*psi))

    def bc(ya, yb):
        return np.array([yb[0] - 1.0, yb[1]])

    y_guess = np.vstack((np.ones_like(x_grid), np.zeros_like(x_grid)))
    sol = solve_bvp(ode, bc, x_grid, y_guess, tol=tol, max_nodes=max_nodes)
    if not sol.success:
        print(f"Warning: solve_bvp did not converge for kappa={kappa:.8f}: {sol.message}")

    psi, psi_x = sol.sol(x_grid)
    phi_plus = np.exp(-kappa*x_grid)*psi
    weighted_norm = trapezoid(np.exp(-2.0*kappa*x_grid)*psi**2, x_grid)
    log_c = -np.log(weighted_norm)

    return {
        "kappa": kappa,
        "psi": psi,
        "psi_x": psi_x,
        "phi_plus": phi_plus,
        "weighted_norm": weighted_norm,
        "log_c": log_c,
        "c": np.exp(log_c),
        "x_peak": x_grid[np.argmax(u_grid)] if x_peak is None else x_peak,
        "sol": sol,
    }


def solve_jost_bvps(x_grid, u_grid, kappa_values, tol=1e-5, max_nodes=25000, x_peaks=None):
    """Solve the BVP for each kappa in kappa_values."""
    pos_real_kappas = np.real_if_close(np.asarray(kappa_values), tol=1000).astype(float)
    mask = np.isfinite(pos_real_kappas) & (pos_real_kappas > 0)
    pos_real_kappas = pos_real_kappas[mask]

    if x_peaks is not None:
        x_peaks = np.asarray(x_peaks)[mask]
    else:
        x_peaks = [None for _ in pos_real_kappas]

    return [
        solve_jost_bvp(x_grid, u_grid, kappa, tol=tol, max_nodes=max_nodes, x_peak=x_peak)
        for kappa,x_peak in zip(pos_real_kappas, x_peaks)
    ]


def compute_weighted_norm(x_grid, fcn, quad_method="cutoff_gauss"):
    kappa = fcn["kappa"]
    psi = fcn["psi"]

    if quad_method == "trapezoidal":
        return trapezoid(np.exp(-2.0*kappa*x_grid)*psi**2, x_grid)

    if quad_method == "cutoff_gauss":
        x_peak = fcn.get("x_peak", x_grid[0])
        x_cut = max(x_grid[0], x_peak - 4.0/kappa)
        z, w = leggauss(400)
        xq = 0.5*(x_grid[-1] - x_cut)*z + 0.5*(x_grid[-1] + x_cut)
        wq = 0.5*(x_grid[-1] - x_cut)*w
        psi = fcn["sol"].sol(xq)[0]
        return np.sum(wq*np.exp(-2.0*kappa*xq)*psi**2)

    raise ValueError("quad_method must be \"cutoff_gauss\" or \"trapezoidal\".")


def compute_norming_constants(x_grid, jost_fcns, quad_method="cutoff_gauss"):
    constants = {"kappa":[], "weighted_norm":[], "log_c":[], "c":[]}

    for fcn in jost_fcns:
        kappa = fcn["kappa"]
        weighted_norm = compute_weighted_norm(x_grid, fcn, quad_method=quad_method)
        log_c = -np.log(weighted_norm)
        c = np.exp(log_c)

        fcn["weighted_norm"] = weighted_norm
        fcn["log_c"] = log_c
        fcn["c"] = c

        constants["kappa"].append(kappa)
        constants["weighted_norm"].append(weighted_norm)
        constants["log_c"].append(log_c)
        constants["c"].append(c)

    return {key: np.asarray(value) for key, value in constants.items()}


def print_norming_constants(label, constants):
    print(label)
    for i, (kappa, weighted_norm, log_c, c) in enumerate(
        zip(constants["kappa"], constants["weighted_norm"], constants["log_c"], constants["c"]), start=1):
        print(f"i={i}: kappa={kappa:.4f}, integral={weighted_norm:.4e}, log c={log_c:.4f}, c={c:.4e}")


def IST_reflectionless(x_grid, kappas, cs):
    kappas = np.real_if_close(np.asarray(kappas), tol=1000).astype(float)
    cs = np.real_if_close(np.asarray(cs), tol=1000).astype(float)

    mask = np.isfinite(kappas) & np.isfinite(cs) & (kappas > 0) & (cs > 0)
    kappas = kappas[mask]
    cs = cs[mask]

    u_rec = np.zeros_like(x_grid, dtype=float)
    if len(kappas) == 0:
        return u_rec

    for m,xm in enumerate(x_grid):
        z = np.sqrt(cs)*np.exp(-kappas*xm)
        y = kappas*z
        A = np.eye(len(kappas)) + np.outer(z,z)/(kappas[:,None] + kappas[None,:])
        Ainv_z = np.linalg.solve(A,z)
        u_rec[m] = 4*y@Ainv_z - 2*(z@Ainv_z)**2

    return u_rec


class scattering_data:
    def __init__(self, x_grid, u_grid, tol=1e-5, max_nodes=25000, quad_method="cutoff_gauss", min_kappa=0.2, auto_compute=True):
        self.x = np.asarray(x_grid)
        self.u = np.asarray(u_grid)
        self.dx = self.x[1] - self.x[0]
        self.Nx = len(self.x)

        self.tol = tol
        self.max_nodes = max_nodes
        self.quad_method = quad_method
        self.min_kappa = min_kappa

        self.Dxx = None
        self.L = None
        self.L_interior = None
        self.Lambda = None
        self.Q = None
        self.n = None
        self.point_spectrum = None
        self.continuous_spectrum = None
        self.kappas = None
        self.jost_fcns = None
        self.norming_constants = None
        self.weighted_norms = None
        self.log_cs = None
        self.cs = None

        if auto_compute:
            self.compute_all()

    def compute_lax_operator(self):
        self.Dxx = laplacian_1D(self.Nx, self.dx)
        self.L = -(self.Dxx + np.diag(self.u))
        self.L_interior = self.L[1:-1, 1:-1]
        return self.L

    def compute_eigen_decomp(self):
        if self.L is None:
            self.compute_lax_operator()

        self.Lambda, self.Q = np.linalg.eigh(self.L_interior)
        return self.Lambda, self.Q

    def compute_spectrum(self):
        if self.Lambda is None:
            self.compute_eigen_decomp()

        self.n = np.sum(self.Lambda < -self.min_kappa**2)
        self.point_spectrum = self.Lambda[0:self.n]
        self.continuous_spectrum = self.Lambda[self.n:]
        return self.point_spectrum, self.continuous_spectrum

    def compute_kappas(self):
        if self.point_spectrum is None:
            self.compute_spectrum()

        self.kappas = np.sqrt(-self.point_spectrum)
        return self.kappas

    def solve_jost_bvps(self):
        if self.kappas is None:
            self.compute_kappas()

        x_peaks = None
        if self.Q is not None and self.n is not None and self.n > 0:
            x_peaks = self.x[1:-1][np.argmax(np.abs(self.Q[:, :self.n]), axis=0)]

        self.jost_fcns = solve_jost_bvps(self.x, self.u, self.kappas, tol=self.tol, max_nodes=self.max_nodes, x_peaks=x_peaks)
        return self.jost_fcns

    def compute_norming_constants(self):
        if self.jost_fcns is None:
            self.solve_jost_bvps()

        self.norming_constants = compute_norming_constants(self.x, self.jost_fcns, quad_method=self.quad_method)
        self.kappas = self.norming_constants["kappa"]
        self.weighted_norms = self.norming_constants["weighted_norm"]
        self.log_cs = self.norming_constants["log_c"]
        self.cs = self.norming_constants["c"]
        return self.norming_constants

    def compute_all(self):
        self.compute_lax_operator()
        self.compute_eigen_decomp()
        self.compute_spectrum()
        self.compute_kappas()
        self.solve_jost_bvps()
        self.compute_norming_constants()
        return self

    def print_norming_constants(self, label="Norming constants"):
        if self.norming_constants is None:
            self.compute_norming_constants()

        print_norming_constants(label, self.norming_constants)

    def plot_spectrum(self, num_continuous=25):
        if self.point_spectrum is None:
            self.compute_spectrum()

        continuous_spectrum = self.continuous_spectrum[:num_continuous]

        plt.figure(figsize=(7,1.8))
        plt.axhline(0, color='black', lw=0.7, alpha=0.7)
        plt.axvline(0, color='black', lw=0.7, alpha=0.7)
        plt.scatter(self.point_spectrum.real, self.point_spectrum.imag, color='red', marker='x', label='Point spectrum', zorder=10)
        plt.scatter(continuous_spectrum.real, continuous_spectrum.imag, color='C0', marker='.', label='Continuous spectrum', zorder=10)
        plt.xlabel(r'Re$(\lambda_i)$', loc='right', labelpad=-33)
        plt.ylabel(r'Im$(\lambda_i)$', loc='top', labelpad=-49)
        plt.grid(True, linestyle='-', alpha=0.3)
        plt.legend()
        plt.show()

    def plot_kappas(self):
        if self.kappas is None:
            self.compute_kappas()

        plt.figure(figsize=(7,1.8))
        plt.axhline(0, color='black', lw=0.7, alpha=0.7)
        plt.axvline(0, color='black', lw=0.7, alpha=0.7)
        plt.scatter(self.kappas.real, self.kappas.imag, color='C0', marker='.', s=80, label=r'$\kappa_i$', zorder=10)
        plt.xlabel(r'Re$(\kappa_i)$', loc='right', labelpad=-33)
        plt.ylabel(r'Im$(\kappa_i)$', loc='top', labelpad=-49)
        plt.grid(True, linestyle='-', alpha=0.3)
        plt.legend(loc='lower left')
        plt.show()

    def plot_jost_fcns(self):
        if self.jost_fcns is None:
            self.solve_jost_bvps()

        fig, axes = plt.subplots(1, 2, figsize=(9,3.5), sharex=True, sharey=True)

        for i,fcn in enumerate(self.jost_fcns, start=1):
            axes[0].plot(self.x, fcn["psi"], lw=1.0, label=rf"$\psi(x;\kappa_{i})$")
            axes[1].plot(self.x, fcn["phi_plus"], lw=1.0, label=rf"$\phi_{{+}}(x;\kappa_{i})$")

        axes[0].set_title(r"BVP solution, $\psi(x;\kappa_{i})$")
        axes[0].set_xlabel(r"$x$")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)

        axes[1].set_title(r"Jost solution, $\phi_+\!(x;\kappa_{i})$")
        axes[1].set_xlabel(r"$x$")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_norming_constants(self):
        if self.norming_constants is None:
            self.compute_norming_constants()

        plt.figure(figsize=(7,1.8))
        plt.axhline(0, color='black', lw=0.7, alpha=0.7)
        plt.axvline(0, color='black', lw=0.7, alpha=0.7)
        # plt.scatter(self.cs.real, self.cs.imag, color='C0', marker='.', s=80, label=r'$c_i$', zorder=10)
        plt.scatter(self.log_cs.real, self.log_cs.imag, color='C0', marker='.', s=80, label=r'$\log c_i$', zorder=10)
        plt.xlabel(r'Re$(\log c_i)$', loc='left', labelpad=-33)
        plt.ylabel(r'Im$(\log c_i)$', loc='top', labelpad=-49)
        plt.grid(True, linestyle='-', alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
        plt.show()

class scattering_data_sequence:
    def __init__(self, x_grid, U_grid, t_grid=None, x_axis=-1, snapshot_indices=None, auto_compute=True, **kwargs):
        self.x = np.asarray(x_grid)
        self.U_full = np.moveaxis(np.asarray(U_grid), x_axis, -1)
        if self.U_full.ndim != 2:
            raise ValueError('U_grid must be two-dimensional with one x-axis and one t-axis.')
        if self.U_full.shape[-1] != len(self.x):
            raise ValueError('The x-axis of U_grid must have the same length as x_grid.')

        t_full = np.arange(self.U_full.shape[0]) if t_grid is None else np.asarray(t_grid)
        if len(t_full) != self.U_full.shape[0]:
            raise ValueError('t_grid must have the same length as the time axis of U_grid.')

        if snapshot_indices is None:
            self.snapshot_indices = np.arange(self.U_full.shape[0])
        else:
            self.snapshot_indices = np.arange(self.U_full.shape[0])[snapshot_indices]

        self.t = t_full[self.snapshot_indices]
        self.U = self.U_full[self.snapshot_indices]
        self.scattering_kwargs = kwargs
        self.snapshots = [None for _ in range(len(self.t))]

        self.n = None
        self.kappas = None
        self.point_spectrum = None
        self.continuous_spectrum = None
        self.norming_constants = None
        self.weighted_norms = None
        self.log_cs = None
        self.cs = None

        if auto_compute:
            self.compute_all()

    def __len__(self):
        return len(self.t)

    def __getitem__(self, j):
        if self.snapshots[j] is None:
            return self.compute_snapshot(j)
        return self.snapshots[j]

    def compute_snapshot(self, j):
        self.snapshots[j] = scattering_data(self.x, self.U[j], **self.scattering_kwargs)
        self.collect()
        return self.snapshots[j]

    def compute_all(self):
        for j in range(len(self.t)):
            if self.snapshots[j] is None:
                self.snapshots[j] = scattering_data(self.x, self.U[j], **self.scattering_kwargs)
        self.collect()
        return self

    def collect(self):
        self.n = np.asarray([0 if sd is None else sd.n for sd in self.snapshots])
        self.kappas = [None if sd is None else sd.kappas for sd in self.snapshots]
        self.point_spectrum = [None if sd is None else sd.point_spectrum for sd in self.snapshots]
        self.continuous_spectrum = [None if sd is None else sd.continuous_spectrum for sd in self.snapshots]
        self.norming_constants = [None if sd is None else sd.norming_constants for sd in self.snapshots]
        self.weighted_norms = [None if sd is None else sd.weighted_norms for sd in self.snapshots]
        self.log_cs = [None if sd is None else sd.log_cs for sd in self.snapshots]
        self.cs = [None if sd is None else sd.cs for sd in self.snapshots]
        return self
