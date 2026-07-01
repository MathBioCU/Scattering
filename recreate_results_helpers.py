# Helper utilities for the recreate_paper_results notebook.
import torch
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from scipy.integrate import solve_ivp
from IPython.display import display, Math

from wsindy_ode import *
from compute_scattering_data import scattering_data_sequence, IST_reflectionless

__all__ = [
  'compute_tpr', 'to_torch_solution', 'add_noise_to_states', 'plot_states',
  'load_kdv_field', 'scattering_series', 'scattering_series_n',
  'fit_wsindy_system', 'fit_wsindy_stacked', 'fit_wsindy_bic', 'fit_wsindy_bic_stacked',
  'fit_wsindy_bic_stacked_eq',
  'RMSE_of_forecast', 'RMSE_of_forecast_pooled',
]


# True positive ratio (TPR)
def compute_tpr(w, w_true):
  w_nonzero = (w != 0)
  w_true_nonzero = (w_true != 0)

  TP = torch.sum(w_nonzero & w_true_nonzero).item()
  FN = torch.sum(~w_nonzero & w_true_nonzero).item()
  FP = torch.sum(w_nonzero & ~w_true_nonzero).item()

  TPR = TP / (TP + FN + FP)
  return TPR

def to_torch_solution(sol):
  return [torch.tensor(ui, dtype=torch.float64) for ui in sol.y]

def add_noise_to_states(states, noise):
  if noise == 0:
    return states
  return [add_noise(ui, noise) for ui in states]

def plot_states(t, states, names, title):
  plt.figure(figsize=(7,3))
  for ui,name in zip(states,names):
    plt.plot(t, ui, label='$' + name + '(t)$')
  plt.xlabel('$t$')
  plt.title(title)
  plt.grid(True, alpha=0.3)
  plt.legend(loc='upper left')
  plt.show()

# Utilities for running WSINDy on stacked scattering data...
def load_kdv_field(path):
  with Dataset(path) as nc:
    x = nc['x'][:].data
    t = nc['t'][:].data
    U = nc['u'][:].data
  return x, t, U

def scattering_series(x, U, t, noise=0.0, min_kappa=0.5):
  U = torch.tensor(U, dtype=torch.float64)
  if noise > 0:
    U = add_noise(U, noise)
  seq = scattering_data_sequence(x, U.numpy(), t_grid=t, min_kappa=min_kappa)

  kappa = torch.tensor([float(k[0]) for k in seq.kappas], dtype=torch.float64)
  log_c = torch.tensor([float(c[0]) for c in seq.log_cs], dtype=torch.float64)
  return [kappa, log_c], seq


# Fits a WSINDy model to a system of ODEs
def fit_wsindy_system(states, t, names, beta, m, p, s, Lambda=1e-3, rescale=True, verbosity=False):
  alpha = [[1], [0]]

  models = []
  odes = []

  for i,Ui in enumerate(states):
    aux_inds = [j for j in range(len(states)) if j != i]
    V = [states[j] for j in aux_inds]
    local_names = [names[i]] + [names[j] for j in aux_inds]

    model = WSINDy(Ui, alpha, beta, t, V=V, names=local_names,
                   m=m, p=p, s=s, verbosity=verbosity, rescale=rescale)

    [G,powers,derivs,rhs_names] = model.create_default_library()
    display(Math(r'\Theta_{' + names[i] + r'}=' + r'\{' + r', \, '.join(rhs_names) + r'\}'))

    model.build_lhs(names[i] + model.derivative_names[0])
    model.set_library(G, powers, derivs, rhs_names)

    _ = model.MSTLS(Lambda=Lambda)
    model.print_report()

    models.append(model)
    odes.append(symbolic_ode(model.lhs_name, model.rhs_names, model.coeffs))

  for ode in odes:
    display(Math(ode))
  return models,odes

def fit_wsindy_stacked(sim_states, t, names, beta, m, p, s, Lambda=None, rescale=True, verbosity=False):
  alpha = [[1], [0]]
  n_states = len(sim_states[0])
  models, odes, coeffs = [], [], []

  for i in range(n_states):
    libraries, lhs_blocks, container, rhs_names = [], [], None, None

    for states in sim_states:
      aux_inds = [j for j in range(n_states) if j != i]
      V = [states[j] for j in aux_inds]
      local_names = [names[i]] + [names[j] for j in aux_inds]

      model = WSINDy(states[i], alpha, beta, t, V=V, names=local_names,
                     m=m, p=p, s=s, verbosity=verbosity, rescale=rescale)

      [G, powers, derivs, rhs] = model.create_default_library()
      model.build_lhs(names[i] + model.derivative_names[0])
      model.set_library(G, powers, derivs, rhs)

      libraries.append(model.library)
      lhs_blocks.append(model.lhs)
      if container is None:
        container, rhs_names = model, rhs

    display(Math(r'\Theta_{' + names[i] + r'}=' + r'\{' + r', \, '.join(rhs_names) + r'\}'))

    # Stack the weak-form systems and solve a single regression
    container.library = torch.cat(libraries, dim=0)
    container.lhs = torch.cat(lhs_blocks, dim=0)

    _ = container.MSTLS(Lambda=Lambda)
    container.print_report()

    models.append(container)
    coeffs.append(container.coeffs)
    odes.append(symbolic_ode(container.lhs_name, rhs_names, container.coeffs))

  for ode in odes:
    display(Math(ode))
  return models, odes, coeffs


def fit_wsindy_bic(states, t, names, beta, m, p, s, rescale=False, verbosity=False, **bic_kwargs):
  alpha = [[1], [0]]
  models, odes, coeffs = [], [], []

  for i,Ui in enumerate(states):
    aux_inds = [j for j in range(len(states)) if j != i]
    V = [states[j] for j in aux_inds]
    local_names = [names[i]] + [names[j] for j in aux_inds]

    model = WSINDy(Ui, alpha, beta, t, V=V, names=local_names,
                   m=m, p=p, s=s, verbosity=verbosity, rescale=rescale)

    [G,powers,derivs,rhs_names] = model.create_default_library()
    display(Math(r'\Theta_{' + names[i] + r'}=' + r'\{' + r', \, '.join(rhs_names) + r'\}'))

    model.build_lhs(names[i] + model.derivative_names[0])
    model.set_library(G, powers, derivs, rhs_names)

    _ = model.MSTLS_with_BIC_testing(**bic_kwargs)
    model.print_report()

    models.append(model)
    coeffs.append(model.coeffs)
    odes.append(symbolic_ode(model.lhs_name, model.rhs_names, model.coeffs))

  for ode in odes:
    display(Math(ode + (' 0' if ode.rstrip().endswith('=') else '')))  # show "= 0" for trivial
  return models, odes, coeffs


# Multi-soliton scattering data: returns [kappa_1, ..., kappa_n, log c_1, ..., log c_n]
def scattering_series_n(x, U, t, n_solitons, noise=0.0, min_kappa=0.5):
  U = torch.tensor(U, dtype=torch.float64)
  if noise > 0:
    U = add_noise(U, noise)
  seq = scattering_data_sequence(x, U.numpy(), t_grid=t, min_kappa=min_kappa)

  kappas, log_cs = [], []
  for k,lc in zip(seq.kappas, seq.log_cs):
    order = np.argsort(k)[::-1][:n_solitons]   # n largest kappas, descending
    kappas.append(k[order]); log_cs.append(lc[order])
  kappas, log_cs = np.array(kappas), np.array(log_cs)   # (n_t, n_solitons)

  states = [torch.tensor(kappas[:, i], dtype=torch.float64) for i in range(n_solitons)] \
         + [torch.tensor(log_cs[:, i], dtype=torch.float64) for i in range(n_solitons)]
  return states, seq

# Stacked + BIC-testing
def fit_wsindy_bic_stacked(sim_states, t, names, beta, m, p, s, rescale=False, verbosity=False, **bic_kwargs):
  alpha = [[1], [0]]
  n_states = len(sim_states[0])
  models, odes, coeffs = [], [], []

  for i in range(n_states):
    libraries, lhs_blocks, container, rhs_names = [], [], None, None

    for states in sim_states:
      aux_inds = [j for j in range(n_states) if j != i]
      V = [states[j] for j in aux_inds]
      local_names = [names[i]] + [names[j] for j in aux_inds]

      model = WSINDy(states[i], alpha, beta, t, V=V, names=local_names,
                     m=m, p=p, s=s, verbosity=verbosity, rescale=rescale)

      [G, powers, derivs, rhs] = model.create_default_library()
      model.build_lhs(names[i] + model.derivative_names[0])
      model.set_library(G, powers, derivs, rhs)

      libraries.append(model.library)
      lhs_blocks.append(model.lhs)
      if container is None:
        container, rhs_names = model, rhs

    display(Math(r'\Theta_{' + names[i] + r'}=' + r'\{' + r', \, '.join(rhs_names) + r'\}'))

    # Stack the weak-form systems and solve a single BIC-tested regression
    container.library = torch.cat(libraries, dim=0)
    container.lhs = torch.cat(lhs_blocks, dim=0)

    _ = container.MSTLS_with_BIC_testing(**bic_kwargs)
    container.print_report()

    models.append(container)
    coeffs.append(container.coeffs)
    odes.append(symbolic_ode(container.lhs_name, rhs_names, container.coeffs))

  for ode in odes:
    display(Math(ode + (' 0' if ode.rstrip().endswith('=') else '')))  # show "= 0" for trivial
  return models, odes, coeffs


def fit_wsindy_bic_stacked_eq(sim_states, t, names, beta, eq, m, p, s, rescale=False,
                              verbosity=False, **bic_kwargs):
  alpha = [[1], [0]]
  n_states = len(sim_states[0])
  i = eq
  libraries, lhs_blocks, container, rhs_names = [], [], None, None

  for states in sim_states:
    aux_inds = [j for j in range(n_states) if j != i]
    V = [states[j] for j in aux_inds]
    local_names = [names[i]] + [names[j] for j in aux_inds]

    model = WSINDy(states[i], alpha, beta, t, V=V, names=local_names,
                   m=m, p=p, s=s, verbosity=verbosity, rescale=rescale)

    [G, powers, derivs, rhs] = model.create_default_library()
    model.build_lhs(names[i] + model.derivative_names[0])
    model.set_library(G, powers, derivs, rhs)

    libraries.append(model.library)
    lhs_blocks.append(model.lhs)
    if container is None:
      container, rhs_names = model, rhs

  display(Math(r'\Theta_{' + names[i] + r'}=' + r'\{' + r', \, '.join(rhs_names) + r'\}'))

  # Stack the weak-form systems and solve a single BIC-tested regression
  container.library = torch.cat(libraries, dim=0)
  container.lhs = torch.cat(lhs_blocks, dim=0)

  _ = container.MSTLS_with_BIC_testing(**bic_kwargs)
  container.print_report()

  ode = symbolic_ode(container.lhs_name, rhs_names, container.coeffs)
  display(Math(ode + (' 0' if ode.rstrip().endswith('=') else '')))
  return container, ode, container.coeffs


def _forecast_rhs(y, coeffs, beta):
  n = len(coeffs)
  per_eq = np.ndim(beta[0]) == 2   # list-of-libraries vs single shared library
  y = np.asarray(y)
  dydt = np.empty(n)
  for i in range(n):
    order = [i] + [j for j in range(n) if j != i]
    bi = np.asarray(beta[i] if per_eq else beta)
    monomials = np.prod(y[order]**bi, axis=1)
    dydt[i] = np.asarray(coeffs[i]) @ monomials
  return dydt

def RMSE_of_forecast(states, U_star, x, t, coeffs, beta, plot=True):
  x = np.asarray(x)
  t = np.asarray(t, dtype=float)
  U_star = np.asarray(U_star)
  n_solitons = len(states) // 2

  coeffs = [np.asarray(w, dtype=float) for w in coeffs]
  # Forward-simulate from the true initial conditions with RK4 (scipy's RK45)
  y0 = np.array([float(s[0]) for s in states])
  sol = solve_ivp(lambda _t, y: _forecast_rhs(y, coeffs, beta), (t[0], t[-1]), y0,
                  t_eval=t, method='RK45', rtol=1e-10, atol=1e-12)
  Y = sol.y

  # Reconstruct each forecast snapshot with the reflectionless (tau-determinant) IST
  kappas_t = Y[:n_solitons].T
  cs_t = np.exp(Y[n_solitons:2*n_solitons].T)
  U_IST = np.array([IST_reflectionless(x, kappas_t[k], cs_t[k]) for k in range(len(t))])

  # Pointwise RMSE: per-snapshot and total = ||U* - U_IST||_F / sqrt(M N)
  err = U_star - U_IST
  rmse_series = np.sqrt(np.mean(err**2, axis=1))
  total_rmse = float(np.sqrt(np.mean(err**2)))

  if plot:
    _plot_forecast(x, t, U_star, U_IST, rmse_series)
  return total_rmse, rmse_series, U_IST

def RMSE_of_forecast_pooled(states, U_star, x, t, coeffs, beta, n_solitons, plot=True):
  x = np.asarray(x)
  t = np.asarray(t, dtype=float)
  U_star = np.asarray(U_star)

  coeffs = [np.asarray(w, dtype=float) for w in coeffs]

  # Forward-simulate each soliton independently with the shared (kappa, log c) law (RK4)
  kappas_t = np.empty((len(t), n_solitons))
  logcs_t = np.empty((len(t), n_solitons))
  for i in range(n_solitons):
    y0 = np.array([float(states[i][0]), float(states[n_solitons + i][0])])
    sol = solve_ivp(lambda _t, y: _forecast_rhs(y, coeffs, beta), (t[0], t[-1]), y0,
                    t_eval=t, method='RK45', rtol=1e-10, atol=1e-12)
    kappas_t[:, i], logcs_t[:, i] = sol.y[0], sol.y[1]

  # Reconstruct each forecast snapshot with the reflectionless (tau-determinant) IST
  cs_t = np.exp(logcs_t)
  U_IST = np.array([IST_reflectionless(x, kappas_t[k], cs_t[k]) for k in range(len(t))])

  # Pointwise RMSE: per-snapshot and total = ||U* - U_IST||_F / sqrt(M N)
  err = U_star - U_IST
  rmse_series = np.sqrt(np.mean(err**2, axis=1))
  total_rmse = float(np.sqrt(np.mean(err**2)))

  if plot:
    _plot_forecast(x, t, U_star, U_IST, rmse_series)
  return total_rmse, rmse_series, U_IST

def _plot_forecast(x, t, U_star, U_IST, rmse_series):
  vmin = min(U_star.min(), U_IST.min())
  vmax = max(U_star.max(), U_IST.max())

  fig = plt.figure(figsize=(12, 4))
  gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.4, wspace=0.3)
  ax0 = fig.add_subplot(gs[0, 0])
  ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
  ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
  axb = fig.add_subplot(gs[1, :])

  ax0.pcolormesh(t, x, U_star.T, cmap='ocean_r', shading='gouraud', vmin=vmin, vmax=vmax)
  ax0.set_title(r'Clean data, $u^{*}(x,t)$')
  ax0.set_xlabel(r'$t$')
  ax0.set_ylabel(r'$x$')

  pc1 = ax1.pcolormesh(t, x, U_IST.T, cmap='ocean_r', shading='gouraud', vmin=vmin, vmax=vmax)
  ax1.set_title(r'IST Reconstruction, $u_{\rm IST}(x,t)$')
  ax1.set_xlabel(r'$t$')
  fig.colorbar(pc1, ax=[ax0, ax1])   # shared colorbar keeps the two equal width

  pc2 = ax2.pcolormesh(t, x, np.abs(U_star - U_IST).T, cmap='magma_r', shading='gouraud')
  ax2.set_title(r'Error, $|u^{*} - u_{\rm IST}|(x,t)$')
  ax2.set_xlabel(r'$t$')
  fig.colorbar(pc2, ax=ax2)

  axb.plot(t, rmse_series, color='darkred', lw=1.5)
  axb.set_xlim(t[0], t[-1])
  axb.set_yscale('log')
  axb.set_xlabel(r'$t$')
  axb.set_ylabel('RMSE')
  axb.grid(True, alpha=0.3)

  plt.show()
