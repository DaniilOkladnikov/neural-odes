import os
import argparse
import time
import numpy as np

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dynamics import get_system, SYSTEMS
from models import (ODEFunc, SequencePredictor, MLPPredictor,
                    RunningAverageMeter,
                    count_parameters, compute_default_hidden_sizes)

# ---------------------------------------------------------------------------
# Two-stage CLI argument parsing
# ---------------------------------------------------------------------------
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument('--system', type=str, default='double_pendulum',
                        choices=list(SYSTEMS.keys()))
pre_args, _ = pre_parser.parse_known_args()

system = get_system(pre_args.system)
state_dim = system['state_dim']
angle_indices = system.get('angle_indices', [])
repr_dim = state_dim + len(angle_indices)  # e.g. 4 + 2 = 6 for double pendulum

parser = argparse.ArgumentParser(
    f'Neural ODE vs RNN/LSTM/GRU on {system["name"]}'
)
parser.add_argument('--system', type=str, default='double_pendulum',
                    choices=list(SYSTEMS.keys()))
parser.add_argument('--models', type=str, default='both',
                    choices=['node', 'rnn', 'mlp', 'both'],
                    help='Which models to train: node (Neural ODE only), '
                         'rnn (RNN/GRU/LSTM only), mlp (MLP only), both (default)')
# ODE solver
parser.add_argument('--method', type=str,
                    choices=['dopri5', 'adams', 'euler', 'rk4'], default='dopri5')
parser.add_argument('--step_size', type=float, default=0.005,
                    help='Step size for fixed-step solvers (euler, rk4)')
parser.add_argument('--adjoint', action='store_true')
# Dataset
parser.add_argument('--n_trajectories', type=int, default=50000)
parser.add_argument('--traj_time', type=float, default=1.0,
                    help='Duration of each trajectory in seconds')
parser.add_argument('--traj_dt', type=float, default=0.05,
                    help='Timestep within each trajectory')
parser.add_argument('--E_max', type=float, default=None,
                    help='Max energy for IC sampling (auto-computed if not given)')
parser.add_argument('--test_split', type=float, default=0.1)
parser.add_argument('--gen_batch_size', type=int, default=500,
                    help='Batch size for trajectory generation (memory)')
# Training
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--niters', type=int, default=20000)
parser.add_argument('--eta_max', type=float, default=3e-4,
                    help='Initial / maximum learning rate')
parser.add_argument('--eta_min', type=float, default=1e-5,
                    help='Minimum learning rate for cosine annealing')
parser.add_argument('--lr_restart_period', type=int, default=20000,
                    help='Period for cosine annealing warm restarts (iterations)')
parser.add_argument('--energy_weight', type=float, default=0.1)
parser.add_argument('--energy_warmup', type=int, default=5000,
                    help='Iterations before energy loss starts ramping up')
parser.add_argument('--energy_rampup', type=int, default=5000,
                    help='Iterations over which energy weight ramps from 0 to target')
# Jacobian regularization (Neural ODE only)
parser.add_argument('--jac_weight', type=float, default=0.01,
                    help='Jacobian Frobenius norm penalty weight (0 to disable)')
parser.add_argument('--jac_warmup', type=int, default=2000,
                    help='Iterations before Jacobian loss starts ramping up')
parser.add_argument('--jac_rampup', type=int, default=5000,
                    help='Iterations over which Jacobian weight ramps from 0 to target')
parser.add_argument('--jac_samples', type=int, default=256,
                    help='Points subsampled from trajectory for Jacobian estimation')
# Derivative matching loss (Neural ODE only)
parser.add_argument('--traj_weight', type=float, default=1.0,
                    help='Trajectory MSE loss weight')
parser.add_argument('--deriv_weight', type=float, default=0.1,
                    help='Derivative matching loss weight (0 to disable)')
parser.add_argument('--deriv_warmup', type=int, default=0,
                    help='Iterations before derivative loss starts ramping up')
parser.add_argument('--deriv_rampup', type=int, default=0,
                    help='Iterations over which derivative weight ramps from 0 to target')
parser.add_argument('--deriv_samples', type=int, default=256,
                    help='Points subsampled from trajectory for derivative matching')
# Horizon curriculum
parser.add_argument('--horizon_start_iter', type=int, default=5000,
                    help='Iteration at which to start increasing training horizon')
parser.add_argument('--horizon_rate', type=float, default=0.2,
                    help='Horizon increase rate in seconds per 1000 iterations')
parser.add_argument('--horizon_max', type=float, default=1.0,
                    help='Maximum training horizon in seconds')
# Monitoring
parser.add_argument('--log_every', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=1000)
parser.add_argument('--div_threshold', type=float, default=0.5,
                    help='Divergence threshold in radians')
parser.add_argument('--n_lyap', type=int, default=100,
                    help='Number of samples for FTLE Lyapunov computation')
parser.add_argument('--lyap_delta', type=float, default=1e-5,
                    help='Perturbation magnitude for FTLE computation')
parser.add_argument('--no_wandb', action='store_true')
# Model sizes
parser.add_argument('--target_params', type=int, default=100000,
                    help='Target parameter count for auto-computed hidden sizes')
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--node_hidden', type=int, default=None)
parser.add_argument('--rnn_hidden', type=int, default=None)
parser.add_argument('--gru_hidden', type=int, default=None)
parser.add_argument('--lstm_hidden', type=int, default=None)
parser.add_argument('--mlp_hidden', type=int, default=None)
# Output
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--resume_iter', type=int, default=0,
                    help='Iteration to resume training from')
parser.add_argument('--constant_lr', action='store_true',
                    help='Use constant learning rate instead of cosine annealing')
parser.add_argument('--wandb_id', type=str, default=None,
                    help='Wandb run ID to resume (e.g. golden-haze-28)')
parser.add_argument('--seed', type=int, default=42)
# System-specific args
system['add_cli_args'](parser)

args = parser.parse_args()

# Fill in hidden sizes that weren't explicitly set
H_node, H_rnn, H_gru, H_lstm, H_mlp = compute_default_hidden_sizes(
    repr_dim, args.num_layers, args.target_params)
if args.node_hidden is None:
    args.node_hidden = H_node
if args.rnn_hidden is None:
    args.rnn_hidden = H_rnn
if args.gru_hidden is None:
    args.gru_hidden = H_gru
if args.lstm_hidden is None:
    args.lstm_hidden = H_lstm
if args.mlp_hidden is None:
    args.mlp_hidden = H_mlp

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Options for fixed-step solvers
if args.method in ('euler', 'rk4'):
    solver_kwargs = dict(method=args.method, options={'step_size': args.step_size})
else:
    solver_kwargs = dict(method=args.method)

device = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')

output_dir = f'png_{args.system}'
model_dir = f'models_{args.system}'


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# ---------------------------------------------------------------------------
# Angle → sin/cos representation
# ---------------------------------------------------------------------------
def angles_to_sincos(y):
    """Replace angle dims with (sin, cos) pairs. (..., state_dim) → (..., repr_dim)."""
    if not angle_indices:
        return y
    pieces = []
    for i in range(state_dim):
        if i in angle_indices:
            pieces.append(torch.sin(y[..., i:i+1]))
            pieces.append(torch.cos(y[..., i:i+1]))
        else:
            pieces.append(y[..., i:i+1])
    return torch.cat(pieces, dim=-1)


def sincos_to_angles(y):
    """Inverse: (sin, cos) pairs → angles via atan2. (..., repr_dim) → (..., state_dim)."""
    if not angle_indices:
        return y
    pieces = []
    j = 0
    for i in range(state_dim):
        if i in angle_indices:
            pieces.append(torch.atan2(y[..., j:j+1], y[..., j+1:j+2]))
            j += 2
        else:
            pieces.append(y[..., j:j+1])
            j += 1
    return torch.cat(pieces, dim=-1)


# ---------------------------------------------------------------------------
# Generate dataset: energy-based IC sampling + batch integration
# ---------------------------------------------------------------------------
sample_ics_fn = system['sample_ics_fn']
compute_E_max_fn = system['compute_E_max_fn']

if sample_ics_fn is None:
    raise ValueError(
        f"System '{args.system}' does not support energy-based IC sampling."
    )

dynamics = system['dynamics_fn'](args)

if args.E_max is None:
    args.E_max = compute_E_max_fn(args.m1, args.m2, args.l1, args.l2, args.g)

# Evaluation time vector (initial horizon)
n_time_points = int(round(args.traj_time / args.traj_dt)) + 1
t = torch.linspace(0., args.traj_time, n_time_points).to(device)

# Compute max training horizon for data generation (horizon curriculum)
if args.horizon_start_iter < args.niters:
    iters_of_increase = args.niters - args.horizon_start_iter
    max_train_horizon = min(
        args.traj_time + args.horizon_rate * iters_of_increase / 1000,
        args.horizon_max
    )
else:
    max_train_horizon = args.traj_time
gen_horizon = max(args.traj_time, max_train_horizon)
n_gen_points = int(round(gen_horizon / args.traj_dt)) + 1
t_gen = torch.linspace(0., gen_horizon, n_gen_points).to(device)

print(f'Sampling {args.n_trajectories} initial conditions '
      f'with E in [0, {args.E_max:.2f}] ...')
all_y0 = sample_ics_fn(
    args.n_trajectories, args.E_max,
    args.m1, args.m2, args.l1, args.l2, args.g
).to(device)

print(f'Integrating {args.n_trajectories} trajectories '
      f'({n_gen_points} steps, dt={args.traj_dt}s, horizon={gen_horizon:.1f}s) ...')
all_trajectories = []
for i in range(0, args.n_trajectories, args.gen_batch_size):
    batch_y0 = all_y0[i:i + args.gen_batch_size]
    with torch.no_grad():
        batch_traj = odeint(dynamics, batch_y0, t_gen, **solver_kwargs)
    all_trajectories.append(batch_traj)
    done = min(i + args.gen_batch_size, args.n_trajectories)
    if done % 2000 == 0 or done == args.n_trajectories:
        print(f'  {done}/{args.n_trajectories}')

all_traj_phys = torch.cat(all_trajectories, dim=1)  # (T, N, state_dim) — 4D physical
all_y0_phys = all_y0  # (N, state_dim) — keep 4D for evaluation

# Convert to sin/cos representation for training
all_traj = angles_to_sincos(all_traj_phys)  # (T, N, repr_dim)
all_y0 = angles_to_sincos(all_y0_phys)      # (N, repr_dim)

# Train / test split
n_test = max(1, int(args.n_trajectories * args.test_split))
n_train = args.n_trajectories - n_test
perm = torch.randperm(args.n_trajectories)
train_idx, test_idx = perm[:n_train], perm[n_train:]

train_traj = all_traj[:, train_idx, :]
test_traj = all_traj[:, test_idx, :]
train_y0 = all_y0[train_idx]
test_y0 = all_y0[test_idx]

# Keep 4D physical ICs for evaluation metrics (energy, divergence)
train_y0_phys = all_y0_phys[train_idx]
test_y0_phys = all_y0_phys[test_idx]

print(f'Dataset: {n_train} train / {n_test} test trajectories')

# ---------------------------------------------------------------------------
# Data normalization
# ---------------------------------------------------------------------------
y_mean = train_traj.mean(dim=(0, 1), keepdim=True)   # (1, 1, repr_dim)
y_std = train_traj.std(dim=(0, 1), keepdim=True) + 1e-8

train_traj_norm = (train_traj - y_mean) / y_std
test_traj_norm = (test_traj - y_mean) / y_std
train_y0_norm = (train_y0 - y_mean.squeeze(0)) / y_std.squeeze(0)
test_y0_norm = (test_y0 - y_mean.squeeze(0)) / y_std.squeeze(0)


def denormalize(y):
    return y * y_std + y_mean


def get_batch(n_points=None):
    if n_points is None:
        n_points = n_time_points
    idx = np.random.choice(n_train, args.batch_size, replace=False)
    batch_y0 = train_y0_norm[idx]
    batch_t = t_gen[:n_points]
    batch_y = train_traj_norm[:n_points, idx, :]
    batch_y0_phys = train_y0_phys[idx]
    return batch_y0, batch_t, batch_y, batch_y0_phys


# ---------------------------------------------------------------------------
# Jacobian regularization (Lipschitz continuity)
# ---------------------------------------------------------------------------
def compute_jac_reg(func_net, pred_y, n_samples=256):
    """Estimate mean ||df/dy||_F^2 via Hutchinson's stochastic trace estimator.

    Detaches trajectory points from the ODE solver graph and re-forwards
    through the network to build a fresh autograd graph for the VJP.
    """
    T, B, D = pred_y.shape
    flat_y = pred_y.detach().reshape(-1, D)
    n_total = flat_y.shape[0]
    n_samples = min(n_samples, n_total)
    idx = torch.randperm(n_total, device=flat_y.device)[:n_samples]
    y = flat_y[idx].requires_grad_(True)

    f = func_net(y)                               # (n_samples, D)
    v = torch.randn_like(f)                        # random projection
    vjp, = torch.autograd.grad(f, y, v, create_graph=True)
    return torch.mean(vjp ** 2)


# ---------------------------------------------------------------------------
# Derivative matching (true dynamics supervision)
# ---------------------------------------------------------------------------
def compute_true_deriv_normalized(y_norm, dynamics_fn, y_mean_sq, y_std_sq):
    """Compute true dy_norm/dt at sampled points in normalized sin/cos space.

    Converts to physical space, calls analytical dynamics, then applies the
    chain rule through the sin/cos transformation and normalization.
    """
    # Denormalize to sin/cos space
    y_sc = y_norm * y_std_sq + y_mean_sq                  # (N, repr_dim)

    # Recover physical angles for the dynamics call
    y_phys = sincos_to_angles(y_sc)                        # (N, state_dim)

    # True derivative in physical space (autonomous, t=0)
    dy_phys = dynamics_fn(0.0, y_phys)                     # (N, state_dim)

    # Chain rule: convert dy_phys → dy_sincos
    pieces = []
    j = 0
    for i in range(state_dim):
        if i in angle_indices:
            sin_val = y_sc[..., j:j+1]
            cos_val = y_sc[..., j+1:j+2]
            dtheta = dy_phys[..., i:i+1]
            pieces.append(cos_val * dtheta)                # d(sin θ)/dt
            pieces.append(-sin_val * dtheta)               # d(cos θ)/dt
            j += 2
        else:
            pieces.append(dy_phys[..., i:i+1])
            j += 1
    dy_sc = torch.cat(pieces, dim=-1)                      # (N, repr_dim)

    # Normalize the derivative
    return dy_sc / y_std_sq


def compute_deriv_loss(func_net, traj_data, dynamics_fn, y_mean_sq, y_std_sq,
                       n_samples=256):
    """Derivative matching loss: MSE between network and true vector field.

    Samples random (time, trajectory) pairs from the full pre-computed
    training dataset for broad state-space coverage.
    """
    T, N = traj_data.shape[:2]
    t_idx = torch.randint(T, (n_samples,), device=traj_data.device)
    n_idx = torch.randint(N, (n_samples,), device=traj_data.device)
    y_sample = traj_data[t_idx, n_idx]                     # (n_samples, repr_dim)

    pred_deriv = func_net(y_sample)                        # network prediction
    with torch.no_grad():
        true_deriv = compute_true_deriv_normalized(
            y_sample, dynamics_fn, y_mean_sq, y_std_sq)

    return torch.mean((pred_deriv - true_deriv) ** 2)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------
energy_fn = system['energy_fn']
has_energy = energy_fn is not None


def compute_energy_violation(pred, y0, energy_fn, args):
    """Mean |E(t) - E(0)| over time and samples."""
    E_pred = energy_fn(pred, args)                    # (T, N)
    E0 = energy_fn(y0.unsqueeze(0), args)             # (1, N)
    return torch.mean(torch.abs(E_pred - E0)).item()


def compute_divergence_time(pred, true, t_array, threshold):
    """Mean time until max angle error exceeds threshold."""
    angle_error = torch.max(
        torch.abs(pred[..., 0] - true[..., 0]),
        torch.abs(pred[..., 1] - true[..., 1])
    )  # (T, N)
    exceeded = angle_error > threshold
    first_idx = exceeded.float().argmax(dim=0)        # (N,)
    never = ~exceeded.any(dim=0)                       # (N,)
    div_t = t_array[first_idx]
    div_t[never] = t_array[-1]
    return div_t.mean().item()


def compute_ftle(y_final, y_pert_final, delta_norms, T):
    """Finite-Time Lyapunov Exponent for each sample.

    Args:
        y_final:       (N, state_dim) final state of original trajectories
        y_pert_final:  (N, state_dim) final state of perturbed trajectories
        delta_norms:   (N,) ||delta_0|| for each sample
        T:             float, integration time
    Returns:
        ftle: (N,) tensor of FTLE values
    """
    diff = y_pert_final - y_final
    # Wrap angle dimensions to [-pi, pi] to avoid 2pi jumps
    for idx in angle_indices:
        diff[..., idx] = torch.atan2(torch.sin(diff[..., idx]), torch.cos(diff[..., idx]))
    separation = torch.norm(diff, dim=-1)           # (N,)
    ratio = separation / delta_norms.clamp(min=1e-30)
    return (1.0 / T) * torch.log(ratio.clamp(min=1e-30))  # (N,)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    makedirs(output_dir)
    makedirs(model_dir)

    # --- Build models (repr_dim inputs/outputs) ---
    train_node = args.models in ('node', 'both')
    train_rnn = args.models in ('rnn', 'both')
    train_mlp = args.models in ('mlp', 'both')

    models = {}
    seq_models = {}
    names = []
    colors = {'Neural ODE': 'blue', 'RNN': 'red', 'GRU': 'orange',
              'LSTM': 'purple', 'MLP': 'green'}

    if train_node:
        node_func = ODEFunc(repr_dim, args.node_hidden, args.num_layers).to(device)
        models['Neural ODE'] = node_func
        names.append('Neural ODE')

    if train_rnn:
        rnn_model = SequencePredictor(repr_dim, args.rnn_hidden, 'rnn',
                                      args.num_layers).to(device)
        gru_model = SequencePredictor(repr_dim, args.gru_hidden, 'gru',
                                      args.num_layers).to(device)
        lstm_model = SequencePredictor(repr_dim, args.lstm_hidden, 'lstm',
                                       args.num_layers).to(device)
        seq_models = {'RNN': rnn_model, 'GRU': gru_model, 'LSTM': lstm_model}
        models.update(seq_models)
        names.extend(['RNN', 'GRU', 'LSTM'])

    if train_mlp:
        mlp_model = MLPPredictor(repr_dim, args.mlp_hidden,
                                 args.num_layers).to(device)
        seq_models['MLP'] = mlp_model
        models['MLP'] = mlp_model
        names.append('MLP')

    print('=' * 50)
    print(f'System: {system["name"]} ({state_dim}D state -> {repr_dim}D sin/cos repr)')
    for name, model in models.items():
        print(f'  {name:12s}: {count_parameters(model)} params')
    print('=' * 50)

    # ------------------------------------------------------------------
    # Fixed evaluation sets
    # ------------------------------------------------------------------
    n_eval = min(1000, n_test)
    eval_perm = torch.randperm(n_test)
    eval_idx = eval_perm[:n_eval]
    eval_traj_norm = test_traj_norm[:n_time_points, eval_idx, :]  # (T_init, n_eval, repr_dim) — normalized 6D
    eval_y0_phys = test_y0_phys[eval_idx]            # (n_eval, state_dim) — 4D for energy
    eval_y0_norm = test_y0_norm[eval_idx]

    # Divergence time: full eval set, 10s simulation
    n_div = n_eval
    div_idx = eval_idx[:n_div]
    div_y0_phys = test_y0_phys[div_idx]           # 4D for odeint with physical dynamics
    div_y0_norm = test_y0_norm[div_idx]
    t_long = torch.linspace(0., 10.0, int(10.0 / args.traj_dt) + 1).to(device)
    n_long = len(t_long)

    print(f'Pre-computing 10s ground truth for {n_div} divergence-time samples ...')
    div_true_chunks = []
    with torch.no_grad():
        for i in range(0, n_div, args.gen_batch_size):
            chunk = odeint(dynamics, div_y0_phys[i:i + args.gen_batch_size],
                           t_long, **solver_kwargs)
            chunk[..., 0] = torch.atan2(torch.sin(chunk[..., 0]), torch.cos(chunk[..., 0]))
            chunk[..., 1] = torch.atan2(torch.sin(chunk[..., 1]), torch.cos(chunk[..., 1]))
            div_true_chunks.append(chunk)
            print(f'  {min(i + args.gen_batch_size, n_div)}/{n_div}')
        div_true = torch.cat(div_true_chunks, dim=1)  # (T_long, n_div, 4)

    # FTLE (Lyapunov) pre-computation
    n_lyap = min(args.n_lyap, n_eval)
    lyap_idx = eval_idx[:n_lyap]
    lyap_y0_phys = test_y0_phys[lyap_idx]              # (n_lyap, 4)
    lyap_y0_norm = test_y0_norm[lyap_idx]              # (n_lyap, 6)

    # Random perturbation directions in 4D physical space
    lyap_gen = torch.Generator(device=device).manual_seed(args.seed + 999)
    lyap_directions = torch.randn(n_lyap, state_dim, device=device, generator=lyap_gen)
    lyap_directions = lyap_directions / torch.norm(lyap_directions, dim=-1, keepdim=True)
    lyap_delta_vecs = args.lyap_delta * lyap_directions       # (n_lyap, 4)
    lyap_delta_norms = torch.norm(lyap_delta_vecs, dim=-1)    # (n_lyap,)

    # Perturbed ICs → normalized sin/cos representation
    lyap_y0_pert_phys = lyap_y0_phys + lyap_delta_vecs        # (n_lyap, 4)
    lyap_y0_pert_sincos = angles_to_sincos(lyap_y0_pert_phys) # (n_lyap, 6)
    lyap_y0_pert_norm = (lyap_y0_pert_sincos - y_mean.squeeze(0)) / y_std.squeeze(0)

    T_lyap = t[-1].item()  # args.traj_time

    print(f'Pre-computing true FTLE for {n_lyap} samples ...')
    with torch.no_grad():
        lyap_true_orig_chunks = []
        for i in range(0, n_lyap, args.gen_batch_size):
            chunk = odeint(dynamics, lyap_y0_phys[i:i + args.gen_batch_size],
                           t, **solver_kwargs)
            lyap_true_orig_chunks.append(chunk[-1])            # final time: (B, 4)
        lyap_true_orig_final = torch.cat(lyap_true_orig_chunks, dim=0)

        lyap_true_pert_chunks = []
        for i in range(0, n_lyap, args.gen_batch_size):
            chunk = odeint(dynamics, lyap_y0_pert_phys[i:i + args.gen_batch_size],
                           t, **solver_kwargs)
            lyap_true_pert_chunks.append(chunk[-1])
        lyap_true_pert_final = torch.cat(lyap_true_pert_chunks, dim=0)

        true_ftle = compute_ftle(lyap_true_orig_final, lyap_true_pert_final,
                                 lyap_delta_norms, T_lyap)
        print(f'  True FTLE: mean={true_ftle.mean().item():.4f}, '
              f'std={true_ftle.std().item():.4f}')

    # Trajectory viz: 3 fixed ICs, 5s horizon
    n_viz = 3
    viz_time = 5.0
    n_viz_points = int(round(viz_time / args.traj_dt)) + 1
    t_viz = torch.linspace(0., viz_time, n_viz_points).to(device)
    viz_y0_phys = test_y0_phys[0:n_viz]           # (3, 4)
    viz_y0_norm = test_y0_norm[0:n_viz]            # (3, 6)
    with torch.no_grad():
        viz_true = odeint(dynamics, viz_y0_phys, t_viz, **solver_kwargs)  # (T_viz, 3, 4)
        viz_true[..., 0] = torch.atan2(torch.sin(viz_true[..., 0]), torch.cos(viz_true[..., 0]))
        viz_true[..., 1] = torch.atan2(torch.sin(viz_true[..., 1]), torch.cos(viz_true[..., 1]))

    # ------------------------------------------------------------------
    # wandb init
    # ------------------------------------------------------------------
    wb_prefix = {
        'Neural ODE': 'neural_ode', 'RNN': 'rnn',
        'GRU': 'gru', 'LSTM': 'lstm', 'MLP': 'mlp',
    }
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb_kwargs = dict(project='neural-odes-dp', config=vars(args))
        if args.wandb_id:
            wandb_kwargs['id'] = args.wandb_id
            wandb_kwargs['resume'] = 'must'
        wandb.init(**wandb_kwargs)
        for name in names:
            prefix = wb_prefix[name]
            wandb.define_metric(f'{prefix}/step')
            wandb.define_metric(f'{prefix}/*', step_metric=f'{prefix}/step')
    elif not args.no_wandb:
        print('Warning: wandb not installed, logging disabled')

    # ------------------------------------------------------------------
    # Real-time plot setup
    # ------------------------------------------------------------------
    fig_batch, (ax_bmse, ax_nfe) = plt.subplots(1, 2, figsize=(12, 4))
    fig_test, (ax_tmse, ax_ev, ax_dt, ax_ftle) = plt.subplots(1, 4, figsize=(20, 4))
    state_labels = system['state_labels']
    n_viz_cols = state_dim + (1 if has_energy else 0)  # 4 state vars + energy
    fig_traj, axes_traj = plt.subplots(n_viz, n_viz_cols, figsize=(4 * n_viz_cols, 3 * n_viz))
    fig_traj.suptitle(f'Fixed Trajectories ({viz_time}s)')

    log = {name: {
        'iters_fast': [], 'batch_mse': [], 'nfe': [],
        'iters_slow': [], 'test_mse': [], 'energy_viol': [], 'div_time': [],
        'ftle_error': [],
    } for name in names}

    def update_batch_plots(model_name):
        ax_bmse.cla()
        ax_bmse.set_title('Batch MSE')
        ax_bmse.set_xlabel('Iteration')
        ax_bmse.set_yscale('log')
        d = log[model_name]
        if d['iters_fast']:
            ax_bmse.plot(d['iters_fast'], d['batch_mse'],
                         color=colors[model_name], label=model_name, alpha=0.8)
        ax_bmse.legend(fontsize=7)

        ax_nfe.cla()
        if model_name == 'Neural ODE':
            ax_nfe.set_title('NFE (Neural ODE)')
            ax_nfe.set_xlabel('Iteration')
            if d['nfe']:
                ax_nfe.plot(d['iters_fast'][:len(d['nfe'])], d['nfe'], color='blue')
        else:
            ax_nfe.set_title('NFE')
            ax_nfe.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=14, transform=ax_nfe.transAxes, color='gray')

        fig_batch.suptitle(f'Training {model_name} (every {args.log_every} iters)')
        fig_batch.tight_layout()

    def update_test_plots(model_name):
        d = log[model_name]
        for ax, key, title, yscale in [
            (ax_tmse, 'test_mse', 'Test MSE', 'log'),
            (ax_ev, 'energy_viol', 'Energy Violation', 'log'),
            (ax_dt, 'div_time', 'Mean Divergence Time (s)', 'linear'),
            (ax_ftle, 'ftle_error', 'FTLE Error', 'log'),
        ]:
            ax.cla()
            ax.set_title(title)
            ax.set_xlabel('Iteration')
            if yscale == 'log':
                ax.set_yscale('log')
            if d['iters_slow']:
                ax.plot(d['iters_slow'], d[key],
                        color=colors[model_name], label=model_name,
                        alpha=0.8, marker='o', markersize=3)
            ax.legend(fontsize=7)

        fig_test.suptitle(f'Test Eval — {model_name} (every {args.eval_every} iters)')
        fig_test.tight_layout()

    def update_traj_plot(model_name, viz_pred):
        """Update the fixed trajectory figure (3 trajectories x 5 columns)."""
        t_np = t_viz.cpu().numpy()
        col_labels = list(state_labels) + (['Energy'] if has_energy else [])

        for ax in axes_traj.flat:
            ax.cla()

        for row in range(n_viz):
            true_np = viz_true[:, row, :].cpu().numpy()
            pred_np = viz_pred[:, row, :].cpu().numpy()

            for col in range(state_dim):
                ax = axes_traj[row, col]
                ax.plot(t_np, true_np[:, col], 'k-', lw=1.5, label='True')
                ax.plot(t_np, pred_np[:, col], '--', color=colors[model_name],
                        lw=1.2, label=model_name)
                if row == 0:
                    ax.set_title(col_labels[col])
                if row == n_viz - 1:
                    ax.set_xlabel('t (s)')
                if col == 0:
                    ax.set_ylabel(f'Traj {row + 1}')
                ax.legend(fontsize=6)

            if has_energy:
                ax_e = axes_traj[row, state_dim]
                E_true = energy_fn(viz_true[:, row:row + 1, :], args)[:, 0].cpu().numpy()
                E_pred = energy_fn(viz_pred[:, row:row + 1, :], args)[:, 0].cpu().numpy()
                ax_e.plot(t_np, E_true, 'k-', lw=1.5, label='True')
                ax_e.plot(t_np, E_pred, '--', color=colors[model_name],
                          lw=1.2, label=model_name)
                if row == 0:
                    ax_e.set_title('Energy')
                if row == n_viz - 1:
                    ax_e.set_xlabel('t (s)')
                ax_e.legend(fontsize=6)

        fig_traj.suptitle(f'Fixed Trajectories — {model_name}')
        fig_traj.tight_layout()

        if use_wandb:
            prefix = wb_prefix[model_name]
            wandb.log({f'{prefix}/trajectory_plot': wandb.Image(fig_traj),
                       f'{prefix}/step': log[model_name]['iters_slow'][-1]})

    # ------------------------------------------------------------------
    # Helper: full evaluation on test set
    # ------------------------------------------------------------------
    def evaluate_node(func):
        """Evaluate Neural ODE on fixed eval sets."""
        with torch.no_grad():
            # Test MSE in normalized space (same scale as batch MSE)
            chunks = []
            for i in range(0, n_eval, args.gen_batch_size):
                c = eval_y0_norm[i:i + args.gen_batch_size]
                chunks.append(odeint(func, c, t, **solver_kwargs))
            eval_pred_norm = torch.cat(chunks, dim=1)
            test_mse = torch.mean((eval_pred_norm - eval_traj_norm) ** 2).item()

            # Energy violation needs 4D physical state
            eval_pred_4d = sincos_to_angles(denormalize(eval_pred_norm))
            ev = compute_energy_violation(eval_pred_4d, eval_y0_phys, energy_fn, args) if has_energy else 0.0

            # Divergence time (10s) — batched
            div_pred_chunks = []
            for i in range(0, n_div, args.gen_batch_size):
                c = div_y0_norm[i:i + args.gen_batch_size]
                div_pred_chunks.append(sincos_to_angles(denormalize(
                    odeint(func, c, t_long, **solver_kwargs))))
            div_pred = torch.cat(div_pred_chunks, dim=1)
            dt_val = compute_divergence_time(div_pred, div_true, t_long, args.div_threshold)

            # FTLE (Lyapunov) error
            lyap_orig_chunks = []
            for i in range(0, n_lyap, args.gen_batch_size):
                c = lyap_y0_norm[i:i + args.gen_batch_size]
                chunk = odeint(func, c, t, **solver_kwargs)
                lyap_orig_chunks.append(sincos_to_angles(denormalize(chunk)[-1]))
            lyap_pred_orig_final = torch.cat(lyap_orig_chunks, dim=0)

            lyap_pert_chunks = []
            for i in range(0, n_lyap, args.gen_batch_size):
                c = lyap_y0_pert_norm[i:i + args.gen_batch_size]
                chunk = odeint(func, c, t, **solver_kwargs)
                lyap_pert_chunks.append(sincos_to_angles(denormalize(chunk)[-1]))
            lyap_pred_pert_final = torch.cat(lyap_pert_chunks, dim=0)

            pred_ftle = compute_ftle(lyap_pred_orig_final, lyap_pred_pert_final,
                                     lyap_delta_norms, T_lyap)
            ftle_err = torch.mean(torch.abs(pred_ftle - true_ftle)).item()

            # Trajectory viz — return 4D for plotting (5s)
            viz_pred_4d = sincos_to_angles(denormalize(odeint(func, viz_y0_norm, t_viz, **solver_kwargs)))

        return test_mse, ev, dt_val, ftle_err, viz_pred_4d

    def evaluate_seq(model):
        """Evaluate sequence model on fixed eval sets."""
        with torch.no_grad():
            chunks = []
            for i in range(0, n_eval, args.gen_batch_size):
                c = eval_y0_norm[i:i + args.gen_batch_size]
                chunks.append(model.predict_trajectory(c, n_time_points))
            eval_pred_norm = torch.cat(chunks, dim=1)
            test_mse = torch.mean((eval_pred_norm - eval_traj_norm) ** 2).item()

            eval_pred_4d = sincos_to_angles(denormalize(eval_pred_norm))
            ev = compute_energy_violation(eval_pred_4d, eval_y0_phys, energy_fn, args) if has_energy else 0.0

            # Divergence time (10s) — batched
            div_pred_chunks = []
            for i in range(0, n_div, args.gen_batch_size):
                c = div_y0_norm[i:i + args.gen_batch_size]
                div_pred_chunks.append(sincos_to_angles(denormalize(
                    model.predict_trajectory(c, n_long))))
            div_pred = torch.cat(div_pred_chunks, dim=1)
            dt_val = compute_divergence_time(div_pred, div_true, t_long, args.div_threshold)

            # FTLE (Lyapunov) error
            lyap_orig_chunks = []
            for i in range(0, n_lyap, args.gen_batch_size):
                c = lyap_y0_norm[i:i + args.gen_batch_size]
                pred = model.predict_trajectory(c, n_time_points)
                lyap_orig_chunks.append(sincos_to_angles(denormalize(pred)[-1]))
            lyap_pred_orig_final = torch.cat(lyap_orig_chunks, dim=0)

            lyap_pert_chunks = []
            for i in range(0, n_lyap, args.gen_batch_size):
                c = lyap_y0_pert_norm[i:i + args.gen_batch_size]
                pred = model.predict_trajectory(c, n_time_points)
                lyap_pert_chunks.append(sincos_to_angles(denormalize(pred)[-1]))
            lyap_pred_pert_final = torch.cat(lyap_pert_chunks, dim=0)

            pred_ftle = compute_ftle(lyap_pred_orig_final, lyap_pred_pert_final,
                                     lyap_delta_norms, T_lyap)
            ftle_err = torch.mean(torch.abs(pred_ftle - true_ftle)).item()

            viz_pred_4d = sincos_to_angles(denormalize(model.predict_trajectory(viz_y0_norm, n_viz_points)))

        return test_mse, ev, dt_val, ftle_err, viz_pred_4d

    # ==================================================================
    # Train Neural ODE
    # ==================================================================
    train_time_node = 0.0
    if train_node:
        print('\n=== Training Neural ODE ===')
        node_optimizer = optim.Adam(node_func.parameters(), lr=args.eta_max,
                                weight_decay=1e-4)
        
        if args.resume_iter > 0:
            ckpt_path = os.path.join(model_dir, 'checkpoints', f'neural_ode_iter{args.resume_iter}.pt')
            if os.path.exists(ckpt_path):
                print(f'Loading checkpoint: {ckpt_path}')
                node_func.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                print(f'Warning: Checkpoint {ckpt_path} not found')

        if args.constant_lr:
            node_scheduler = None
        else:
            node_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                node_optimizer, T_0=args.lr_restart_period, T_mult=1, eta_min=args.eta_min
            )
        start = time.time()

        for itr in range(args.resume_iter + 1, args.resume_iter + args.niters + 1):
            node_optimizer.zero_grad()
            rel_itr = itr - args.resume_iter

            # Horizon curriculum
            if rel_itr < args.horizon_start_iter:
                current_horizon = args.traj_time
            else:
                current_horizon = min(
                    args.traj_time + args.horizon_rate * (rel_itr - args.horizon_start_iter) / 1000,
                    args.horizon_max
                )
            n_current_points = int(round(current_horizon / args.traj_dt)) + 1

            batch_y0, batch_t, batch_y, batch_y0_phys = get_batch(n_current_points)

            node_func.nfe = 0

            # Optimization: Only solve ODE if required for loss or logging
            need_solve_grad = (args.traj_weight > 0) or (args.energy_weight > 0) or (args.jac_weight > 0)
            need_solve_log = (itr % args.log_every == 0)

            if need_solve_grad:
                pred_y = odeint(node_func, batch_y0, batch_t, **solver_kwargs).to(device)
            elif need_solve_log:
                with torch.no_grad():
                    pred_y = odeint(node_func, batch_y0, batch_t, **solver_kwargs).to(device)
            else:
                pred_y = None

            nfe = node_func.nfe

            if args.traj_weight > 0:
                traj_loss = torch.mean((pred_y - batch_y) ** 2)
            else:
                traj_loss = torch.tensor(0.0, device=device)

            # Energy conservation loss
            if args.energy_weight > 0:
                pred_4d = sincos_to_angles(denormalize(pred_y))
                E_pred = energy_fn(pred_4d, args)                   # (T, batch)
                E0 = energy_fn(batch_y0_phys.unsqueeze(0), args)    # (1, batch)
                energy_loss = torch.mean(torch.abs(E_pred - E0))
            else:
                energy_loss = torch.tensor(0.0, device=device)

            # Jacobian regularization (Lipschitz continuity)
            if args.jac_weight > 0:
                jac_loss = compute_jac_reg(node_func.net, pred_y,
                                           n_samples=args.jac_samples)
            else:
                jac_loss = torch.tensor(0.0, device=device)

            # Derivative matching loss
            if args.deriv_weight > 0:
                y_mean_sq = y_mean.squeeze()
                y_std_sq = y_std.squeeze()
                deriv_loss = compute_deriv_loss(node_func.net, train_traj_norm, dynamics,
                                                y_mean_sq, y_std_sq,
                                                n_samples=args.deriv_samples)
            else:
                deriv_loss = torch.tensor(0.0, device=device)

            # Curriculum: energy weight = 0 during warmup, linear ramp, then constant
            if rel_itr <= args.energy_warmup:
                ew = 0.0
            elif rel_itr <= args.energy_warmup + args.energy_rampup:
                ew = args.energy_weight * (rel_itr - args.energy_warmup) / args.energy_rampup
            else:
                ew = args.energy_weight

            # Jacobian weight curriculum
            if rel_itr <= args.jac_warmup:
                jw = 0.0
            elif rel_itr <= args.jac_warmup + args.jac_rampup:
                jw = args.jac_weight * (rel_itr - args.jac_warmup) / args.jac_rampup
            else:
                jw = args.jac_weight

            # Derivative matching weight curriculum
            if rel_itr <= args.deriv_warmup:
                dw = 0.0
            elif rel_itr <= args.deriv_warmup + args.deriv_rampup:
                dw = args.deriv_weight * (rel_itr - args.deriv_warmup) / args.deriv_rampup
            else:
                dw = args.deriv_weight

            loss = args.traj_weight * traj_loss + ew * energy_loss + jw * jac_loss + dw * deriv_loss
            loss.backward()
            clip_grad_norm_(node_func.parameters(), max_norm=1.0)
            node_optimizer.step()
            if node_scheduler:
                node_scheduler.step()

            if itr % args.log_every == 0:
                with torch.no_grad():
                    if pred_y is not None:
                        batch_mse = torch.mean((pred_y - batch_y) ** 2).item()
                    else:
                        batch_mse = 0.0

                log['Neural ODE']['iters_fast'].append(itr)
                log['Neural ODE']['batch_mse'].append(batch_mse)
                log['Neural ODE']['nfe'].append(nfe)

                print(f'  Iter {itr:5d} | MSE {batch_mse:.6f} | '
                      f'E_loss {energy_loss.item():.4f} | ew {ew:.4f} | '
                      f'J_loss {jac_loss.item():.4f} | jw {jw:.4f} | '
                      f'D_loss {deriv_loss.item():.4f} | dw {dw:.4f} | '
                      f'NFE {nfe} | horizon {current_horizon:.2f}s')

                if use_wandb:
                    wandb.log({
                        'neural_ode/batch_mse': batch_mse,
                        'neural_ode/energy_loss': energy_loss.item(),
                        'neural_ode/energy_weight': ew,
                        'neural_ode/jac_loss': jac_loss.item(),
                        'neural_ode/jac_weight': jw,
                        'neural_ode/deriv_loss': deriv_loss.item(),
                        'neural_ode/deriv_weight': dw,
                        'neural_ode/lr': node_optimizer.param_groups[0]['lr'],
                        'neural_ode/nfe': nfe,
                        'neural_ode/train_horizon': current_horizon,
                        'neural_ode/step': itr,
                    })

                update_batch_plots('Neural ODE')


            if itr % args.eval_every == 0:
                test_mse, ev, dt_val, ftle_err, viz_pred = evaluate_node(node_func)

                log['Neural ODE']['iters_slow'].append(itr)
                log['Neural ODE']['test_mse'].append(test_mse)
                log['Neural ODE']['energy_viol'].append(ev)
                log['Neural ODE']['div_time'].append(dt_val)
                log['Neural ODE']['ftle_error'].append(ftle_err)

                print(f'    [eval] Test MSE {test_mse:.6f} | '
                      f'Energy Viol {ev:.4f} | Div Time {dt_val:.3f}s | '
                      f'FTLE Error {ftle_err:.4f}')

                if use_wandb:
                    wandb.log({
                        'neural_ode/test_mse': test_mse,
                        'neural_ode/energy_violation': ev,
                        'neural_ode/divergence_time': dt_val,
                        'neural_ode/ftle_error': ftle_err,
                        'neural_ode/step': itr,
                    })

                update_test_plots('Neural ODE')
                update_traj_plot('Neural ODE', viz_pred)


            if itr % 1000 == 0:
                ckpt_dir = os.path.join(model_dir, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(node_func.state_dict(),
                           os.path.join(ckpt_dir, f'neural_ode_iter{itr}.pt'))
                print(f'    [checkpoint] Neural ODE saved at iter {itr}')

        train_time_node = time.time() - start

    # ==================================================================
    # Train sequence models (RNN, GRU, LSTM)
    # ==================================================================
    train_times_seq = {}

    for model_name, seq_model in seq_models.items():
        print(f'\n=== Training {model_name} ===')
        optimizer = optim.Adam(seq_model.parameters(), lr=args.eta_max,
                               weight_decay=1e-4)
        
        if args.resume_iter > 0:
            fname = model_name.lower().replace(' ', '_')
            ckpt_path = os.path.join(model_dir, 'checkpoints', f'{fname}_iter{args.resume_iter}.pt')
            if os.path.exists(ckpt_path):
                print(f'Loading checkpoint: {ckpt_path}')
                seq_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                print(f'Warning: Checkpoint {ckpt_path} not found')

        if args.constant_lr:
            scheduler = None
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=args.lr_restart_period, T_mult=1, eta_min=args.eta_min
            )
        start = time.time()
        prefix = wb_prefix[model_name]

        for itr in range(args.resume_iter + 1, args.resume_iter + args.niters + 1):
            optimizer.zero_grad()
            rel_itr = itr - args.resume_iter

            # Horizon curriculum
            if rel_itr < args.horizon_start_iter:
                current_horizon = args.traj_time
            else:
                current_horizon = min(
                    args.traj_time + args.horizon_rate * (rel_itr - args.horizon_start_iter) / 1000,
                    args.horizon_max
                )
            n_current_points = int(round(current_horizon / args.traj_dt)) + 1

            batch_y0, batch_t, batch_y, batch_y0_phys = get_batch(n_current_points)
            rnn_pred = seq_model.autoregressive_rollout(batch_y0, n_current_points)
            if args.traj_weight > 0:
                traj_loss = torch.mean((rnn_pred - batch_y) ** 2)
            else:
                traj_loss = torch.tensor(0.0, device=device)

            # Energy conservation loss
            if args.energy_weight > 0:
                pred_4d = sincos_to_angles(denormalize(rnn_pred[1:]))
                E_pred = energy_fn(pred_4d, args)
                E0 = energy_fn(batch_y0_phys.unsqueeze(0), args)
                energy_loss = torch.mean(torch.abs(E_pred - E0))
            else:
                energy_loss = torch.tensor(0.0, device=device)

            # Curriculum: energy weight = 0 during warmup, linear ramp, then constant
            if rel_itr <= args.energy_warmup:
                ew = 0.0
            elif rel_itr <= args.energy_warmup + args.energy_rampup:
                ew = args.energy_weight * (rel_itr - args.energy_warmup) / args.energy_rampup
            else:
                ew = args.energy_weight

            loss = args.traj_weight * traj_loss + ew * energy_loss
            loss.backward()
            clip_grad_norm_(seq_model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            if itr % args.log_every == 0:
                with torch.no_grad():
                    batch_mse = torch.mean((rnn_pred - batch_y) ** 2).item()

                log[model_name]['iters_fast'].append(itr)
                log[model_name]['batch_mse'].append(batch_mse)

                print(f'  Iter {itr:5d} | MSE {batch_mse:.6f} | '
                      f'E_loss {energy_loss.item():.4f} | ew {ew:.4f} | '
                      f'horizon {current_horizon:.2f}s')

                if use_wandb:
                    wandb.log({
                        f'{prefix}/batch_mse': batch_mse,
                        f'{prefix}/energy_loss': energy_loss.item(),
                        f'{prefix}/energy_weight': ew,
                        f'{prefix}/lr': optimizer.param_groups[0]['lr'],
                        f'{prefix}/train_horizon': current_horizon,
                        f'{prefix}/step': itr,
                    })

                update_batch_plots(model_name)


            if itr % args.eval_every == 0:
                test_mse, ev, dt_val, ftle_err, viz_pred = evaluate_seq(seq_model)

                log[model_name]['iters_slow'].append(itr)
                log[model_name]['test_mse'].append(test_mse)
                log[model_name]['energy_viol'].append(ev)
                log[model_name]['div_time'].append(dt_val)
                log[model_name]['ftle_error'].append(ftle_err)

                print(f'    [eval] Test MSE {test_mse:.6f} | '
                      f'Energy Viol {ev:.4f} | Div Time {dt_val:.3f}s | '
                      f'FTLE Error {ftle_err:.4f}')

                if use_wandb:
                    wandb.log({
                        f'{prefix}/test_mse': test_mse,
                        f'{prefix}/energy_violation': ev,
                        f'{prefix}/divergence_time': dt_val,
                        f'{prefix}/ftle_error': ftle_err,
                        f'{prefix}/step': itr,
                    })

                update_test_plots(model_name)
                update_traj_plot(model_name, viz_pred)


            if itr % 1000 == 0:
                ckpt_dir = os.path.join(model_dir, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                fname = model_name.lower().replace(' ', '_')
                torch.save(seq_model.state_dict(),
                           os.path.join(ckpt_dir, f'{fname}_iter{itr}.pt'))
                print(f'    [checkpoint] {model_name} saved at iter {itr}')

        train_times_seq[model_name] = time.time() - start

    # ==================================================================
    # Save models
    # ==================================================================
    for name, model in models.items():
        fname = name.lower().replace(' ', '_') + '.pt'
        torch.save(model.state_dict(), os.path.join(model_dir, fname))
    torch.save({'y_mean': y_mean.cpu(), 'y_std': y_std.cpu()},
               os.path.join(model_dir, 'norm_stats.pt'))
    print(f'\nModels saved to {model_dir}/')

    # ==================================================================
    # Final evaluation & save plots to disk
    # ==================================================================
    print('\n' + '=' * 60)
    print('FINAL RESULTS')
    print('=' * 60)

    # Print summary table
    print(f'\n{"":20s} ' + ' '.join(f'{n:>12s}' for n in names))
    print('-' * 72)
    print(f'{"Parameters":20s} ' + ' '.join(
        f'{count_parameters(models[n]):>12d}' for n in names))

    all_train_times = {'Neural ODE': train_time_node, **train_times_seq}
    print(f'{"Train time (s)":20s} ' + ' '.join(
        f'{all_train_times[n]:>12.1f}' for n in names))

    for nm in names:
        d = log[nm]
        if d['iters_slow']:
            print(f'  {nm:12s} final — '
                  f'Test MSE: {d["test_mse"][-1]:.6f}  '
                  f'Energy Viol: {d["energy_viol"][-1]:.4f}  '
                  f'Div Time: {d["div_time"][-1]:.3f}s  '
                  f'FTLE Error: {d["ftle_error"][-1]:.4f}')
    print('=' * 72)

    # Save monitoring figures
    fig_batch.savefig(f'{output_dir}/training_monitor.png', dpi=150)
    fig_test.savefig(f'{output_dir}/test_evaluation.png', dpi=150)
    print(f'Saved {output_dir}/training_monitor.png')
    print(f'Saved {output_dir}/test_evaluation.png')

    # Final trajectory plot with all models
    fig_final, axes_final = plt.subplots(n_viz, n_viz_cols,
                                         figsize=(4 * n_viz_cols, 3 * n_viz))
    t_np = t_viz.cpu().numpy()
    col_labels = list(state_labels) + (['Energy'] if has_energy else [])

    final_preds = {}
    with torch.no_grad():
        if train_node:
            final_preds['Neural ODE'] = sincos_to_angles(denormalize(
                odeint(node_func, viz_y0_norm, t_viz, **solver_kwargs)))
        for nm, sm in seq_models.items():
            final_preds[nm] = sincos_to_angles(denormalize(
                sm.predict_trajectory(viz_y0_norm, n_viz_points)))

    for row in range(n_viz):
        true_np = viz_true[:, row, :].cpu().numpy()

        for col in range(state_dim):
            ax = axes_final[row, col]
            ax.plot(t_np, true_np[:, col], 'k-', lw=1.5, label='True')
            for nm in names:
                pred_np = final_preds[nm][:, row, :].cpu().numpy()
                ax.plot(t_np, pred_np[:, col], '--', color=colors[nm],
                        lw=1, label=nm, alpha=0.8)
            if row == 0:
                ax.set_title(col_labels[col])
            if row == n_viz - 1:
                ax.set_xlabel('t (s)')
            if col == 0:
                ax.set_ylabel(f'Traj {row + 1}')
            ax.legend(fontsize=5)

        if has_energy:
            ax_e = axes_final[row, state_dim]
            E_true = energy_fn(viz_true[:, row:row + 1, :], args)[:, 0].cpu().numpy()
            ax_e.plot(t_np, E_true, 'k-', lw=1.5, label='True')
            for nm in names:
                E_p = energy_fn(final_preds[nm][:, row:row + 1, :], args)[:, 0].cpu().numpy()
                ax_e.plot(t_np, E_p, '--', color=colors[nm], lw=1,
                          label=nm, alpha=0.8)
            if row == 0:
                ax_e.set_title('Energy')
            if row == n_viz - 1:
                ax_e.set_xlabel('t (s)')
            ax_e.legend(fontsize=5)

    fig_final.suptitle(f'All Models — Fixed Trajectories ({viz_time}s)')
    fig_final.tight_layout()
    fig_final.savefig(f'{output_dir}/trajectory_final.png', dpi=150)
    print(f'Saved {output_dir}/trajectory_final.png')

    if use_wandb:
        wandb.log({'final/trajectory_plot': wandb.Image(fig_final)})
        wandb.finish()
