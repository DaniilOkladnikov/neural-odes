import os
import argparse
import time
import numpy as np

import torch
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dynamics import get_system, SYSTEMS
from models import (ODEFunc, SequencePredictor, RunningAverageMeter,
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
# ODE solver
parser.add_argument('--method', type=str,
                    choices=['dopri5', 'adams', 'euler', 'rk4'], default='dopri5')
parser.add_argument('--step_size', type=float, default=0.005,
                    help='Step size for fixed-step solvers (euler, rk4)')
parser.add_argument('--adjoint', action='store_true')
# Dataset
parser.add_argument('--n_trajectories', type=int, default=20000)
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
parser.add_argument('--niters', type=int, default=6000)
parser.add_argument('--lr', type=float, default=2e-4)
# Monitoring
parser.add_argument('--log_every', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=1000)
parser.add_argument('--div_threshold', type=float, default=0.5,
                    help='Divergence threshold in radians')
parser.add_argument('--no_wandb', action='store_true')
# Model sizes
H_node, H_rnn, H_gru, H_lstm = compute_default_hidden_sizes(repr_dim)
parser.add_argument('--node_hidden', type=int, default=H_node)
parser.add_argument('--rnn_hidden', type=int, default=H_rnn)
parser.add_argument('--gru_hidden', type=int, default=H_gru)
parser.add_argument('--lstm_hidden', type=int, default=H_lstm)
parser.add_argument('--num_layers', type=int, default=5)
# Output
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
# System-specific args
system['add_cli_args'](parser)

args = parser.parse_args()

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

n_time_points = int(round(args.traj_time / args.traj_dt)) + 1
t = torch.linspace(0., args.traj_time, n_time_points).to(device)

print(f'Sampling {args.n_trajectories} initial conditions '
      f'with E in [0, {args.E_max:.2f}] ...')
all_y0 = sample_ics_fn(
    args.n_trajectories, args.E_max,
    args.m1, args.m2, args.l1, args.l2, args.g
).to(device)

print(f'Integrating {args.n_trajectories} trajectories '
      f'({n_time_points} steps, dt={args.traj_dt}s) ...')
all_trajectories = []
for i in range(0, args.n_trajectories, args.gen_batch_size):
    batch_y0 = all_y0[i:i + args.gen_batch_size]
    with torch.no_grad():
        batch_traj = odeint(dynamics, batch_y0, t, **solver_kwargs)
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


def get_batch():
    idx = np.random.choice(n_train, args.batch_size, replace=False)
    batch_y0 = train_y0_norm[idx]
    batch_t = t
    batch_y = train_traj_norm[:, idx, :]
    batch_y0_phys = train_y0_phys[idx]
    return batch_y0, batch_t, batch_y, batch_y0_phys


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    makedirs(output_dir)
    makedirs(model_dir)

    # --- Build models (repr_dim inputs/outputs) ---
    node_func = ODEFunc(repr_dim, args.node_hidden, args.num_layers).to(device)
    rnn_model = SequencePredictor(repr_dim, args.rnn_hidden, 'rnn',
                                  args.num_layers).to(device)
    gru_model = SequencePredictor(repr_dim, args.gru_hidden, 'gru',
                                  args.num_layers).to(device)
    lstm_model = SequencePredictor(repr_dim, args.lstm_hidden, 'lstm',
                                   args.num_layers).to(device)

    models = {
        'Neural ODE': node_func,
        'RNN': rnn_model,
        'GRU': gru_model,
        'LSTM': lstm_model,
    }
    seq_models = {'RNN': rnn_model, 'GRU': gru_model, 'LSTM': lstm_model}
    names = ['Neural ODE', 'RNN', 'GRU', 'LSTM']
    colors = {'Neural ODE': 'blue', 'RNN': 'red', 'GRU': 'orange', 'LSTM': 'purple'}

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
    eval_traj_norm = test_traj_norm[:, eval_idx, :]  # (T, n_eval, repr_dim) — normalized 6D
    eval_y0_phys = test_y0_phys[eval_idx]            # (n_eval, state_dim) — 4D for energy
    eval_y0_norm = test_y0_norm[eval_idx]

    # Divergence time: 50 fixed samples, 10s simulation
    n_div = min(50, n_eval)
    div_idx = eval_idx[:n_div]
    div_y0_phys = test_y0_phys[div_idx]           # 4D for odeint with physical dynamics
    div_y0_norm = test_y0_norm[div_idx]
    t_long = torch.linspace(0., 10.0, int(10.0 / args.traj_dt) + 1).to(device)
    n_long = len(t_long)

    print(f'Pre-computing 10s ground truth for {n_div} divergence-time samples ...')
    with torch.no_grad():
        div_true = odeint(dynamics, div_y0_phys, t_long, **solver_kwargs)  # 4D
        div_true[..., 0] = torch.atan2(torch.sin(div_true[..., 0]), torch.cos(div_true[..., 0]))
        div_true[..., 1] = torch.atan2(torch.sin(div_true[..., 1]), torch.cos(div_true[..., 1]))

    # Trajectory viz: 1 fixed IC, 10s
    viz_y0_phys = test_y0_phys[0:1]               # 4D for odeint
    viz_y0_norm = test_y0_norm[0:1]
    with torch.no_grad():
        viz_true = odeint(dynamics, viz_y0_phys, t_long, **solver_kwargs)  # (T_long, 1, 4)
        viz_true[..., 0] = torch.atan2(torch.sin(viz_true[..., 0]), torch.cos(viz_true[..., 0]))
        viz_true[..., 1] = torch.atan2(torch.sin(viz_true[..., 1]), torch.cos(viz_true[..., 1]))

    # ------------------------------------------------------------------
    # wandb init
    # ------------------------------------------------------------------
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project='neural-odes-dp', config=vars(args))
        for prefix in ['neural_ode', 'rnn', 'gru', 'lstm']:
            wandb.define_metric(f'{prefix}/*', step_metric=f'{prefix}/step')
    elif not args.no_wandb:
        print('Warning: wandb not installed, logging disabled')

    wb_prefix = {
        'Neural ODE': 'neural_ode', 'RNN': 'rnn',
        'GRU': 'gru', 'LSTM': 'lstm',
    }

    # ------------------------------------------------------------------
    # Real-time plot setup
    # ------------------------------------------------------------------
    plt.ion()
    fig_batch, (ax_bmse, ax_nfe) = plt.subplots(1, 2, figsize=(12, 4))
    fig_test, (ax_tmse, ax_ev, ax_dt) = plt.subplots(1, 3, figsize=(16, 4))
    state_labels = system['state_labels']
    fig_traj, axes_traj = plt.subplots(2, 3, figsize=(16, 8))
    fig_traj.suptitle('Fixed Trajectory (10s)')

    log = {name: {
        'iters_fast': [], 'batch_mse': [], 'nfe': [],
        'iters_slow': [], 'test_mse': [], 'energy_viol': [], 'div_time': [],
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
        fig_batch.canvas.draw_idle()
        fig_batch.canvas.flush_events()

    def update_test_plots(model_name):
        d = log[model_name]
        for ax, key, title, yscale in [
            (ax_tmse, 'test_mse', 'Test MSE', 'log'),
            (ax_ev, 'energy_viol', 'Energy Violation', 'log'),
            (ax_dt, 'div_time', 'Mean Divergence Time (s)', 'linear'),
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
        fig_test.canvas.draw_idle()
        fig_test.canvas.flush_events()

    def update_traj_plot(model_name, viz_pred):
        """Update the fixed trajectory figure for the currently training model."""
        t_np = t_long.cpu().numpy()
        true_np = viz_true[:, 0, :].cpu().numpy()
        pred_np = viz_pred[:, 0, :].cpu().numpy()

        for ax in axes_traj.flat:
            ax.cla()

        for i in range(4):
            ax = axes_traj.flat[i]
            ax.plot(t_np, true_np[:, i], 'k-', lw=1.5, label='True')
            ax.plot(t_np, pred_np[:, i], '--', color=colors[model_name],
                    lw=1.2, label=model_name)
            ax.set_ylabel(state_labels[i])
            ax.legend(fontsize=7)
            if i >= 2:
                ax.set_xlabel('t (s)')

        if has_energy:
            ax_e = axes_traj.flat[4]
            E_true = energy_fn(viz_true, args)[:, 0].cpu().numpy()
            E_pred = energy_fn(viz_pred, args)[:, 0].cpu().numpy()
            ax_e.plot(t_np, E_true, 'k-', lw=1.5, label='True')
            ax_e.plot(t_np, E_pred, '--', color=colors[model_name],
                      lw=1.2, label=model_name)
            ax_e.set_ylabel('Energy')
            ax_e.set_xlabel('t (s)')
            ax_e.legend(fontsize=7)

        axes_traj.flat[5].axis('off')
        fig_traj.suptitle(f'Fixed Trajectory — {model_name}')
        fig_traj.tight_layout()
        fig_traj.canvas.draw_idle()
        fig_traj.canvas.flush_events()

    # ------------------------------------------------------------------
    # Helper: full evaluation on test set
    # ------------------------------------------------------------------
    def evaluate_node(func):
        """Evaluate Neural ODE on fixed eval sets. Returns test_mse, energy_viol, div_time, viz_pred_4d."""
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

            # Divergence time (10s) — convert to 4D for angle comparison
            div_pred = sincos_to_angles(denormalize(odeint(func, div_y0_norm, t_long, **solver_kwargs)))
            dt_val = compute_divergence_time(div_pred, div_true, t_long, args.div_threshold)

            # Trajectory viz (10s) — return 4D for plotting
            viz_pred_4d = sincos_to_angles(denormalize(odeint(func, viz_y0_norm, t_long, **solver_kwargs)))

        return test_mse, ev, dt_val, viz_pred_4d

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

            div_pred = sincos_to_angles(denormalize(model.predict_trajectory(div_y0_norm, n_long)))
            dt_val = compute_divergence_time(div_pred, div_true, t_long, args.div_threshold)

            viz_pred_4d = sincos_to_angles(denormalize(model.predict_trajectory(viz_y0_norm, n_long)))

        return test_mse, ev, dt_val, viz_pred_4d

    # ==================================================================
    # Train Neural ODE
    # ==================================================================
    print('\n=== Training Neural ODE ===')
    node_optimizer = optim.Adam(node_func.parameters(), lr=args.lr)
    start = time.time()

    for itr in range(1, args.niters + 1):
        node_optimizer.zero_grad()
        batch_y0, batch_t, batch_y, batch_y0_phys = get_batch()

        node_func.nfe = 0
        pred_y = odeint(node_func, batch_y0, batch_t, **solver_kwargs).to(device)
        nfe = node_func.nfe

        traj_loss = torch.mean(torch.abs(pred_y - batch_y))

        # Energy conservation loss
        pred_4d = sincos_to_angles(denormalize(pred_y))
        E_pred = energy_fn(pred_4d, args)                   # (T, batch)
        E0 = energy_fn(batch_y0_phys.unsqueeze(0), args)    # (1, batch)
        energy_loss = torch.mean(torch.abs(E_pred - E0))

        loss = traj_loss + 0.2 * energy_loss
        loss.backward()
        node_optimizer.step()

        if itr % args.log_every == 0:
            with torch.no_grad():
                batch_mse = torch.mean((pred_y - batch_y) ** 2).item()

            log['Neural ODE']['iters_fast'].append(itr)
            log['Neural ODE']['batch_mse'].append(batch_mse)
            log['Neural ODE']['nfe'].append(nfe)

            print(f'  Iter {itr:5d} | MSE {batch_mse:.6f} | '
                  f'E_loss {energy_loss.item():.4f} | NFE {nfe}')

            if use_wandb:
                wandb.log({
                    'neural_ode/batch_mse': batch_mse,
                    'neural_ode/energy_loss': energy_loss.item(),
                    'neural_ode/nfe': nfe,
                    'neural_ode/step': itr,
                })

            update_batch_plots('Neural ODE')
            plt.pause(0.001)

        if itr % args.eval_every == 0:
            test_mse, ev, dt_val, viz_pred = evaluate_node(node_func)

            log['Neural ODE']['iters_slow'].append(itr)
            log['Neural ODE']['test_mse'].append(test_mse)
            log['Neural ODE']['energy_viol'].append(ev)
            log['Neural ODE']['div_time'].append(dt_val)

            print(f'    [eval] Test MSE {test_mse:.6f} | '
                  f'Energy Viol {ev:.4f} | Div Time {dt_val:.3f}s')

            if use_wandb:
                wandb.log({
                    'neural_ode/test_mse': test_mse,
                    'neural_ode/energy_violation': ev,
                    'neural_ode/divergence_time': dt_val,
                    'neural_ode/step': itr,
                })

            update_test_plots('Neural ODE')
            update_traj_plot('Neural ODE', viz_pred)
            plt.pause(0.001)

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
        optimizer = optim.Adam(seq_model.parameters(), lr=args.lr)
        start = time.time()
        prefix = wb_prefix[model_name]

        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y, batch_y0_phys = get_batch()
            rnn_input = batch_y[:-1]
            rnn_target = batch_y[1:]
            rnn_pred, _ = seq_model(rnn_input)
            traj_loss = torch.mean(torch.abs(rnn_pred - rnn_target))

            # Energy conservation loss
            pred_4d = sincos_to_angles(denormalize(rnn_pred))
            E_pred = energy_fn(pred_4d, args)
            E0 = energy_fn(batch_y0_phys.unsqueeze(0), args)
            energy_loss = torch.mean(torch.abs(E_pred - E0))

            loss = traj_loss + 0.2 * energy_loss
            loss.backward()
            optimizer.step()

            if itr % args.log_every == 0:
                with torch.no_grad():
                    batch_mse = torch.mean((rnn_pred - rnn_target) ** 2).item()

                log[model_name]['iters_fast'].append(itr)
                log[model_name]['batch_mse'].append(batch_mse)

                print(f'  Iter {itr:5d} | MSE {batch_mse:.6f} | '
                      f'E_loss {energy_loss.item():.4f}')

                if use_wandb:
                    wandb.log({
                        f'{prefix}/batch_mse': batch_mse,
                        f'{prefix}/energy_loss': energy_loss.item(),
                        f'{prefix}/step': itr,
                    })

                update_batch_plots(model_name)
                plt.pause(0.001)

            if itr % args.eval_every == 0:
                test_mse, ev, dt_val, viz_pred = evaluate_seq(seq_model)

                log[model_name]['iters_slow'].append(itr)
                log[model_name]['test_mse'].append(test_mse)
                log[model_name]['energy_viol'].append(ev)
                log[model_name]['div_time'].append(dt_val)

                print(f'    [eval] Test MSE {test_mse:.6f} | '
                      f'Energy Viol {ev:.4f} | Div Time {dt_val:.3f}s')

                if use_wandb:
                    wandb.log({
                        f'{prefix}/test_mse': test_mse,
                        f'{prefix}/energy_violation': ev,
                        f'{prefix}/divergence_time': dt_val,
                        f'{prefix}/step': itr,
                    })

                update_test_plots(model_name)
                update_traj_plot(model_name, viz_pred)
                plt.pause(0.001)

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
                  f'Div Time: {d["div_time"][-1]:.3f}s')
    print('=' * 72)

    # Save monitoring figures
    fig_batch.savefig(f'{output_dir}/training_monitor.png', dpi=150)
    fig_test.savefig(f'{output_dir}/test_evaluation.png', dpi=150)
    print(f'Saved {output_dir}/training_monitor.png')
    print(f'Saved {output_dir}/test_evaluation.png')

    # Final trajectory plot with all models (local only)
    fig_final, axes_final = plt.subplots(2, 3, figsize=(16, 8))
    t_np = t_long.cpu().numpy()
    true_np = viz_true[:, 0, :].cpu().numpy()

    final_preds = {}
    with torch.no_grad():
        final_preds['Neural ODE'] = sincos_to_angles(denormalize(
            odeint(node_func, viz_y0_norm, t_long, **solver_kwargs)))
        for nm, sm in seq_models.items():
            final_preds[nm] = sincos_to_angles(denormalize(
                sm.predict_trajectory(viz_y0_norm, n_long)))

    for i in range(4):
        ax = axes_final.flat[i]
        ax.plot(t_np, true_np[:, i], 'k-', lw=1.5, label='True')
        for nm in names:
            pred_np = final_preds[nm][:, 0, :].cpu().numpy()
            ax.plot(t_np, pred_np[:, i], '--', color=colors[nm],
                    lw=1, label=nm, alpha=0.8)
        ax.set_ylabel(state_labels[i])
        ax.legend(fontsize=6)
        if i >= 2:
            ax.set_xlabel('t (s)')

    if has_energy:
        ax_e = axes_final.flat[4]
        E_true = energy_fn(viz_true, args)[:, 0].cpu().numpy()
        ax_e.plot(t_np, E_true, 'k-', lw=1.5, label='True')
        for nm in names:
            E_p = energy_fn(final_preds[nm], args)[:, 0].cpu().numpy()
            ax_e.plot(t_np, E_p, '--', color=colors[nm], lw=1,
                      label=nm, alpha=0.8)
        ax_e.set_ylabel('Energy')
        ax_e.set_xlabel('t (s)')
        ax_e.legend(fontsize=6)

    axes_final.flat[5].axis('off')
    fig_final.suptitle('All Models — Fixed Trajectory (10s)')
    fig_final.tight_layout()
    fig_final.savefig(f'{output_dir}/trajectory_final.png', dpi=150)
    print(f'Saved {output_dir}/trajectory_final.png')

    if use_wandb:
        wandb.finish()

    plt.ioff()
    plt.show()
