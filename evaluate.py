"""Evaluation script: trajectory comparison & OOD generalization analysis."""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from dynamics import get_system
from models import ODEFunc, SequencePredictor, compute_default_hidden_sizes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

system = get_system('double_pendulum')
state_dim = system['state_dim']          # 4
angle_indices = system['angle_indices']  # [0, 1]
repr_dim = state_dim + len(angle_indices)  # 6
state_labels = system['state_labels']
energy_fn = system['energy_fn']
sample_ics_fn = system['sample_ics_fn']

# Physical parameters (defaults)
m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.81

# Fake args for energy_fn
class Args:
    pass
args = Args()
args.m1, args.m2, args.l1, args.l2, args.g = m1, m2, l1, l2, g

# Dynamics for ground truth integration
from dynamics import DoublePendulumDynamics
dynamics = DoublePendulumDynamics(m1, m2, l1, l2, g)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dynamics = dynamics.to(device)

model_dir = 'models_double_pendulum'
out_dir = 'png_double_pendulum'
os.makedirs(out_dir, exist_ok=True)

# V_max = max potential energy (both arms up, no velocity)
V_max = (m1 + m2) * g * l1 + m2 * g * l2
E_max_train = 1.5 * V_max  # training E_max

# Architecture sizes (inferred from saved checkpoints)
num_layers_node = 3
H_node = 220
num_layers_seq = 3
H_rnn = 140
H_gru = 80
H_lstm = 69

# Load normalization stats
norm = torch.load(os.path.join(model_dir, 'norm_stats.pt'), map_location=device)
y_mean = norm['y_mean'].to(device)  # (1, 1, repr_dim)
y_std = norm['y_std'].to(device)

solver_kwargs = dict(method='dopri5')


# ---------------------------------------------------------------------------
# Representation helpers (from train.py)
# ---------------------------------------------------------------------------
def angles_to_sincos(y):
    pieces = []
    for i in range(state_dim):
        if i in angle_indices:
            pieces.append(torch.sin(y[..., i:i+1]))
            pieces.append(torch.cos(y[..., i:i+1]))
        else:
            pieces.append(y[..., i:i+1])
    return torch.cat(pieces, dim=-1)


def sincos_to_angles(y):
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


def normalize(y_sincos):
    return (y_sincos - y_mean.squeeze(0)) / y_std.squeeze(0)


def denormalize(y_norm):
    return y_norm * y_std + y_mean


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
def make_models():
    """Instantiate empty models with correct architecture."""
    return {
        'Neural ODE': ODEFunc(repr_dim, H_node, num_layers_node).to(device),
        'RNN': SequencePredictor(repr_dim, H_rnn, 'rnn', num_layers_seq).to(device),
        'GRU': SequencePredictor(repr_dim, H_gru, 'gru', num_layers_seq).to(device),
        'LSTM': SequencePredictor(repr_dim, H_lstm, 'lstm', num_layers_seq).to(device),
    }


def load_final_models():
    """Load final trained models."""
    models = make_models()
    paths = {
        'Neural ODE': os.path.join(model_dir, 'checkpoints', 'neural_ode_iter100000.pt'),
        'RNN': os.path.join(model_dir, 'rnn.pt'),
        'GRU': os.path.join(model_dir, 'gru.pt'),
        'LSTM': os.path.join(model_dir, 'lstm.pt'),
    }
    for name, path in paths.items():
        models[name].load_state_dict(torch.load(path, map_location=device))
        models[name].eval()
    return models


def predict(model, name, y0_norm, t_eval, n_steps):
    """Predict trajectory → return physical 4D states."""
    with torch.no_grad():
        if name == 'Neural ODE':
            pred_norm = odeint(model, y0_norm, t_eval, **solver_kwargs)
        else:
            pred_norm = model.predict_trajectory(y0_norm, n_steps)
    return sincos_to_angles(denormalize(pred_norm))


# ---------------------------------------------------------------------------
# Sample ICs at specific energy levels
# ---------------------------------------------------------------------------
def sample_at_energy(E_target, n_samples=1, tol=0.05):
    """Sample ICs with energy close to E_target (within tol fraction)."""
    E_lo = E_target * (1 - tol)
    E_hi = E_target * (1 + tol)
    results = []
    while len(results) < n_samples:
        batch = sample_ics_fn(5000, E_hi, m1, m2, l1, l2, g)
        E = energy_fn(batch, args)
        mask = (E >= E_lo) & (E <= E_hi)
        results.append(batch[mask])
        if sum(r.shape[0] for r in results) >= n_samples:
            break
    return torch.cat(results, dim=0)[:n_samples]


def sample_in_energy_range(E_lo, E_hi, n_samples):
    """Sample ICs with energy uniformly in [E_lo, E_hi]."""
    results = []
    while len(results) < 1 or sum(r.shape[0] for r in results) < n_samples:
        batch = sample_ics_fn(10000, E_hi, m1, m2, l1, l2, g)
        E = energy_fn(batch, args)
        mask = (E >= E_lo) & (E <= E_hi)
        if mask.sum() > 0:
            results.append(batch[mask])
    return torch.cat(results, dim=0)[:n_samples]


# ---------------------------------------------------------------------------
# Part 1: Trajectory comparison (3 energy levels, all models)
# ---------------------------------------------------------------------------
def plot_trajectory_comparison():
    print('=== Part 1: Trajectory Comparison ===')
    models = load_final_models()

    # 3 energy levels
    E_levels = [0.2 * V_max, 0.7 * V_max, 1.3 * V_max]
    level_names = ['Low', 'Mid', 'High']

    viz_time = 5.0
    dt = 0.05
    n_points = int(round(viz_time / dt)) + 1
    t_eval = torch.linspace(0., viz_time, n_points).to(device)

    colors = {'Neural ODE': 'C0', 'RNN': 'C1', 'GRU': 'C2', 'LSTM': 'C3'}
    model_names = ['Neural ODE', 'RNN', 'GRU', 'LSTM']
    n_cols = state_dim + 1  # 4 state vars + energy

    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 3.5 * 3))
    col_labels = list(state_labels) + ['Energy']

    for row, (E_target, lname) in enumerate(zip(E_levels, level_names)):
        print(f'  Sampling IC at E ~ {E_target:.2f} ({lname})...')
        y0_phys = sample_at_energy(E_target, 1).to(device)  # (1, 4)
        E_actual = energy_fn(y0_phys, args).item()
        print(f'    E_actual = {E_actual:.4f}')

        # Ground truth
        with torch.no_grad():
            true_traj = odeint(dynamics, y0_phys, t_eval, **solver_kwargs)  # (T, 1, 4)

        # Normalize IC for model input
        y0_sc = angles_to_sincos(y0_phys)  # (1, 6)
        y0_norm = normalize(y0_sc)

        t_np = t_eval.cpu().numpy()
        true_np = true_traj[:, 0, :].cpu().numpy()  # (T, 4)
        # Wrap angles to [-pi, pi]
        for ai in angle_indices:
            true_np[:, ai] = (true_np[:, ai] + np.pi) % (2 * np.pi) - np.pi

        # State variables
        for col in range(state_dim):
            ax = axes[row, col]
            ax.plot(t_np, true_np[:, col], 'k-', lw=1.5, label='True')
            for name in model_names:
                pred = predict(models[name], name, y0_norm, t_eval, n_points)
                pred_np = pred[:, 0, :].cpu().numpy()
                for ai in angle_indices:
                    pred_np[:, ai] = (pred_np[:, ai] + np.pi) % (2 * np.pi) - np.pi
                ax.plot(t_np, pred_np[:, col], '--', color=colors[name],
                        lw=1.2, label=name, alpha=0.8)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=24)
            if row == 2:
                ax.set_xlabel('t (s)', fontsize=18)
            if col == 0:
                ax.set_ylabel(f'{lname} E', fontsize=18)
            ax.tick_params(axis='both', labelsize=20)
            ax.grid(True, alpha=0.3)

        # Energy column
        ax_e = axes[row, state_dim]
        E_true = energy_fn(true_traj[:, 0:1, :], args)[:, 0].cpu().numpy()
        ax_e.plot(t_np, E_true, 'k-', lw=1.5, label='True')
        for name in model_names:
            pred = predict(models[name], name, y0_norm, t_eval, n_points)
            E_pred = energy_fn(pred[:, 0:1, :], args)[:, 0].cpu().numpy()
            ax_e.plot(t_np, E_pred, '--', color=colors[name],
                      lw=1.2, label=name, alpha=0.8)
        if row == 0:
            ax_e.set_title('Energy', fontsize=24)
        if row == 2:
            ax_e.set_xlabel('t (s)', fontsize=18)
        ax_e.tick_params(axis='both', labelsize=20)
        ax_e.grid(True, alpha=0.3)

    # Single shared legend at the top of the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=20,
               bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    path = os.path.join(out_dir, 'trajectory_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 2: OOD generalization across checkpoints
# ---------------------------------------------------------------------------
def plot_ood_generalization():
    print('\n=== Part 2: OOD Generalization ===')

    # Sample 100 OOD ICs with E in [1.5 * V_max, 2.0 * V_max]
    E_lo_ood = 1.5 * V_max
    E_hi_ood = 2.0 * V_max
    print(f'Sampling 100 OOD ICs with E in [{E_lo_ood:.2f}, {E_hi_ood:.2f}]...')
    ood_y0_phys = sample_in_energy_range(E_lo_ood, E_hi_ood, 100).to(device)
    print(f'  Got {ood_y0_phys.shape[0]} ICs')

    # Ground truth trajectories (1s horizon)
    traj_time = 1.0
    dt = 0.05
    n_points = int(round(traj_time / dt)) + 1
    t_eval = torch.linspace(0., traj_time, n_points).to(device)

    print('Integrating OOD ground truth...')
    with torch.no_grad():
        true_traj = odeint(dynamics, ood_y0_phys, t_eval, **solver_kwargs)  # (T, 100, 4)

    # Prepare normalized inputs
    y0_sc = angles_to_sincos(ood_y0_phys)
    y0_norm = normalize(y0_sc)
    true_traj_sc = angles_to_sincos(true_traj)
    true_traj_norm = (true_traj_sc - y_mean) / y_std

    # Checkpoint iterations
    iters = list(range(1000, 101000, 1000))
    model_names = ['Neural ODE', 'RNN', 'GRU', 'LSTM']
    colors = {'Neural ODE': 'C0', 'RNN': 'C1', 'GRU': 'C2', 'LSTM': 'C3'}

    fname_map = {
        'Neural ODE': 'neural_ode',
        'RNN': 'rnn',
        'GRU': 'gru',
        'LSTM': 'lstm',
    }

    results = {name: {'mse': [], 'energy_viol': []} for name in model_names}

    for name in model_names:
        print(f'\nEvaluating {name} checkpoints...')
        # Create model once, reload weights each time
        models_dict = make_models()
        model = models_dict[name]

        for it in iters:
            ckpt_path = os.path.join(
                model_dir, 'checkpoints', f'{fname_map[name]}_iter{it}.pt')
            if not os.path.exists(ckpt_path):
                results[name]['mse'].append(float('nan'))
                results[name]['energy_viol'].append(float('nan'))
                continue

            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            with torch.no_grad():
                if name == 'Neural ODE':
                    pred_norm = odeint(model, y0_norm, t_eval, **solver_kwargs)
                else:
                    pred_norm = model.predict_trajectory(y0_norm, n_points)

                # MSE in normalized space
                mse = torch.mean((pred_norm - true_traj_norm) ** 2).item()

                # Energy violation in physical space
                pred_4d = sincos_to_angles(denormalize(pred_norm))
                E_pred = energy_fn(pred_4d, args)          # (T, N)
                E0 = energy_fn(ood_y0_phys.unsqueeze(0), args)  # (1, N)
                ev = torch.mean(torch.abs(E_pred - E0)).item()

            results[name]['mse'].append(mse)
            results[name]['energy_viol'].append(ev)

            if it % 10000 == 0:
                print(f'  iter {it:6d} | MSE {mse:.6f} | EV {ev:.4f}')

    iters_arr = np.array(iters)

    # --- OOD Test MSE plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in model_names:
        ax.plot(iters_arr, results[name]['mse'],
                label=name, color=colors[name])
    ax.set_title('OOD Test MSE / Step — All Models', fontsize='xx-large')
    ax.set_xlabel('Step', fontsize='x-large')
    ax.set_ylabel('MSE', fontsize='x-large')
    ax.set_yscale('log')
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    ax.tick_params(axis='both', labelsize='x-large')
    ax.xaxis.get_offset_text().set_fontsize('x-large')
    ax.yaxis.get_offset_text().set_fontsize('x-large')
    ax.legend(fontsize='large')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'ood_test_mse.png')
    plt.savefig(path, dpi=150)
    print(f'\nSaved {path}')
    plt.close(fig)

    # --- OOD Energy Violation plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in model_names:
        ax.plot(iters_arr, results[name]['energy_viol'],
                label=name, color=colors[name])
    ax.set_title('OOD Energy Violation / Step — All Models', fontsize='xx-large')
    ax.set_xlabel('Step', fontsize='x-large')
    ax.set_ylabel('Energy Violation', fontsize='x-large')
    ax.set_yscale('log')
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    ax.tick_params(axis='both', labelsize='x-large')
    ax.xaxis.get_offset_text().set_fontsize('x-large')
    ax.yaxis.get_offset_text().set_fontsize('x-large')
    ax.legend(fontsize='large')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'ood_energy_violation.png')
    plt.savefig(path, dpi=150)
    print(f'\nSaved {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    plot_trajectory_comparison()
    plot_ood_generalization()
    print('\nDone!')
