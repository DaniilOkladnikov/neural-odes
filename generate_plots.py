"""Unified plot generation: wandb metrics + OOD checkpoint evaluation."""

import os
import numpy as np
import torch
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from evaluate import (
    make_models, angles_to_sincos, sincos_to_angles,
    normalize, denormalize, sample_in_energy_range,
    energy_fn, args as eval_args, V_max, device, dynamics,
    model_dir, y_mean, y_std, solver_kwargs,
    plot_trajectory_comparison,
)

OUT_DIR = 'png_double_pendulum'
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {'Neural ODE': 'C0', 'RNN': 'C1', 'GRU': 'C2', 'LSTM': 'C3'}
MODEL_NAMES = ['Neural ODE', 'RNN', 'GRU', 'LSTM']
FNAME_MAP = {'Neural ODE': 'neural_ode', 'RNN': 'rnn', 'GRU': 'gru', 'LSTM': 'lstm'}
SEQ_MODELS = [('rnn', 'RNN'), ('gru', 'GRU'), ('lstm', 'LSTM')]


def _style_ax(ax, title=None, ylabel=None, log_y=False, sci_y=False):
    if title:
        ax.set_title(title, fontsize=24)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel('Step', fontsize=18)
    if log_y:
        ax.set_yscale('log')
    if sci_y:
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    ax.tick_params(axis='both', labelsize=20)
    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.yaxis.get_offset_text().set_fontsize(20)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_wandb_data():
    api = wandb.Api()
    node_run = api.run("neural-odes-dp/zguy1g8j")
    rnn_run = api.run("neural-odes-dp/98qg5m7u")
    node_history = node_run.history(samples=10000)
    rnn_history = rnn_run.history(samples=10000)
    return node_history, rnn_history


def compute_ood_metrics(iters, n_samples=100):
    """Load each checkpoint and compute OOD MSE + energy violation."""
    E_lo = 1.5 * V_max
    E_hi = 2.0 * V_max
    print(f'Sampling {n_samples} OOD ICs with E in [{E_lo:.2f}, {E_hi:.2f}]...')
    ood_y0 = sample_in_energy_range(E_lo, E_hi, n_samples).to(device)
    print(f'  Got {ood_y0.shape[0]} ICs')

    traj_time, dt = 1.0, 0.05
    n_points = int(round(traj_time / dt)) + 1
    t_eval = torch.linspace(0., traj_time, n_points).to(device)

    print('Integrating OOD ground truth...')
    with torch.no_grad():
        true_traj = odeint(dynamics, ood_y0, t_eval, **solver_kwargs)

    y0_sc = angles_to_sincos(ood_y0)
    y0_norm = normalize(y0_sc)
    true_traj_sc = angles_to_sincos(true_traj)
    true_traj_norm = (true_traj_sc - y_mean) / y_std

    results = {name: {'mse': [], 'energy_viol': []} for name in MODEL_NAMES}

    for name in MODEL_NAMES:
        print(f'\nEvaluating {name} checkpoints...')
        model = make_models()[name]

        for it in iters:
            ckpt = os.path.join(
                model_dir, 'checkpoints', f'{FNAME_MAP[name]}_iter{it}.pt')
            if not os.path.exists(ckpt):
                results[name]['mse'].append(float('nan'))
                results[name]['energy_viol'].append(float('nan'))
                continue

            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            with torch.no_grad():
                if name == 'Neural ODE':
                    pred_norm = odeint(model, y0_norm, t_eval, **solver_kwargs)
                else:
                    pred_norm = model.predict_trajectory(y0_norm, n_points)

                mse = torch.mean((pred_norm - true_traj_norm) ** 2).item()

                pred_4d = sincos_to_angles(denormalize(pred_norm))
                E_pred = energy_fn(pred_4d, eval_args)
                E0 = energy_fn(ood_y0.unsqueeze(0), eval_args)
                ev = torch.mean(torch.abs(E_pred - E0)).item()

            results[name]['mse'].append(mse)
            results[name]['energy_viol'].append(ev)

            if it % 10000 == 0:
                print(f'  iter {it:6d} | MSE {mse:.6f} | EV {ev:.4f}')

    return results


# ---------------------------------------------------------------------------
# Plot 1: Neural ODE summary (4 subplots)
# ---------------------------------------------------------------------------
def plot_neural_ode_figure(node_history, ood_results, iters_arr):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) Learning Rate
    df = node_history[['neural_ode/step', 'neural_ode/lr']].dropna()
    axes[0, 0].plot(df['neural_ode/step'], df['neural_ode/lr'])
    _style_ax(axes[0, 0], title='Learning Rate', ylabel='LR', sci_y=True)

    # 2) Derivative MSE: batch, test (train dist.), OOD
    df_b = node_history[['neural_ode/step', 'neural_ode/batch_mse']].dropna()
    df_t = node_history[['neural_ode/step', 'neural_ode/test_mse']].dropna()
    axes[0, 1].plot(df_b['neural_ode/step'], df_b['neural_ode/batch_mse'],
                 label='Batch MSE', alpha=0.7)
    axes[0, 1].plot(df_t['neural_ode/step'], df_t['neural_ode/test_mse'],
                 label='Test MSE (train dist.)')
    axes[0, 1].plot(iters_arr, ood_results['Neural ODE']['mse'],
                 label='OOD MSE')
    axes[0, 1].legend(fontsize=20)
    _style_ax(axes[0, 1], title='Derivative MSE', ylabel='MSE', log_y=True)

    # 3) Energy: batch loss, test violation, OOD violation
    df_el = node_history[['neural_ode/step', 'neural_ode/energy_loss']].dropna()
    df_ev = node_history[['neural_ode/step', 'neural_ode/energy_violation']].dropna()
    print(f'  Batch energy loss: {len(df_el)} points, '
          f'range [{df_el["neural_ode/energy_loss"].min():.6f}, '
          f'{df_el["neural_ode/energy_loss"].max():.6f}]')
    axes[1, 0].plot(df_el['neural_ode/step'], df_el['neural_ode/energy_loss'],
                 label='Batch Energy Loss')
    axes[1, 0].plot(df_ev['neural_ode/step'], df_ev['neural_ode/energy_violation'],
                 label='Test Energy Viol.')
    axes[1, 0].plot(iters_arr, ood_results['Neural ODE']['energy_viol'],
                 label='OOD Energy Viol.')
    axes[1, 0].legend(fontsize=20)
    _style_ax(axes[1, 0], title='Energy Violation', ylabel='Energy Violation', log_y=True)

    # 4) Divergence Time
    df_d = node_history[['neural_ode/step', 'neural_ode/divergence_time']].dropna()
    axes[1, 1].plot(df_d['neural_ode/step'], df_d['neural_ode/divergence_time'])
    _style_ax(axes[1, 1], title='Divergence Time', ylabel='Time (s)')

    # Print last divergence time values
    if len(df_d) > 0:
        last_step = df_d['neural_ode/step'].iloc[-1]
        last_div_time = df_d['neural_ode/divergence_time'].iloc[-1]
        print(f'  Last divergence time: {last_div_time:.4f} s (at step {last_step:.0f})')

    plt.tight_layout(h_pad=4.0)
    path = os.path.join(OUT_DIR, 'neural_ode_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Comparison — train distribution (from wandb)
# ---------------------------------------------------------------------------
def plot_comparison_train(node_history, rnn_history):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Print last divergence time for all models
    print('\nLast divergence time values:')
    df_dt = node_history[['neural_ode/step', 'neural_ode/divergence_time']].dropna()
    if len(df_dt) > 0:
        print(f'  Neural ODE: {df_dt["neural_ode/divergence_time"].iloc[-1]:.4f} s '
              f'(step {df_dt["neural_ode/step"].iloc[-1]:.0f})')
    for prefix, display in SEQ_MODELS:
        df_dt = rnn_history[[f'{prefix}/step', f'{prefix}/divergence_time']].dropna()
        if len(df_dt) > 0:
            print(f'  {display}: {df_dt[f"{prefix}/divergence_time"].iloc[-1]:.4f} s '
                  f'(step {df_dt[f"{prefix}/step"].iloc[-1]:.0f})')

    # Print last test energy violation for all models
    print('\nLast test energy violation values:')
    df_ev = node_history[['neural_ode/step', 'neural_ode/energy_violation']].dropna()
    if len(df_ev) > 0:
        print(f'  Neural ODE: {df_ev["neural_ode/energy_violation"].iloc[-1]:.6f} '
              f'(step {df_ev["neural_ode/step"].iloc[-1]:.0f})')
    for prefix, display in SEQ_MODELS:
        df_ev = rnn_history[[f'{prefix}/step', f'{prefix}/energy_violation']].dropna()
        if len(df_ev) > 0:
            print(f'  {display}: {df_ev[f"{prefix}/energy_violation"].iloc[-1]:.6f} '
                  f'(step {df_ev[f"{prefix}/step"].iloc[-1]:.0f})')

    for ax, (metric, title, log_y) in zip(axes, [
        ('energy_violation', 'Energy Violation / Step — All Models', True),
        ('divergence_time', 'Divergence Time / Step — All Models', False),
    ]):
        # Neural ODE
        df = node_history[['neural_ode/step', f'neural_ode/{metric}']].dropna()
        ax.plot(df['neural_ode/step'], df[f'neural_ode/{metric}'],
                label='Neural ODE', color=COLORS['Neural ODE'])
        # Sequence models
        for prefix, display in SEQ_MODELS:
            sk, mk = f'{prefix}/step', f'{prefix}/{metric}'
            df = rnn_history[[sk, mk]].dropna()
            ax.plot(df[sk], df[mk], label=display, color=COLORS[display])
        ax.legend(fontsize=20)
        _style_ax(ax, title=title, log_y=log_y)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'comparison_train.png')
    plt.savefig(path, dpi=150)
    print(f'Saved {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Comparison — OOD (from checkpoints)
# ---------------------------------------------------------------------------
def plot_comparison_ood(ood_results, iters_arr):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name in MODEL_NAMES:
        axes[0].plot(iters_arr, ood_results[name]['energy_viol'],
                     label=name, color=COLORS[name])
        axes[1].plot(iters_arr, ood_results[name]['mse'],
                     label=name, color=COLORS[name])

    axes[0].legend(fontsize=20)
    _style_ax(axes[0], title='OOD Energy Violation / Step — All Models',
              ylabel='Energy Violation', log_y=True)

    axes[1].legend(fontsize=20)
    _style_ax(axes[1], title='OOD Test MSE / Step — All Models',
              ylabel='MSE', log_y=True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'comparison_ood.png')
    plt.savefig(path, dpi=150)
    print(f'Saved {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('Fetching wandb data...')
    node_history, rnn_history = fetch_wandb_data()

    iters = list(range(1000, 101000, 1000))
    ood_results = compute_ood_metrics(iters)
    iters_arr = np.array(iters)

    plot_neural_ode_figure(node_history, ood_results, iters_arr)
    plot_comparison_train(node_history, rnn_history)
    plot_comparison_ood(ood_results, iters_arr)
    plot_trajectory_comparison()

    print('\nDone! All plots saved.')


if __name__ == '__main__':
    main()
