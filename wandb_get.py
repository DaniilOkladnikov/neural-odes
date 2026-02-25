import wandb
import matplotlib.pyplot as plt
import os

api = wandb.Api()

# Fetch data from both runs
node_run = api.run("neural-odes-dp/zguy1g8j")
rnn_run = api.run("neural-odes-dp/98qg5m7u")

node_history = node_run.history(samples=10000)
rnn_history = rnn_run.history(samples=10000)

out_dir = "png_double_pendulum"
os.makedirs(out_dir, exist_ok=True)


def plot_single(df, step_key, metric_key, title, filename,
                log_scale=False, sci_notation=False, label=None):
    """Save a single metric plot to its own file."""
    data = df[[step_key, metric_key]].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data[step_key], data[metric_key], label=label)
    ax.set_title(f"{title} / Step", fontsize="xx-large")
    if log_scale:
        ax.set_yscale("log")
    if sci_notation:
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    ax.tick_params(axis="both", labelsize="x-large")
    ax.xaxis.get_offset_text().set_fontsize("x-large")
    ax.yaxis.get_offset_text().set_fontsize("x-large")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend(fontsize="large")
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ---- Neural ODE individual plots ----
node_metrics = [
    ("neural_ode/lr", "Learning Rate", "neural_ode_lr.png", False, True),
    ("neural_ode/test_mse", "Derivative Test MSE", "neural_ode_test_mse.png", True, False),
    ("neural_ode/batch_mse", "Derivative Batch MSE", "neural_ode_batch_mse.png", True, False),
    ("neural_ode/energy_violation", "Energy Violation", "neural_ode_energy_violation.png", True, False),
    ("neural_ode/divergence_time", "Divergence Time", "neural_ode_divergence_time.png", False, False),
]

for key, title, fname, log_scale, sci in node_metrics:
    plot_single(node_history, "neural_ode/step", key,
                f"Neural ODE — {title}", fname, log_scale, sci)

# ---- RNN, GRU, LSTM individual plots ----
seq_models = [
    ("rnn", "RNN"),
    ("gru", "GRU"),
    ("lstm", "LSTM"),
]

seq_metrics = [
    ("lr", "Learning Rate", False, True),
    ("test_mse", "Trajectory Test MSE", True, False),
    ("batch_mse", "Trajectory Batch MSE", True, False),
    ("energy_violation", "Energy Violation", True, False),
    ("divergence_time", "Divergence Time", False, False),
]

for prefix, display_name in seq_models:
    for metric_suffix, title, log_scale, sci in seq_metrics:
        key = f"{prefix}/{metric_suffix}"
        step_key = f"{prefix}/step"
        fname = f"{prefix}_{metric_suffix}.png"
        plot_single(rnn_history, step_key, key,
                    f"{display_name} — {title}", fname, log_scale, sci)

# ---- Comparison plots: Energy Violation across all models ----
fig, ax = plt.subplots(figsize=(10, 6))
colors = {"Neural ODE": "C0", "RNN": "C1", "GRU": "C2", "LSTM": "C3"}

# Neural ODE
df = node_history[["neural_ode/step", "neural_ode/energy_violation"]].dropna()
ax.plot(df["neural_ode/step"], df["neural_ode/energy_violation"],
        label="Neural ODE", color=colors["Neural ODE"])

for prefix, display_name in seq_models:
    step_key = f"{prefix}/step"
    metric_key = f"{prefix}/energy_violation"
    df = rnn_history[[step_key, metric_key]].dropna()
    ax.plot(df[step_key], df[metric_key],
            label=display_name, color=colors[display_name])

ax.set_title("Energy Violation / Step — All Models", fontsize="xx-large")
ax.set_xlabel("Step", fontsize="x-large")
ax.set_ylabel("Energy Violation", fontsize="x-large")
ax.set_yscale("log")
ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
ax.tick_params(axis="both", labelsize="x-large")
ax.xaxis.get_offset_text().set_fontsize("x-large")
ax.yaxis.get_offset_text().set_fontsize("x-large")
ax.legend(fontsize="large")
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(out_dir, "comparison_energy_violation.png")
plt.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close(fig)

# ---- Comparison plots: Divergence Time across all models ----
fig, ax = plt.subplots(figsize=(10, 6))

df = node_history[["neural_ode/step", "neural_ode/divergence_time"]].dropna()
ax.plot(df["neural_ode/step"], df["neural_ode/divergence_time"],
        label="Neural ODE", color=colors["Neural ODE"])

for prefix, display_name in seq_models:
    step_key = f"{prefix}/step"
    metric_key = f"{prefix}/divergence_time"
    df = rnn_history[[step_key, metric_key]].dropna()
    ax.plot(df[step_key], df[metric_key],
            label=display_name, color=colors[display_name])

ax.set_title("Divergence Time / Step — All Models", fontsize="xx-large")
ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
ax.tick_params(axis="both", labelsize="x-large")
ax.xaxis.get_offset_text().set_fontsize("x-large")
ax.yaxis.get_offset_text().set_fontsize("x-large")
ax.legend(fontsize="large")
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(out_dir, "comparison_divergence_time.png")
plt.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close(fig)

# ---- Comparison plots: Trajectory Test MSE across RNN models ----
fig, ax = plt.subplots(figsize=(10, 6))

for prefix, display_name in seq_models:
    step_key = f"{prefix}/step"
    metric_key = f"{prefix}/test_mse"
    df = rnn_history[[step_key, metric_key]].dropna()
    ax.plot(df[step_key], df[metric_key],
            label=display_name, color=colors[display_name])

ax.set_title("Trajectory Test MSE / Step — RNN Models", fontsize="xx-large")
ax.set_yscale("log")
ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
ax.tick_params(axis="both", labelsize="x-large")
ax.xaxis.get_offset_text().set_fontsize("x-large")
ax.yaxis.get_offset_text().set_fontsize("x-large")
ax.legend(fontsize="large")
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(out_dir, "comparison_test_mse.png")
plt.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close(fig)

# ---- Comparison plots: Trajectory Batch MSE across RNN models ----
fig, ax = plt.subplots(figsize=(10, 6))

for prefix, display_name in seq_models:
    step_key = f"{prefix}/step"
    metric_key = f"{prefix}/batch_mse"
    df = rnn_history[[step_key, metric_key]].dropna()
    ax.plot(df[step_key], df[metric_key],
            label=display_name, color=colors[display_name])

ax.set_title("Trajectory Batch MSE / Step — RNN Models", fontsize="xx-large")
ax.set_yscale("log")
ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
ax.tick_params(axis="both", labelsize="x-large")
ax.xaxis.get_offset_text().set_fontsize("x-large")
ax.yaxis.get_offset_text().set_fontsize("x-large")
ax.legend(fontsize="large")
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(out_dir, "comparison_batch_mse.png")
plt.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close(fig)

print("\nDone! All plots saved.")
