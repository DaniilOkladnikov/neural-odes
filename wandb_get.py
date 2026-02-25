import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
run = api.run("neural-odes-dp/7zjycquo")

history = run.history(samples=10000)

metrics = [
    ("neural_ode/lr", "Learning Rate", False, True),
    ("neural_ode/test_mse", "Test MSE", True, False),
    ("neural_ode/energy_violation", "Energy Violation", True, False),
    ("neural_ode/divergence_time", "Divergence Time", False, False),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (key, title, log_scale, sci_notation) in zip(axes.flat, metrics):
    df = history[["neural_ode/step", key]].dropna()
    ax.plot(df["neural_ode/step"], df[key])
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

plt.tight_layout(h_pad=4)
plt.savefig("png_double_pendulum/wandb_metrics.png", dpi=150)
print("Saved to png_double_pendulum/wandb_metrics.png")
plt.show()
