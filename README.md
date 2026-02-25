# Neural ODEs vs Recurrent Networks for Dynamical Systems

Comparing Neural ODEs against RNN, GRU, and LSTM baselines for learning chaotic dynamical systems (double pendulum).

## Project Structure

| File | Description |
|------|-------------|
| `train.py` | Main training script for all models |
| `models.py` | Model definitions (Neural ODE, RNN, GRU, LSTM, MLP) |
| `dynamics.py` | Ground-truth dynamical systems (double pendulum) |
| `evaluate.py` | Model evaluation and metrics |
| `generate_plots.py` | Visualization and plot generation |
| `wandb_get.py` | Fetch metrics from Weights & Biases |

## Setup

Requires [uv](https://docs.astral.sh/uv/) for dependency management.

## Training

### Neural ODE

```bash
uv run train.py \
  --system double_pendulum \
  --method dopri5 \
  --target_params 100000 \
  --num_layers 3 \
  --eta_max 3e-4 --eta_min 1e-5 \
  --deriv_weight 1 --traj_weight 0 \
  --models both \
  --niters 100000 \
  --n_trajectories 100000
```

### RNN / GRU / LSTM

```bash
uv run train.py \
  --system double_pendulum \
  --method dopri5 \
  --target_params 100000 \
  --num_layers 3 \
  --eta_max 3e-4 --eta_min 1e-5 \
  --deriv_weight 0 --traj_weight 1 \
  --models rnn \
  --niters 100000 \
  --n_trajectories 50000
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--system` | Dynamical system (`double_pendulum`) | `double_pendulum` |
| `--method` | ODE solver (`dopri5`, `euler`, `rk4`, `adams`) | `dopri5` |
| `--models` | Which models to train (`node`, `rnn`, `mlp`, `both`) | `both` |
| `--target_params` | Target parameter count for hidden size auto-computation | `100000` |
| `--num_layers` | Number of hidden layers | `5` |
| `--niters` | Training iterations | `20000` |
| `--deriv_weight` | Derivative matching loss weight (Neural ODE) | `0.1` |
| `--traj_weight` | Trajectory MSE loss weight | `1.0` |
| `--eta_max` / `--eta_min` | Learning rate range for cosine annealing | `3e-4` / `1e-5` |
| `--n_trajectories` | Number of training trajectories | `50000` |
| `--no_wandb` | Disable Weights & Biases logging | `false` |

## Output

- Model checkpoints: `models_<system>/checkpoints/`
- Final models: `models_<system>/`
- Plots: `png_<system>/`
