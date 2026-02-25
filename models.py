import math

import torch.nn as nn


class ODEFunc(nn.Module):

    def __init__(self, state_dim, hidden_dim=150, num_layers=5):
        super().__init__()
        self.nfe = 0
        layers = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        self.nfe += 1
        return self.net(y)


class SequencePredictor(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type='rnn', num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        RNNClass = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[rnn_type]
        self.rnn = RNNClass(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, h0=None):
        """Teacher-forcing forward with residual. x: (seq_len, batch, state_dim)."""
        rnn_out, h_n = self.rnn(x, h0)
        return x + self.fc(rnn_out), h_n

    def autoregressive_rollout(self, y0, n_steps):
        """Autoregressive rollout with gradients. y0: (batch, state_dim).
        Returns (n_steps, batch, state_dim) including y0."""
        import torch
        preds = [y0]
        current = y0.unsqueeze(0)  # (1, batch, dim)
        h = None
        for _ in range(n_steps - 1):
            rnn_out, h = self.rnn(current, h)
            next_state = current + self.fc(rnn_out)
            preds.append(next_state.squeeze(0))
            current = next_state
        return torch.stack(preds, dim=0)

    def predict_trajectory(self, y0, n_steps):
        """Autoregressive rollout with residual. y0: (batch, state_dim)."""
        import torch
        preds = [y0]
        current = y0.unsqueeze(0)
        h = None
        with torch.no_grad():
            for _ in range(n_steps - 1):
                rnn_out, h = self.rnn(current, h)
                next_state = current + self.fc(rnn_out)
                preds.append(next_state.squeeze(0))
                current = next_state
        return torch.stack(preds, dim=0)


class MLPPredictor(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=5):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, input_size))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x, h0=None):
        """Teacher-forcing forward with residual. x: (seq_len, batch, state_dim)."""
        return x + self.net(x), None

    def autoregressive_rollout(self, y0, n_steps):
        """Autoregressive rollout with gradients. y0: (batch, state_dim).
        Returns (n_steps, batch, state_dim) including y0."""
        import torch
        preds = [y0]
        current = y0
        for _ in range(n_steps - 1):
            current = current + self.net(current)
            preds.append(current)
        return torch.stack(preds, dim=0)

    def predict_trajectory(self, y0, n_steps):
        """Autoregressive rollout without gradients. y0: (batch, state_dim)."""
        import torch
        preds = [y0]
        current = y0
        with torch.no_grad():
            for _ in range(n_steps - 1):
                current = current + self.net(current)
                preds.append(current)
        return torch.stack(preds, dim=0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def compute_default_hidden_sizes(state_dim, num_layers=5, target_params=100000):
    """Compute hidden sizes for NODE/RNN/GRU/LSTM to hit ~target_params."""
    D = state_dim
    L = num_layers

    def _solve(a, b, c):
        disc = b * b - 4 * a * c
        if disc < 0:
            return max(1, int(target_params ** 0.5))
        return max(1, round((-b + math.sqrt(disc)) / (2 * a)))

    # NODE: (L-1)*H^2 + (L + 2*D)*H + D = target
    H_node = _solve(L - 1, L + 2 * D, D - target_params)

    # RNN: (2L-1)*H^2 + (2D + 2L)*H + D = target
    H_rnn = _solve(2 * L - 1, 2 * D + 2 * L, D - target_params)

    # GRU: 3*(2L-1)*H^2 + (6D + 6L - 2D)*H + D = target
    # Layer 0: 3*(D*H + H^2 + 2H), Layers 1..L-1: 3*(2*H^2 + 2H), fc: D*H + D
    # = 3*H^2 + 3*D*H + 6H + (L-1)*(6*H^2 + 6H) + D*H + D
    # = (6L-3)*H^2 + (4D + 6L)*H + D
    H_gru = _solve(6 * L - 3, 4 * D + 6 * L, D - target_params)

    # LSTM: 4*(2L-1)*H^2 + (5D + 8L)*H + D
    # Layer 0: 4*(D*H + H^2 + 2H), Layers 1..L-1: 4*(2*H^2 + 2H), fc: D*H + D
    # = 4*H^2 + 4*D*H + 8H + (L-1)*(8*H^2 + 8H) + D*H + D
    # = (8L-4)*H^2 + (5D + 8L)*H + D
    H_lstm = _solve(8 * L - 4, 5 * D + 8 * L, D - target_params)

    # MLP: same architecture as NODE (no RNN cell overhead)
    H_mlp = H_node

    return H_node, H_rnn, H_gru, H_lstm, H_mlp
