import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Double Pendulum (4D)
# ---------------------------------------------------------------------------
class DoublePendulumDynamics(nn.Module):

    def __init__(self, m1, m2, l1, l2, g):
        super().__init__()
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g

    def forward(self, t, y):
        th1 = y[..., 0]
        th2 = y[..., 1]
        w1 = y[..., 2]
        w2 = y[..., 3]

        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        delta = th2 - th1
        cos_d = torch.cos(delta)
        sin_d = torch.sin(delta)

        den = m1 + m2 * sin_d ** 2

        dw1 = (m2 * l1 * w1 ** 2 * sin_d * cos_d
               + m2 * g * torch.sin(th2) * cos_d
               + m2 * l2 * w2 ** 2 * sin_d
               - (m1 + m2) * g * torch.sin(th1)) / (l1 * den)

        dw2 = (-m2 * l2 * w2 ** 2 * sin_d * cos_d
               + (m1 + m2) * g * torch.sin(th1) * cos_d
               - (m1 + m2) * l1 * w1 ** 2 * sin_d
               - (m1 + m2) * g * torch.sin(th2)) / (l2 * den)

        return torch.stack([w1, w2, dw1, dw2], dim=-1)


def _add_double_pendulum_args(parser):
    parser.add_argument('--m1', type=float, default=1.0)
    parser.add_argument('--m2', type=float, default=1.0)
    parser.add_argument('--l1', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--g', type=float, default=9.81)


def _make_double_pendulum(args):
    return DoublePendulumDynamics(args.m1, args.m2, args.l1, args.l2, args.g)


def _double_pendulum_energy(y, args):
    th1 = y[..., 0]
    th2 = y[..., 1]
    w1 = y[..., 2]
    w2 = y[..., 3]
    delta = th2 - th1
    T = (0.5 * args.m1 * args.l1 ** 2 * w1 ** 2
         + 0.5 * args.m2 * (args.l1 ** 2 * w1 ** 2 + args.l2 ** 2 * w2 ** 2
                             + 2 * args.l1 * args.l2 * w1 * w2 * torch.cos(delta)))
    V = (-(args.m1 + args.m2) * args.g * args.l1 * torch.cos(th1)
         - args.m2 * args.g * args.l2 * torch.cos(th2))
    return T + V


def _compute_double_pendulum_E_max(m1, m2, l1, l2, g):
    """1.5x the max potential energy (both pendulums pointing up)."""
    V_max = (m1 + m2) * g * l1 + m2 * g * l2
    return 1.5 * V_max


def _sample_double_pendulum_ics(n_samples, E_max, m1, m2, l1, l2, g):
    """Sample initial conditions with prescribed energy via Cholesky decomposition.

    For each sample: pick E in [0, E_max], random angles, solve for velocities.
    Returns tensor of shape (n_samples, 4) = [theta1, theta2, omega1, omega2].
    """
    ics = []
    count = 0
    M11 = (m1 + m2) * l1 ** 2
    M22 = m2 * l2 ** 2
    L11 = math.sqrt(M11)

    while count < n_samples:
        batch = max(n_samples - count, 1000)

        E = torch.rand(batch) * E_max
        th1 = torch.rand(batch) * 2 * math.pi - math.pi
        th2 = torch.rand(batch) * 2 * math.pi - math.pi

        V = (-(m1 + m2) * g * l1 * torch.cos(th1)
             - m2 * g * l2 * torch.cos(th2))
        T_target = E - V

        valid = T_target > 0
        if valid.sum() == 0:
            continue

        th1 = th1[valid]
        th2 = th2[valid]
        T_target = T_target[valid]

        delta = th2 - th1
        cos_d = torch.cos(delta)
        M12 = m2 * l1 * l2 * cos_d

        L21 = M12 / L11
        L22_sq = M22 - L21 ** 2

        pd_valid = L22_sq > 1e-10
        if pd_valid.sum() == 0:
            continue

        th1 = th1[pd_valid]
        th2 = th2[pd_valid]
        T_target = T_target[pd_valid]
        L21 = L21[pd_valid]
        L22 = torch.sqrt(L22_sq[pd_valid])

        phi = torch.rand(th1.shape[0]) * 2 * math.pi
        u1 = torch.cos(phi)
        u2 = torch.sin(phi)

        scale = torch.sqrt(2 * T_target)
        w1 = scale * (u1 / L11 - L21 * u2 / (L11 * L22))
        w2 = scale * (u2 / L22)

        new_ics = torch.stack([th1, th2, w1, w2], dim=-1)
        ics.append(new_ics)
        count += new_ics.shape[0]

    return torch.cat(ics, dim=0)[:n_samples]


# ---------------------------------------------------------------------------
# Single Non-Linear Pendulum (2D)
# ---------------------------------------------------------------------------
class SinglePendulumDynamics(nn.Module):

    def __init__(self, g, l, damping):
        super().__init__()
        self.g, self.l, self.damping = g, l, damping

    def forward(self, t, y):
        theta = y[..., 0]
        omega = y[..., 1]
        dtheta = omega
        domega = -(self.g / self.l) * torch.sin(theta) - self.damping * omega
        return torch.stack([dtheta, domega], dim=-1)


def _add_single_pendulum_args(parser):
    parser.add_argument('--m', type=float, default=1.0)
    parser.add_argument('--l', type=float, default=1.0)
    parser.add_argument('--g', type=float, default=9.81)
    parser.add_argument('--damping', type=float, default=0.0)


def _make_single_pendulum(args):
    return SinglePendulumDynamics(args.g, args.l, args.damping)


def _single_pendulum_energy(y, args):
    theta = y[..., 0]
    omega = y[..., 1]
    T = 0.5 * args.m * (args.l ** 2) * omega ** 2
    V = -args.m * args.g * args.l * torch.cos(theta)
    return T + V


# ---------------------------------------------------------------------------
# Damped Harmonic Oscillator (2D)
# ---------------------------------------------------------------------------
class DampedOscillatorDynamics(nn.Module):

    def __init__(self, mass, k, c):
        super().__init__()
        self.mass, self.k, self.c = mass, k, c

    def forward(self, t, y):
        x = y[..., 0]
        v = y[..., 1]
        dx = v
        dv = -(self.k / self.mass) * x - (self.c / self.mass) * v
        return torch.stack([dx, dv], dim=-1)


def _add_oscillator_args(parser):
    parser.add_argument('--mass', type=float, default=1.0)
    parser.add_argument('--k', type=float, default=4.0, help='Spring constant')
    parser.add_argument('--c', type=float, default=0.5, help='Damping coefficient')


def _make_oscillator(args):
    return DampedOscillatorDynamics(args.mass, args.k, args.c)


def _oscillator_energy(y, args):
    x = y[..., 0]
    v = y[..., 1]
    T = 0.5 * args.mass * v ** 2
    V = 0.5 * args.k * x ** 2
    return T + V


# ---------------------------------------------------------------------------
# Lorenz System (3D)
# ---------------------------------------------------------------------------
class LorenzDynamics(nn.Module):

    def __init__(self, sigma, rho, beta):
        super().__init__()
        self.sigma, self.rho, self.beta = sigma, rho, beta

    def forward(self, t, y):
        x = y[..., 0]
        yy = y[..., 1]
        z = y[..., 2]
        dx = self.sigma * (yy - x)
        dy = x * (self.rho - z) - yy
        dz = x * yy - self.beta * z
        return torch.stack([dx, dy, dz], dim=-1)


def _add_lorenz_args(parser):
    parser.add_argument('--sigma', type=float, default=10.0)
    parser.add_argument('--rho', type=float, default=28.0)
    parser.add_argument('--beta', type=float, default=8.0 / 3.0)


def _make_lorenz(args):
    return LorenzDynamics(args.sigma, args.rho, args.beta)


# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------
SYSTEMS = {
    'double_pendulum': {
        'name': 'Double Pendulum',
        'state_dim': 4,
        'state_labels': [r'$\theta_1$', r'$\theta_2$', r'$\omega_1$', r'$\omega_2$'],
        'dynamics_fn': _make_double_pendulum,
        'energy_fn': _double_pendulum_energy,
        'sample_ics_fn': _sample_double_pendulum_ics,
        'compute_E_max_fn': _compute_double_pendulum_E_max,
        'default_y0': [1.0, 1.5, 0.0, 0.0],
        'phase_indices': (0, 1),
        'phase_labels': (r'$\theta_1$', r'$\theta_2$'),
        'add_cli_args': _add_double_pendulum_args,
        'angle_indices': [0, 1],
        'default_t_end': 30.0,
        'default_t_extrap': 45.0,
    },
    'single_pendulum': {
        'name': 'Single Non-Linear Pendulum',
        'state_dim': 2,
        'state_labels': [r'$\theta$', r'$\omega$'],
        'dynamics_fn': _make_single_pendulum,
        'energy_fn': _single_pendulum_energy,
        'sample_ics_fn': None,
        'compute_E_max_fn': None,
        'default_y0': [2.0, 0.0],
        'phase_indices': (0, 1),
        'phase_labels': (r'$\theta$', r'$\omega$'),
        'add_cli_args': _add_single_pendulum_args,
        'angle_indices': [0],
        'default_t_end': 20.0,
        'default_t_extrap': 30.0,
    },
    'damped_oscillator': {
        'name': 'Damped Harmonic Oscillator',
        'state_dim': 2,
        'state_labels': [r'$x$', r'$v$'],
        'dynamics_fn': _make_oscillator,
        'energy_fn': _oscillator_energy,
        'sample_ics_fn': None,
        'compute_E_max_fn': None,
        'default_y0': [2.0, 0.0],
        'phase_indices': (0, 1),
        'phase_labels': (r'$x$', r'$v$'),
        'add_cli_args': _add_oscillator_args,
        'angle_indices': [],
        'default_t_end': 15.0,
        'default_t_extrap': 25.0,
    },
    'lorenz': {
        'name': 'Lorenz System',
        'state_dim': 3,
        'state_labels': [r'$x$', r'$y$', r'$z$'],
        'dynamics_fn': _make_lorenz,
        'energy_fn': None,
        'sample_ics_fn': None,
        'compute_E_max_fn': None,
        'default_y0': [1.0, 1.0, 1.0],
        'phase_indices': (0, 2),
        'phase_labels': (r'$x$', r'$z$'),
        'add_cli_args': _add_lorenz_args,
        'angle_indices': [],
        'default_t_end': 10.0,
        'default_t_extrap': 15.0,
    },
}


def get_system(name):
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system '{name}'. Choose from: {list(SYSTEMS.keys())}")
    return SYSTEMS[name]
