"""
Time-aware LeJEPA for variable-star light curves (synthetic test bed).

Latent split:
  c   : one static code per window (period, amplitude, shape, class)
  s_i : one dynamic state per observation (phase / current state)

Model variants (set by `predictor` and `sigreg_mode`):
  lejepa      : plain LeJEPA. Pooled embedding per window, the two windows are
                pulled together, SIGReg across the batch. No predictor.
  mlp+batch   : per-token JEPA. MLP predictor gets (s_i, c, dt). SIGReg on s
                across the batch (the usual placement).
  mlp+time    : same MLP predictor, SIGReg on s within each star over time.
  rot+batch   : rotation/damping ODE predictor, SIGReg across the batch.
  rot+time    : rotation/damping ODE predictor + temporal SIGReg (full model).
  mjepa-b     : masked latent prediction. Same masking and same predictor
                network as the MAE, but the target is the encoder's own token
                latents (from the full window) instead of magnitudes.
                SIGReg on all target tokens across the batch.
  mjepa-2     : same, with two-level SIGReg: window-mean latents across the
                batch (static part) and within-window centered latents
                (dynamic part).
  suffix -noerr : the encoder does not get the measurement-error input.
  flag +int   : rotation predictor uses integer harmonics k * w_LS.
  flag +tsig  : LeNEPA-style temporal SIGReg. SIGReg is applied to EACH star's
                own set of token states (not pooled over the batch), so every
                light curve must spread its states over time.
  flag +tvc   : temporal variance-covariance rule (VICReg over time): inside each
                star, every latent dimension must have unit spread over time and
                the dimensions must be uncorrelated. It fixes only the spread,
                not the shape, so a circle (phase) can satisfy it.
  flag +sg    : stop-gradient on the prediction target.
  flag +free  : the second half of the rotation blocks have speed 0 (pure
                damping), so they can hold slowly changing state (e.g. the
                random-walk value) instead of being forced to rotate.
  flag +mae   : hybrid. Adds the masked-reconstruction loss (same encoder,
                masked input, small decoder) to the latent prediction loss.
                Encoder, predictor and decoder are trained jointly.
  flag +inv   : inverse-dynamics head, no negatives. From the two states
                (s_i, s_j) and c, a small MLP must predict the phase advance
                between them, (cos, sin) of w_LS * dt. A latent that is constant
                within a star cannot do this, so time information is required.
  flag +fwd   : forward-only pairs: the target is always in a LATER window
                than the source, so no within-window pairs (whose source state
                has already attended to the target's neighbours).
  flag +cmask : the source states come from a causal pass of the encoder
                (each token sees only earlier points of its window); targets,
                SIGReg and negatives use the normal bidirectional pass.
  flag +dis   : MotionJEPA-style DISReg adapted to light curves. A small
                "difference encoder" embeds the change between the raw local
                patches around two observations (patch_j - patch_i, dt), with
                its own SIGReg so it cannot collapse. A predictor must produce
                that embedding from (s_i, s_j, c). Weights follow the paper:
                0.5 * MSE + 2 * SIGReg(d). Self-supervised, no labels, no
                period estimate, no reconstruction.
  flag +dis2  : improved DISReg for irregular sampling. Time-aligned local
                features at each observation (weighted quadratic fit over real
                time offsets: level, slope, curvature, effective count), an
                antisymmetric change target d = g(x) - g(-x) with x = feature
                differences and dt, SIGReg on d, and TWO heads: inverse
                (s_i, s_j, c) -> d as in MotionJEPA, and forward (s_i, c, dt) -> d.
  flag +ham   : LeHami-style symplectic predictor. q = s, p = time derivative
                of q along the star's trajectory (local quadratic fit over the
                irregular times, a Savitzky-Golay analogue). A separable
                Hamiltonian H = |p|^2/2 + V(q, c, log w_LS) is integrated with
                an exactly symplectic leapfrog (several sub-steps per gap).
                Loss: MSE on q and p after the rollout, SIGReg on q only.
                Used INSTEAD of the rotation predictor for pairs up to HAM_DT_MAX.
  flag +nce   : adds a contrastive term: the predicted state must pick out the
                true target token among ALL tokens of the same star. Any feature
                that is constant within a star cannot help, so static shortcuts
                are ruled out without having to find them by hand.
  angle       : action-angle JEPA. Dynamic state per observation = a point on
                the unit circle (phase) + a few real damped (OU) states; static
                code c = the "actions" (shape, period, amplitude). Predictor:
                zero-parameter rotation by w_LS * dt. Loss: von Mises likelihood
                on the angle (kappa learned from (c, dt), so uncertainty widens
                the angle instead of shrinking a vector), per-star circular
                uniformity (Fourier moments k=1..3 -> 0) as the anti-collapse
                term, plus MSE + per-star SIGReg on the OU states.
  mae         : masked autoencoder baseline. Same encoder; it sees only the
                visible observations, and a small decoder reconstructs the
                magnitudes of the hidden ones at their times.
"""
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# 1. Synthetic data
# --------------------------------------------------------------------------
CLASS_NAMES = ["sinusoid", "sawtooth (RR Lyr-like)", "eclipsing", "damped random walk"]


def _shape(cls, phi):
    """Waveform for each periodic class. phi in [0, 1)."""
    sin = np.sin(2 * np.pi * phi)
    r = 0.15
    saw = np.where(phi < r, -1 + 2 * phi / r, 1 - 2 * (phi - r) / (1 - r))
    def dip(c, w):
        d = ((phi - c + 0.5) % 1.0) - 0.5
        return np.exp(-0.5 * (d / w) ** 2)
    ecl = 0.6 - 2.0 * dip(0.0, 0.035) - 1.0 * dip(0.5, 0.035)
    return np.where(cls == 0, sin, np.where(cls == 1, saw, ecl))


def make_batch(B, N=64, T_w=20.0, gap_max=20.0, P_min=1.0, P_max=10.0,
               rng=None, nightly=False, realistic=False, hetero=None, n_windows=2):
    """n_windows windows per star, one after another, each followed by a random
    gap in [0, gap_max]. Returns numpy arrays with W = n_windows.
      t    : (B, W, N) absolute times in days
      mag  : (B, 2, N) observed magnitudes (centered-ish)
      err  : (B, 2, N) measurement errors
      cls  : (B,) class id
      logP : (B,) log10 period (nan for DRW)
      phase: (B, 2, N) true phase in [0,1) (nan for DRW)
      state: (B, 2, N) true noise-free DRW value (nan for periodic)
    """
    rng = rng or np.random.default_rng()
    cls = rng.integers(0, 4, size=B)
    P = np.exp(rng.uniform(np.log(P_min), np.log(P_max), size=B))
    phase0 = rng.uniform(0, 1, size=B)
    A = rng.uniform(0.3, 1.0, size=B)
    sig = rng.uniform(0.02, 0.15, size=B)
    tau = np.exp(rng.uniform(np.log(1.0), np.log(20.0), size=B))
    W = n_windows
    gaps = rng.uniform(0, gap_max, size=(B, W - 1))

    def sample_times(start):
        if nightly:  # observe only in the first 30% of each day
            day = rng.integers(0, int(T_w), size=(B, N)).astype(float)
            frac = rng.uniform(0, 0.3, size=(B, N))
            tt = day + frac
        else:
            tt = rng.uniform(0, T_w, size=(B, N))
        return np.sort(tt, axis=1) + start[:, None]

    starts = np.zeros((B, W))
    for w in range(1, W):
        starts[:, w] = starts[:, w - 1] + T_w + gaps[:, w - 1]
    t = np.stack([sample_times(starts[:, w]) for w in range(W)], 1)  # (B,W,N)

    phi = (t / P[:, None, None] + phase0[:, None, None]) % 1.0
    periodic = _shape(cls[:, None, None], phi)

    # damped random walk (Ornstein-Uhlenbeck), exact sampling across both windows
    tf = t.reshape(B, W * N)
    x = np.zeros((B, W * N))
    x[:, 0] = rng.normal(size=B)
    for k in range(1, W * N):
        a = np.exp(-(tf[:, k] - tf[:, k - 1]) / tau)
        x[:, k] = a * x[:, k - 1] + np.sqrt(1 - a ** 2) * rng.normal(size=B)
    drw = x.reshape(B, W, N)

    is_drw = (cls == 3)[:, None, None]
    signal = np.where(is_drw, drw, periodic)
    if realistic:
        # per-point errors that vary around the star's typical level
        err = sig[:, None, None] * np.exp(rng.normal(0, 0.4, size=t.shape))
        noise = err * rng.normal(size=t.shape)
        # red-noise systematics (slow drifts, e.g. calibration), OU over both windows
        a_sys = rng.uniform(0.05, 0.3, size=B)
        tau_sys = np.exp(rng.uniform(np.log(3.0), np.log(30.0), size=B))
        ys = np.zeros((B, W * N)); ys[:, 0] = rng.normal(size=B)
        for k in range(1, W * N):
            a = np.exp(-(tf[:, k] - tf[:, k - 1]) / tau_sys)
            ys[:, k] = a * ys[:, k - 1] + np.sqrt(1 - a ** 2) * rng.normal(size=B)
        noise = noise + a_sys[:, None, None] * ys.reshape(B, W, N)
        # outliers: 3% of points jump by 0.5-1.5 mag, with no warning in err
        out = rng.random(t.shape) < 0.03
        noise = noise + out * rng.choice([-1, 1], size=t.shape) * rng.uniform(0.5, 1.5, size=t.shape)
        mag = A[:, None, None] * signal + noise
    else:
        if hetero == "point":
            # heteroscedastic: every point has its own error, around a fixed star level
            err = sig[:, None, None] * np.exp(rng.normal(0, 0.5, size=t.shape))
        elif hetero == "window":
            # heteroscedastic AND the level changes between windows (new season / instrument)
            lvl = sig[:, None] * np.exp(rng.normal(0, 0.5, size=(B, W)))
            err = lvl[:, :, None] * np.exp(rng.normal(0, 0.3, size=t.shape))
        else:
            err = np.broadcast_to(sig[:, None, None], t.shape).copy()
        mag = A[:, None, None] * signal + err * rng.normal(size=t.shape)

    logP = np.where(cls == 3, np.nan, np.log10(P))
    phase = np.where(is_drw, np.nan, phi)
    state = np.where(is_drw, drw, np.nan)
    return dict(t=t, mag=mag, err=err, cls=cls, logP=logP, phase=phase,
                state=state, A=A, tau=tau, sig=sig)


# --------------------------------------------------------------------------
# 2. Encoder: transformer with continuous-time RoPE
#    Attention only sees time differences, so the encoder is exactly
#    shift-invariant in time. It cannot build a "clock" s_i = f(t_i).
# --------------------------------------------------------------------------
class RoPEAttention(nn.Module):
    def __init__(self, d, h, p_min=0.3, p_max=60.0):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        nf = self.dh // 2
        periods = torch.logspace(math.log10(p_min), math.log10(p_max), nf)
        self.register_buffer("freq", 2 * math.pi / periods)  # rad/day

    def rope(self, x, t):  # x: (B,h,N,dh), t: (B,N)
        ang = t[:, None, :, None] * self.freq  # (B,1,N,nf)
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

    def forward(self, x, t, causal=False):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, t), self.rope(k, t)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)  # tokens are time-sorted
        return self.out(y.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = RoPEAttention(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, t, causal=False):
        x = x + self.att(self.n1(x), t, causal)
        return x + self.mlp(self.n2(x))


class Encoder(nn.Module):
    def __init__(self, d=64, layers=3, heads=4, K=8, d_c=32):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.s_head = nn.Linear(d, 2 * K)  # dynamic state per token
        self.c_head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d_c))

    def forward(self, t, mag, err, causal=False):
        # t: (B,N) times relative to the window's first observation
        # causal=True: each token only sees earlier tokens of its window
        x = self.inp(torch.stack([mag, err * 5], -1))
        for b in self.blocks:
            x = b(x, t, causal)
        h = self.norm(x)
        s = self.s_head(h)
        c = self.c_head(h.mean(1))
        return h, s, c


@torch.no_grad()
def lomb_scargle_w(t, y, f_min=0.07, f_max=1.15, n_f=400, chunk=64):
    """Classical Lomb-Scargle on each window. t, y: (M, N).
    Returns the peak angular frequency (rad/day), refined with a parabola."""
    f = torch.linspace(f_min, f_max, n_f, device=t.device)
    w = 2 * math.pi * f
    y = y - y.mean(1, keepdim=True)
    ps = []
    for a0 in range(0, t.shape[0], chunk):
        tt, yy = t[a0:a0 + chunk], y[a0:a0 + chunk]
        wt = tt[..., None] * w                                   # (m, N, F)
        tau2 = torch.atan2(torch.sin(2 * wt).sum(1), torch.cos(2 * wt).sum(1))
        arg = wt - 0.5 * tau2[:, None, :]
        c, s = arg.cos(), arg.sin()
        ps.append((yy[..., None] * c).sum(1) ** 2 / (c ** 2).sum(1)
                  + (yy[..., None] * s).sum(1) ** 2 / (s ** 2).sum(1))
    p = torch.cat(ps)
    k = p.argmax(1).clamp(1, n_f - 2)
    r = torch.arange(len(k), device=t.device)
    a, b, cc = p[r, k - 1], p[r, k], p[r, k + 1]
    shift = 0.5 * (a - cc) / (a - 2 * b + cc - 1e-12)
    return 2 * math.pi * (f[k] + shift.clamp(-1, 1) * (f[1] - f[0]))


def window_w_ls(bt):
    """Lomb-Scargle angular frequency for each window: (B, 2)."""
    B, W, N = bt["t"].shape
    t = bt["t"].reshape(B * W, N)
    return lomb_scargle_w(t - t[:, :1], bt["mag"].reshape(B * W, N)).view(B, W)


# --------------------------------------------------------------------------
# 3. Predictors
# --------------------------------------------------------------------------
class RotationPredictor(nn.Module):
    """s_hat_j = exp(dt * A(c)) s_i, with A(c) block-diagonal 2x2 blocks
    [[-g_k, -w_k],[w_k, -g_k]],  w_k = (k/2) * w0,  w0 = w_LS.
    w_LS is a Lomb-Scargle estimate from the SOURCE window only. Half-integer
    harmonics cover periodograms that peak at P/2 (eclipsing binaries)."""
    def __init__(self, K=8, d_c=32, half=True, free=False):
        super().__init__()
        self.K, self.half, self.free = K, half, free
        self.net = nn.Sequential(nn.Linear(d_c, 64), nn.GELU(), nn.Linear(64, 1 + K))
        with torch.no_grad():  # start with delta = 0 and small damping
            self.net[-1].weight[0].zero_()
            self.net[-1].bias[0] = 0.0
            self.net[-1].bias[1:] = -4.0

    def params(self, c, w_ls):
        o = self.net(c)
        w0 = w_ls  # rotation speed comes from the periodogram; c sets the damping
        g = F.softplus(o[:, 1:])
        return w0, g

    def forward(self, s_i, c, dt, w_ls):
        # s_i: (M, 2K), c: (M, d_c), dt: (M,), w_ls: (M,)
        w0, g = self.params(c, w_ls)
        k = torch.arange(1, self.K + 1, device=s_i.device, dtype=s_i.dtype)
        if self.half:  # half-integer harmonics; "+int" variants use integer ones
            k = 0.5 * k
        if self.free:  # second half of the blocks: no rotation, damping only
            k = torch.cat([k[: self.K // 2], torch.zeros(self.K - self.K // 2, device=k.device, dtype=k.dtype)])
        ang = (w0[:, None] * k) * dt[:, None]
        damp = torch.exp(-g * dt[:, None])
        x1, x2 = s_i[:, 0::2], s_i[:, 1::2]
        y1 = damp * (x1 * ang.cos() - x2 * ang.sin())
        y2 = damp * (x1 * ang.sin() + x2 * ang.cos())
        return torch.stack([y1, y2], -1).flatten(-2)


class MLPPredictor(nn.Module):
    """Generic JEPA predictor conditioned on c and Fourier features of dt."""
    def __init__(self, K=8, d_c=32, hidden=256, nf=16):
        super().__init__()
        periods = torch.logspace(math.log10(0.3), math.log10(80), nf)
        self.register_buffer("freq", 2 * math.pi / periods)
        self.net = nn.Sequential(nn.Linear(2 * K + d_c + 4 * nf + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * K))

    def forward(self, s_i, c, dt, w_ls):
        # same information as the rotation predictor: dt features plus
        # the LS phase advance (k/2) * w_LS * dt
        a = dt[:, None] * self.freq
        kk = 0.5 * torch.arange(1, self.freq.numel() + 1, device=dt.device)
        ph = (w_ls * dt)[:, None] * kk
        f = torch.cat([a.sin(), a.cos(), ph.sin(), ph.cos(), (dt[:, None] / 40.0)], -1)
        return self.net(torch.cat([s_i, c, f], -1))


MAE_MASK_RATIO = 0.6


class MAEDecoder(nn.Module):
    """Takes the encoder's visible tokens plus one learned mask token per hidden
    observation (placed at its time through RoPE) and predicts its magnitude."""
    def __init__(self, d=64, layers=2, heads=4, out_dim=1):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, out_dim)

    def forward(self, h_vis, t_vis, t_mask):
        B, M = t_mask.shape
        x = torch.cat([self.proj(h_vis), self.mask_tok.expand(B, M, -1)], 1)
        t = torch.cat([t_vis, t_mask], 1)
        for b in self.blocks:
            x = b(x, t)
        y = self.out(self.norm(x[:, h_vis.shape[1]:]))
        return y.squeeze(-1) if y.shape[-1] == 1 else y


def mae_loss(model, bt, mask_ratio=MAE_MASK_RATIO):
    B, W, N = bt["t"].shape
    t = bt["t"].reshape(B * W, N)
    t = t - t[:, :1]
    mag, err = bt["mag"].reshape(B * W, N), bt["err"].reshape(B * W, N)
    n_vis = int(round(N * (1 - mask_ratio)))
    perm = torch.rand(B * W, N, device=t.device).argsort(1)
    vis, msk = perm[:, :n_vis].sort(1).values, perm[:, n_vis:].sort(1).values
    g = lambda x, idx: x.gather(1, idx)
    h, _, _ = model.enc(g(t, vis), g(mag, vis), g(err, vis))
    pred = model.dec(h, g(t, vis), g(t, msk))
    L = (pred - g(mag, msk)).pow(2).mean()
    return L, dict(rec=L.item())


def mjepa_loss(model, bt, mask_ratio=MAE_MASK_RATIO, lam_sig=0.05):
    """Masked latent prediction. Context = visible tokens. Target = latent
    tokens z of the FULL window at the hidden times. The predictor gets the
    hidden times through RoPE, so it is conditioned on time."""
    B, W, N = bt["t"].shape
    t = bt["t"].reshape(B * W, N)
    t = t - t[:, :1]
    mag, err = bt["mag"].reshape(B * W, N), bt["err"].reshape(B * W, N)
    if model.noerr:
        err = torch.zeros_like(err)
    n_vis = int(round(N * (1 - mask_ratio)))
    perm = torch.rand(B * W, N, device=t.device).argsort(1)
    vis, msk = perm[:, :n_vis].sort(1).values, perm[:, n_vis:].sort(1).values
    g = lambda x, idx: x.gather(1, idx)
    h_ctx, _, _ = model.enc(g(t, vis), g(mag, vis), g(err, vis))
    h_full, _, _ = model.enc(t, mag, err)
    z = model.z_head(h_full)                                  # (BW, N, dz)
    z_msk = z.gather(1, msk[..., None].expand(-1, -1, z.shape[-1]))
    pred = model.dec(h_ctx, g(t, vis), g(t, msk))
    L_pred = (pred - z_msk).pow(2).mean()
    if model.variant.startswith("mjepa-2"):
        zm = z.mean(1)                                        # static part, per window
        L_sig = sigreg(zm) + sigreg((z - zm[:, None]).reshape(-1, z.shape[-1]))
    else:
        L_sig = sigreg(z.reshape(-1, z.shape[-1]))
    return L_pred + lam_sig * L_sig, dict(pred=L_pred.item(), sig=L_sig.item())


# --------------------------------------------------------------------------
# 4. SIGReg (sliced Epps-Pulley test against N(0, I))
# --------------------------------------------------------------------------
SIGREG_SLICES = 256   # 1024 in the LeJEPA repo; set lcjepa.SIGREG_SLICES = 1024 on a GPU
# Loss weights, following the papers (SIGReg here has the official scale: statistic * N):
LAM = dict(sig_c=0.05,    # LeJEPA: (1-lam)*pred + lam*SIGReg, lam = 0.05
           sig_s=0.05,    # same weight on the per-token states (batch placement)
           tsig=20.0,     # LeNEPA: temporal SIGReg weight lambda_T = 20
           dis_z=0.25, dis_d=2.0, dis_pred=0.5,   # MotionJEPA: lambda_z, lambda_d, lambda_pred
           ham=0.25,      # LeHami: lambda_S on q
           mean_s=1.0, c_inv=1.0)


def _ep_weights(t_max=3.0, n_t=17, device="cpu"):
    tt = torch.linspace(0, t_max, n_t, device=device)
    dt = t_max / (n_t - 1)
    w = torch.full((n_t,), 2 * dt, device=device); w[0] = dt; w[-1] = dt   # symmetric range via doubled weights
    phi = torch.exp(-0.5 * tt ** 2)
    return tt, phi, w * phi


def sigreg(z, num_slices=None, t_max=3.0, n_t=17):
    """SIGReg exactly as in the LeJEPA repository: sliced Epps-Pulley statistic,
    T = N * int |phi_emp(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt over [-t_max, t_max],
    averaged over random unit directions. ~1 for a Gaussian sample of any size."""
    n, d = z.shape
    M = num_slices or SIGREG_SLICES
    A = torch.randn(d, M, device=z.device)
    A = A / A.norm(dim=0, keepdim=True)
    tt, phi, w = _ep_weights(t_max, n_t, z.device)
    x = (z @ A)[..., None] * tt                       # (n, M, n_t)
    err = (x.cos().mean(0) - phi) ** 2 + x.sin().mean(0) ** 2
    return ((err @ w) * n).mean()


def sigreg_per_sample(z, num_slices=None, t_max=3.0, n_t=17):
    """SIGReg applied separately to each sample's own token set (LeNEPA's
    temporal placement), official scale with N = tokens per sample. z: (B, T, d)."""
    B, T, d = z.shape
    M = num_slices or SIGREG_SLICES
    A = torch.randn(d, M, device=z.device)
    A = A / A.norm(dim=0, keepdim=True)
    tt, phi, w = _ep_weights(t_max, n_t, z.device)
    x = (z @ A)[..., None] * tt                       # (B, T, M, n_t)
    err = (x.cos().mean(1) - phi) ** 2 + x.sin().mean(1) ** 2
    return ((err @ w) * T).mean()


def temporal_var_cov(z, eps=1e-4):
    """VICReg-style variance + covariance terms computed over time within each
    sample. z: (B, T, d). Returns (variance hinge, off-diagonal covariance)."""
    B, T, d = z.shape
    zc = z - z.mean(1, keepdim=True)
    cov = zc.transpose(1, 2) @ zc / (T - 1)               # (B, d, d)
    std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2) + eps)
    var_loss = F.relu(1 - std).mean()
    off = cov - torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
    cov_loss = off.pow(2).sum((1, 2)).mean() / d
    return var_loss, cov_loss


def _neighbors(idx, N, offsets):
    """Indices of neighbouring tokens inside the same window, clamped. idx: (B,P)."""
    w = idx // N
    out = [((idx + o).clamp(min=0)).minimum(w * N + N - 1).maximum(w * N) for o in offsets]
    return torch.stack(out, -1)                                            # (B,P,len)


def local_patch(mag_f, t_f, idx, N, offsets=(-2, -1, 1, 2)):
    """Raw local patch around each token: neighbour magnitudes relative to the
    token's own magnitude, and neighbour times relative to the token's time."""
    nb = _neighbors(idx, N, offsets)                                        # (B,P,4)
    m0, t0 = mag_f.gather(1, idx), t_f.gather(1, idx)
    dm = torch.stack([mag_f.gather(1, nb[..., k]) for k in range(nb.shape[-1])], -1) - m0[..., None]
    dt = torch.stack([t_f.gather(1, nb[..., k]) for k in range(nb.shape[-1])], -1) - t0[..., None]
    return dm, dt


def latent_time_derivative(s, t):
    """p_i = d s / d t at each token, from a least-squares quadratic fit of s
    over the token and its two neighbours on each side (within the window),
    using the actual irregular times. s: (M,N,d), t: (M,N). Returns (M,N,d)."""
    M, N, d = s.shape
    idx = torch.arange(N, device=s.device)[None, :].expand(M, N)
    offs = (-2, -1, 0, 1, 2)
    nb = torch.stack([(idx + o).clamp(0, N - 1) for o in offs], -1)         # (M,N,5)
    tn = t.gather(1, nb.reshape(M, -1)).view(M, N, 5)
    tau = tn - t[..., None]                                                 # (M,N,5)
    sn = s.gather(1, nb.reshape(M, -1)[..., None].expand(-1, -1, d)).view(M, N, 5, d)
    X = torch.stack([torch.ones_like(tau), tau, tau ** 2], -1)             # (M,N,5,3)
    XtX = X.transpose(-1, -2) @ X + 1e-3 * torch.eye(3, device=s.device)
    beta = torch.linalg.solve(XtX, X.transpose(-1, -2) @ sn)               # (M,N,3,d)
    return beta[..., 1, :]                                                  # derivative term


DIS2_WIDTH = 0.75   # Gaussian width in days for the local fit


def local_fit_features(mag_f, t_f, N, width=None, K=4):
    """Time-aligned local features at every token: weighted least-squares fit
    m(tau) = a + b tau + c tau^2 over the 2K index-neighbours inside the window,
    with Gaussian weights in the REAL time offset tau (width in days). Returns
    (M, T, 4): [level - own mag, slope, curvature, log effective count]."""
    width = width or DIS2_WIDTH
    B, T = mag_f.shape
    idx = torch.arange(T, device=mag_f.device)[None, :].expand(B, T)
    nb = _neighbors(idx, N, tuple(o for o in range(-K, K + 1)))         # (B,T,2K+1)
    m = torch.stack([mag_f.gather(1, nb[..., k]) for k in range(nb.shape[-1])], -1)
    tau = torch.stack([t_f.gather(1, nb[..., k]) for k in range(nb.shape[-1])], -1) - t_f[..., None]
    w = torch.exp(-0.5 * (tau / width) ** 2)
    X = torch.stack([torch.ones_like(tau), tau, tau ** 2], -1)          # (B,T,2K+1,3)
    Xw = X * w[..., None]
    XtX = Xw.transpose(-1, -2) @ X + 1e-2 * torch.eye(3, device=mag_f.device)
    beta = torch.linalg.solve(XtX, Xw.transpose(-1, -2) @ m[..., None])[..., 0]   # (B,T,3)
    n_eff = w.sum(-1, keepdim=True)
    return torch.cat([beta[..., :1] - mag_f[..., None], beta[..., 1:2], beta[..., 2:3], torch.log(n_eff)], -1)


class AntisymDiffEncoder(nn.Module):
    """d = g(x) - g(-x): antisymmetric in the change x, so d(i->j) = -d(j->i)
    and static (symmetric) content cannot enter the target."""
    def __init__(self, d_in=5, d_out=16):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, d_out))

    def forward(self, x):
        return self.g(x) - self.g(-x)


class DiffEncoder(nn.Module):
    """DISReg difference encoder: embeds (patch_j - patch_i, times, dt)."""
    def __init__(self, d_out=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1 + 4 + 4 + 4 + 1, 64), nn.GELU(), nn.Linear(64, d_out))

    def forward(self, dm_i, dt_i, dm_j, dt_j, m_diff, dt_ij):
        x = torch.cat([m_diff[..., None], dm_j - dm_i, dt_i, dt_j, dt_ij[..., None] / 20.0], -1)
        return self.net(x)


class HamiltonianFlow(nn.Module):
    """Separable Hamiltonian H = |p|^2 / 2 + V(q, cond), integrated by leapfrog.
    Exactly symplectic for any V, any step size, any parameters (LeHami, Prop. 1)."""
    def __init__(self, d_q, d_cond, hidden=128):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(d_q + d_cond, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def grad_V(self, q, cond):
        with torch.enable_grad():
            q = q.requires_grad_(True) if not q.requires_grad else q
            v = self.V(torch.cat([q, cond], -1)).sum()
            return torch.autograd.grad(v, q, create_graph=True)[0]

    def forward(self, q, p, cond, dt, h_max=0.3):
        """Roll (q, p) forward by dt (per sample) with leapfrog sub-steps."""
        n_sub = int(max(1, math.ceil(float(dt.max()) / h_max)))
        h = (dt / n_sub)[:, None]
        p = p - 0.5 * h * self.grad_V(q, cond)
        for k in range(n_sub):
            q = q + h * p
            g = self.grad_V(q, cond)
            p = p - (h if k < n_sub - 1 else 0.5 * h) * g
        return q, p


HAM_DT_MAX, HAM_PAIRS = 6.0, 64


ANGLE_OU = 4          # number of real damped states beside the phase
ANGLE_LAM = dict(uni=1.0, ou=1.0, ou_sig=20.0)


def angle_loss(model, bt, dt_max, lam=None, n_pairs=512):
    """Action-angle JEPA loss. The s head's first 2 dims are the phase vector,
    the next ANGLE_OU dims are real damped states."""
    lam = lam or dict(LAM)
    h, s, c = model.encode_batch(bt)
    B, W, N, ds = s.shape
    u = F.normalize(s[..., :2], dim=-1)                                   # unit circle
    x = s[..., 2:2 + ANGLE_OU]
    i, j, dt, mask = sample_pairs(bt["t"], n_pairs, dt_max)
    g = lambda z, idx: z.reshape(B, W * N, -1).gather(1, idx[..., None].expand(-1, -1, z.shape[-1]))
    ui, uj, xi, xj = g(u, i), g(u, j), g(x, i), g(x, j)
    c_src = c.gather(1, (i // N)[..., None].expand(-1, -1, c.shape[-1]))
    w_src = window_w_ls(bt).gather(1, i // N)
    m = mask.float()
    # --- zero-parameter predictor: rotate the phase vector by w * dt
    th = w_src * dt
    pred = torch.stack([ui[..., 0] * th.cos() - ui[..., 1] * th.sin(),
                        ui[..., 0] * th.sin() + ui[..., 1] * th.cos()], -1)
    cos_err = (pred * uj).sum(-1)
    kappa = F.softplus(model.kappa(torch.cat([c_src, (dt / 20.0)[..., None], torch.log1p(dt)[..., None]], -1)))[..., 0] + 1e-3
    nll = kappa * (1 - cos_err) + torch.log(torch.special.i0e(kappa))      # von Mises NLL (+const)
    L_vm = (nll * m).sum() / m.sum().clamp(min=1)
    # --- per-star circular uniformity: Fourier moments of the phase -> 0
    phi = torch.atan2(u[..., 1], u[..., 0]).reshape(B, W * N)
    L_uni = sum(((torch.cos(k * phi).mean(1)) ** 2 + (torch.sin(k * phi).mean(1)) ** 2).mean() for k in (1, 2, 3))
    # --- damped real states: x_j = exp(-gamma dt) x_i, per-star SIGReg
    gam = F.softplus(model.ou_gamma(c_src))
    L_ou = (((torch.exp(-gam * dt[..., None]) * xi - xj).pow(2).mean(-1)) * m).sum() / m.sum().clamp(min=1)
    L_ou_sig = sigreg_per_sample(x.reshape(B, W * N, ANGLE_OU))
    # --- static code: window invariance + batch SIGReg (LeJEPA)
    L_cinv = (c - c.mean(1, keepdim=True)).pow(2).mean() * W / (W - 1)
    L_sig_c = sigreg(c.reshape(W * B, -1))
    loss = (L_vm + ANGLE_LAM["uni"] * L_uni + ANGLE_LAM["ou"] * L_ou + ANGLE_LAM["ou_sig"] * L_ou_sig
            + lam["c_inv"] * L_cinv + lam["sig_c"] * L_sig_c)
    logs = dict(vm=L_vm.item(), cos_err=float((cos_err * m).sum() / m.sum().clamp(min=1)), kappa=float(kappa.mean()),
                uni=L_uni.item(), ou=L_ou.item(), ou_sig=L_ou_sig.item(), cinv=L_cinv.item(), sig_c=L_sig_c.item(),
                npairs=int(m.sum()))
    return loss, logs


# --------------------------------------------------------------------------
# 5. Model + loss
# --------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, variant="rot+time", K=8, d_c=32, d=64, layers=3, heads=4):
        super().__init__()
        self.variant = variant
        self.enc = Encoder(d, layers, heads, K, d_c)
        if variant.startswith("rot"):
            self.pred = RotationPredictor(K, d_c, half="+int" not in variant, free="+free" in variant)
        elif variant.startswith("mlp"):
            self.pred = MLPPredictor(K, d_c)
        else:
            self.pred = None
        self.noerr = "-noerr" in variant
        base = variant.replace("-noerr", "")
        if base == "angle":
            self.kappa = nn.Sequential(nn.Linear(d_c + 2, 64), nn.GELU(), nn.Linear(64, 1))
            self.ou_gamma = nn.Sequential(nn.Linear(d_c, 32), nn.GELU(), nn.Linear(32, ANGLE_OU))
        if base == "mae":
            self.dec = MAEDecoder(d, 2, heads)
        if "+mae" in variant:
            self.dec = MAEDecoder(d, 2, heads)
        if "+dis" in variant and "+dis2" not in variant:
            self.diff_enc = DiffEncoder(16)
            self.diff_pred = nn.Sequential(nn.Linear(4 * K + d_c, 128), nn.GELU(), nn.Linear(128, 16))
        if "+dis2" in variant:
            self.diff2_enc = AntisymDiffEncoder(5, 16)
            self.diff2_inv = nn.Sequential(nn.Linear(4 * K + d_c, 128), nn.GELU(), nn.Linear(128, 16))
            self.diff2_fwd = nn.Sequential(nn.Linear(2 * K + d_c + 2, 128), nn.GELU(), nn.Linear(128, 16))
        if "+ham" in variant:
            self.ham = HamiltonianFlow(2 * K, d_c + 1)
        if "+inv" in variant:
            self.inv = nn.Sequential(nn.Linear(4 * K + d_c, 128), nn.GELU(), nn.Linear(128, 2))
        if base.startswith("mjepa"):
            self.z_head = nn.Linear(d, 32)
            self.dec = MAEDecoder(d, 2, heads, out_dim=32)

    def encode_batch(self, bt, causal=False):
        """bt: dict of torch tensors (B,W,N). Each window is encoded on its own.
        causal=True gives source states that only see the past of their window."""
        B, W, N = bt["t"].shape
        t = bt["t"].reshape(W * B, N)
        trel = t - t[:, :1]  # time relative to each window's first point
        err = bt["err"].reshape(W * B, N)
        if self.noerr:
            err = torch.zeros_like(err)
        h, s, c = self.enc(trel, bt["mag"].reshape(W * B, N), err, causal)
        h, s, c = h.view(B, W, N, -1), s.view(B, W, N, -1), c.view(B, W, -1)
        if self.variant.startswith(("mae", "mjepa")):  # no trained c head: mean-pool tokens
            c = h.mean(2)
        return h, s, c


def sample_pairs_fwd(t, n_pairs, dt_max, gen=None):
    """Pairs whose target lies in a strictly later window than the source."""
    B, W, N = t.shape
    tf = t.reshape(B, W * N).contiguous()
    dev = t.device
    i = torch.randint(0, (W - 1) * N, (B, n_pairs), generator=gen, device=dev)
    wi = i // N
    wj = wi + 1 + (torch.rand(B, n_pairs, generator=gen, device=dev) * (W - 1 - wi)).long()
    j = wj * N + torch.randint(0, N, (B, n_pairs), generator=gen, device=dev)
    dt = tf.gather(1, j) - tf.gather(1, i)
    mask = (dt > 0) & (dt <= 1.5 * dt_max)
    return i, j, dt, mask


def sample_pairs(t, n_pairs, dt_max, dt_min=0.02, gen=None):
    """Token pairs (i -> j) inside a star, over both windows.
    The target gap is drawn log-uniformly in [dt_min, dt_max], so short gaps
    stay common even late in training (they carry the clean signal for the
    rotation speed). j is the observation closest to t_i + gap."""
    B, W, N = t.shape
    tf = t.reshape(B, W * N).contiguous()  # sorted: window b is after window a
    dev = t.device
    i = torch.randint(0, W * N, (B, n_pairs), generator=gen, device=dev)
    ti = tf.gather(1, i)
    u = torch.rand(B, n_pairs, generator=gen, device=dev)
    gap = dt_min * (dt_max / dt_min) ** u
    j = torch.searchsorted(tf, (ti + gap).contiguous()).clamp(max=W * N - 1)
    jm = (j - 1).clamp(min=0)
    pick_prev = (tf.gather(1, jm) - (ti + gap)).abs() < (tf.gather(1, j) - (ti + gap)).abs()
    j = torch.where(pick_prev, jm, j)
    dt = tf.gather(1, j) - ti
    mask = (dt > 0) & (dt <= 1.5 * dt_max)
    return i, j, dt, mask


NCE_TAU, NCE_WEIGHT = 1.0, 0.2
MAE_WEIGHT = 1.0
INV_WEIGHT = 1.0


def loss_fn(model, bt, dt_max, lam=None, n_pairs=512):
    lam = lam or dict(LAM)
    if "+dis" in model.variant and "+dis2" not in model.variant:
        lam["sig_s"] = LAM["dis_z"]
    if "+ham" in model.variant:
        lam["sig_s"] = LAM["ham"]
    if model.variant.startswith("angle"):
        return angle_loss(model, bt, dt_max, lam)
    if model.variant.startswith("mae"):
        return mae_loss(model, bt)
    if model.variant.startswith("mjepa"):
        return mjepa_loss(model, bt)
    h, s, c = model.encode_batch(bt)
    B, W, N, ds = s.shape
    logs = {}

    if model.variant.startswith("lejepa"):
        z = c  # (B,W,d_c): pooled embedding per window, pulled to the star's mean
        inv = (z - z.mean(1, keepdim=True)).pow(2).mean() * W / (W - 1)
        sr = sigreg(z.reshape(W * B, -1))
        logs.update(inv=inv.item(), sig=sr.item())
        return (1 - lam["sig_c"]) * inv + lam["sig_c"] * sr, logs

    # ---- per-token prediction loss
    if "+fwd" in model.variant:
        i, j, dt, mask = sample_pairs_fwd(bt["t"], n_pairs, dt_max)
    else:
        i, j, dt, mask = sample_pairs(bt["t"], n_pairs, dt_max)
    sf = s.reshape(B, W * N, ds)
    if "+cmask" in model.variant:
        _, s_src, _ = model.encode_batch(bt, causal=True)
        si = s_src.reshape(B, W * N, ds).gather(1, i[..., None].expand(-1, -1, ds))
    else:
        si = sf.gather(1, i[..., None].expand(-1, -1, ds))
    sj = sf.gather(1, j[..., None].expand(-1, -1, ds))
    if "+sg" in model.variant:
        sj = sj.detach()
    c_src = c.gather(1, (i // N)[..., None].expand(-1, -1, c.shape[-1]))  # c of source window
    w_src = window_w_ls(bt).gather(1, i // N)               # LS of the source window
    m = mask.reshape(-1)
    P_ = i.shape[1]
    pred_all = model.pred(si.reshape(-1, ds), c_src.reshape(-1, c.shape[-1]),
                          dt.reshape(-1).clamp(min=0), w_src.reshape(-1)).view(B, P_, ds)
    L_pred = (pred_all.reshape(-1, ds)[m] - sj.reshape(-1, ds)[m]).pow(2).mean()
    if "+dis" in model.variant and "+dis2" not in model.variant:
        mag_f, t_f = bt["mag"].reshape(B, W * N), bt["t"].reshape(B, W * N)
        dm_i, dti = local_patch(mag_f, t_f, i, N)
        dm_j, dtj = local_patch(mag_f, t_f, j, N)
        m_diff = mag_f.gather(1, j) - mag_f.gather(1, i)
        d = model.diff_enc(dm_i, dti, dm_j, dtj, m_diff, dt)                # (B,P,16)
        d_hat = model.diff_pred(torch.cat([si, sj, c_src], -1))
        L_dis_pred = ((d_hat - d).pow(2).mean(-1) * mask).sum() / mask.sum().clamp(min=1)
        L_dis_sig = sigreg(d.reshape(-1, d.shape[-1])[m])
        L_pred = L_pred + LAM["dis_pred"] * L_dis_pred + LAM["dis_d"] * L_dis_sig
        logs.update(dis_pred=L_dis_pred.item(), dis_sig=L_dis_sig.item())
    if "+dis2" in model.variant:
        mag_f, t_f = bt["mag"].reshape(B, W * N), bt["t"].reshape(B, W * N)
        feat = local_fit_features(mag_f, t_f, N)                          # (B, W*N, 4)
        g4 = lambda x, idx: x.gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))
        fi, fj = g4(feat, i), g4(feat, j)
        x = torch.cat([(fj - fi)[..., :3], (mag_f.gather(1, j) - mag_f.gather(1, i))[..., None],
                       dt[..., None] / 20.0], -1)                                 # (B,P,5)
        d = model.diff2_enc(x)
        d_inv = model.diff2_inv(torch.cat([si, sj, c_src], -1))
        dtf = torch.stack([dt / 20.0, torch.log1p(dt)], -1)
        d_fwd = model.diff2_fwd(torch.cat([si, c_src, dtf], -1))
        L_inv2 = ((d_inv - d).pow(2).mean(-1) * mask).sum() / mask.sum().clamp(min=1)
        L_fwd2 = ((d_fwd - d).pow(2).mean(-1) * mask).sum() / mask.sum().clamp(min=1)
        L_sig2 = sigreg(d.reshape(-1, d.shape[-1])[m])
        L_pred = L_pred + LAM["dis_pred"] * (L_inv2 + L_fwd2) + LAM["dis_d"] * L_sig2
        logs.update(dis2_inv=L_inv2.item(), dis2_fwd=L_fwd2.item(), dis2_sig=L_sig2.item())
    if "+ham" in model.variant:
        # symplectic rollout on (q, p) for short gaps; replaces the rotation loss
        t_f = bt["t"].reshape(B, W * N)
        p_all = latent_time_derivative(s.reshape(B * W, N, ds), bt["t"].reshape(B * W, N)).view(B, W * N, ds)
        ih, jh, dth, mh = sample_pairs(bt["t"], HAM_PAIRS, min(dt_max, HAM_DT_MAX), dt_min=0.2)
        mh = mh & (dth <= HAM_DT_MAX)
        g = lambda x, idx: x.gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))
        q0, p0, q1, p1 = g(sf, ih), g(p_all, ih), g(sf, jh), g(p_all, jh)
        cond = torch.cat([g(c, ih // N), torch.log(window_w_ls(bt).gather(1, ih // N))[..., None]], -1)
        mm = mh.reshape(-1)
        qh, ph = model.ham(q0.reshape(-1, ds)[mm], p0.reshape(-1, ds)[mm], cond.reshape(-1, cond.shape[-1])[mm],
                           dth.reshape(-1)[mm])
        L_ham = (qh - q1.reshape(-1, ds)[mm]).pow(2).mean() + (ph - p1.reshape(-1, ds)[mm]).pow(2).mean()
        L_pred = L_ham                                                     # SIGReg on q (= s) stays as set above
        logs.update(ham=L_ham.item(), nsub=int(math.ceil(float(dth.reshape(-1)[mm].max()) / 0.3)))
    if "+inv" in model.variant:
        th = w_src * dt
        y = torch.stack([th.cos(), th.sin()], -1)                                  # (B, P, 2)
        x = torch.cat([si, sj, c_src], -1)
        L_inv = ((model.inv(x) - y).pow(2).mean(-1) * mask).sum() / mask.sum().clamp(min=1)
        L_pred = L_pred + INV_WEIGHT * L_inv
        logs["inv"] = L_inv.item()
    if "+nce" in model.variant:
        # negatives = every other token of the same star (both windows)
        logits = -torch.cdist(pred_all, sf).pow(2) / NCE_TAU      # (B, P, 2N)
        ce = F.cross_entropy(logits.reshape(-1, W * N), j.reshape(-1), reduction="none")
        L_nce = (ce * m).sum() / m.sum().clamp(min=1)
        L_pred = L_pred + NCE_WEIGHT * L_nce
        logs["nce"] = L_nce.item()

    # ---- SIGReg on s
    if "+tsig" in model.variant:
        L_sig_s = sigreg_per_sample(s.reshape(B, W * N, ds)) * (lam["tsig"] / lam["sig_s"])  # LeNEPA weight
        L_mean = torch.zeros((), device=s.device)
    elif "+tvc" in model.variant:
        s_c = s.reshape(B, W * N, ds)
        v_l, c_l = temporal_var_cov(s_c)
        L_sig_s = 10 * v_l + c_l
        L_mean = s_c.mean(1).pow(2).mean()                    # no static offset in s
        logs.update(tvar=v_l.item(), tcov=c_l.item())
    elif model.variant.endswith("time"):
        s_c = s.reshape(B, W * N, ds)
        s_mean = s_c.mean(1, keepdim=True)
        L_sig_s = sigreg((s_c - s_mean).reshape(-1, ds))  # within-star, over time
        L_mean = s_mean.pow(2).mean()                    # no static offset in s
    else:
        L_sig_s = sigreg(s.reshape(-1, ds))              # across batch
        L_mean = torch.zeros((), device=s.device)

    # ---- static code c: invariance across windows + batch SIGReg
    L_cinv = (c - c.mean(1, keepdim=True)).pow(2).mean() * W / (W - 1)
    L_sig_c = sigreg(c.reshape(W * B, -1))

    loss = (L_pred + lam["c_inv"] * L_cinv + lam["sig_s"] * L_sig_s
            + lam["mean_s"] * L_mean + lam["sig_c"] * L_sig_c)
    if "+mae" in model.variant:
        L_rec, _ = mae_loss(model, bt)
        loss = loss + MAE_WEIGHT * L_rec
        logs["rec"] = L_rec.item()
    logs.update(pred=L_pred.item(), sig_s=L_sig_s.item(), mean=L_mean.item(),
                cinv=L_cinv.item(), sig_c=L_sig_c.item(), npairs=int(m.sum()))
    return loss, logs


def to_torch(b, device):
    return {k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32, device=device)
            for k, v in b.items() if k in ("t", "mag", "err")}


@torch.no_grad()
def period_corr(model, rng, data_kw, device, B=256):
    """Correlation between the rotation predictor's period 2*pi/w0 and the
    true period, for periodic stars. Only meaningful for 'rot' variants."""
    d = make_batch(B, rng=rng, **(data_kw or {}))
    bt = to_torch(d, device)
    _, _, c = model.encode_batch(bt)
    w0, _ = model.pred.params(c[:, 0], window_w_ls(bt)[:, 0])
    sel = d["cls"] != 3
    r = w0.cpu().numpy()[sel] / (2 * np.pi / 10 ** d["logP"][sel])
    return float(((np.abs(r - 1) < 0.02) | (np.abs(r - 2) < 0.04)).mean())


def train(variant, steps=3000, B=32, lr=1e-3, data_kw=None, seed=0, device="cpu",
          dt_start=0.5, dt_end=60.0, log_every=250, verbose=True):
    """Joint, single-stage training of encoder + predictor.
    Curriculum: the largest allowed time gap grows from dt_start to dt_end."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data_kw = data_kw or {}
    model = Model(variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    hist = []
    for step in range(steps):
        frac = min(1.0, step / (0.6 * steps))
        dt_max = dt_start * (dt_end / dt_start) ** frac  # geometric curriculum
        bt = to_torch(make_batch(B, rng=rng, **data_kw), device)
        loss, logs = loss_fn(model, bt, dt_max)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % log_every == 0 or step == steps - 1:
            logs.update(step=step, dt_max=round(dt_max, 1), loss=round(loss.item(), 4))
            if variant.startswith("rot"):
                logs["w0_ok"] = period_corr(model, rng, data_kw, device)
            hist.append(logs)
            if verbose:
                print(variant, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in logs.items()},
                      f"{time.time() - t0:.0f}s", flush=True)
    return model, hist


# --------------------------------------------------------------------------
# 6. Evaluation: frozen probes
# --------------------------------------------------------------------------
@torch.no_grad()
def embed_dataset(model, data, device="cpu", bs=256):
    H, S, C = [], [], []
    model.eval()
    n = data["t"].shape[0]
    for k in range(0, n, bs):
        sub = {key: v[k:k + bs] for key, v in data.items()}
        h, s, c = model.encode_batch(to_torch(sub, device))
        H.append(h.cpu()); S.append(s.cpu()); C.append(c.cpu())
    model.train()
    return torch.cat(H).numpy(), torch.cat(S).numpy(), torch.cat(C).numpy()


def evaluate(model, data, device="cpu", seed=0):
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    H, S, C = embed_dataset(model, data, device)
    n, W, N = data["t"].shape
    rs = np.random.default_rng(seed)
    star_train = rs.random(n) < 0.7
    cls, logP = data["cls"], data["logP"]
    per = cls != 3
    out = {}
    # token-level features: (star, window, token) flattened
    tok_star = np.repeat(np.arange(n), W * N)
    tok_train = star_train[tok_star]
    feats = {"s (dynamic head)": S.reshape(n * W * N, -1),
             "h (token hidden)": H.reshape(n * W * N, -1)}
    if model.variant.startswith(("lejepa", "mae", "mjepa")):
        feats.pop("s (dynamic head)")  # untrained head in plain LeJEPA
    c_tok = np.repeat(C.reshape(n * W, -1), N, axis=0)  # window's c for each token

    ph = data["phase"].reshape(-1)
    y_ph = np.stack([np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)], 1)
    tok_per = per[tok_star]
    st = data["state"].reshape(-1)
    tok_drw = ~tok_per

    def ridge_r2(X, y, tr, te):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(sc.transform(X[tr]), y[tr])
        return r2_score(y[te], m.predict(sc.transform(X[te])))

    def mlp_r2(X, y, tr, te, max_n=20000):
        idx = np.where(tr)[0]
        idx = rs.choice(idx, min(max_n, len(idx)), replace=False)
        sc = StandardScaler().fit(X[idx])
        m = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=40, early_stopping=True, random_state=0)
        m.fit(sc.transform(X[idx]), y[idx])
        te_idx = np.where(te)[0]
        te_idx = rs.choice(te_idx, min(20000, len(te_idx)), replace=False)
        return r2_score(y[te_idx], m.predict(sc.transform(X[te_idx])))

    for name, X in feats.items():
        tr, te = tok_train & tok_per, ~tok_train & tok_per
        out[f"phase | {name} | linear R2"] = ridge_r2(X, y_ph, tr, te)
        Xc = np.concatenate([X, c_tok], 1)
        out[f"phase | {name} + c | MLP R2"] = mlp_r2(Xc, y_ph, tr, te)
        tr2, te2 = tok_train & tok_drw, ~tok_train & tok_drw
        out[f"DRW state | {name} | linear R2"] = ridge_r2(X, st, tr2, te2)
    # control: does c alone know the phase? (it should not)
    out["phase | c only (control) | MLP R2"] = mlp_r2(c_tok, y_ph, tok_train & tok_per, ~tok_train & tok_per)

    # static probes: from the window code c alone, and from c + mean-pooled h
    # (the MAE has no c head and uses mean-pooled h, so the second is the fair one)
    ca = C[:, 0]
    trp, tep = star_train & per, ~star_train & per
    X2 = np.concatenate([ca, H[:, 0].mean(1)], 1)
    out["log P | c+meanh | linear R2"] = ridge_r2(X2, logP, trp, tep)
    sc2 = StandardScaler().fit(X2[star_train])
    lr2 = LogisticRegression(max_iter=3000).fit(sc2.transform(X2[star_train]), cls[star_train])
    out["class | c+meanh | accuracy"] = lr2.score(sc2.transform(X2[~star_train]), cls[~star_train])
    trp, tep = star_train & per, ~star_train & per
    out["log P | c | linear R2"] = ridge_r2(ca, logP, trp, tep)
    sc = StandardScaler().fit(ca[star_train])
    lr_ = LogisticRegression(max_iter=2000).fit(sc.transform(ca[star_train]), cls[star_train])
    out["class | c | accuracy"] = lr_.score(sc.transform(ca[~star_train]), cls[~star_train])
    return out


@torch.no_grad()
def forecast_eval(model, data, device="cpu", bs=200):
    """World-model test for rotation models: encode window 0, roll every state
    forward with the predictor to the times of each later window, and compare
    with the encoder's own states there. R2 per target window (all dims, and the
    fundamental rotation block alone). 'persist' = copy the source state."""
    if not model.variant.startswith("rot") or "+ham" in model.variant:
        return {}
    model.eval()
    n, W, N = data["t"].shape
    ss_all, pred_all, per_all, win_all = [], [], [], []
    for k in range(0, n, bs):
        sub = {key: v[k:k + bs] for key, v in data.items()}
        bt = to_torch(sub, device)
        _, s, c = model.encode_batch(bt)
        w_ls = window_w_ls(bt)
        B, _, _, ds = s.shape
        src = s[:, 0]                                                      # (B,N,ds)
        for w in range(1, W):
            dt = bt["t"][:, w][:, None, :] - bt["t"][:, 0][:, :, None]     # (B,N_src,N_tgt)
            si = src[:, :, None, :].expand(-1, -1, N, -1).reshape(-1, ds)
            cc = c[:, 0][:, None, None, :].expand(-1, N, N, -1).reshape(-1, c.shape[-1])
            ww = w_ls[:, 0][:, None, None].expand(-1, N, N).reshape(-1)
            p = model.pred(si, cc, dt.reshape(-1), ww).view(B, N, N, ds)
            tgt = s[:, w][:, None, :, :].expand(-1, N, -1, -1)
            ss_all.append(tgt.reshape(-1, ds).cpu()); pred_all.append(p.reshape(-1, ds).cpu())
            per_all.append(src[:, :, None, :].expand(-1, -1, N, -1).reshape(-1, ds).cpu())
            win_all.append(torch.full((B * N * N,), w))
    model.train()
    S, P, Pe, Wn = (torch.cat(x).numpy() for x in (ss_all, pred_all, per_all, win_all))
    def r2(a, b):
        return 1 - ((a - b) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()
    out = {}
    for w in range(1, W):
        m = Wn == w
        out[f"forecast R2 | window {w} | all dims"] = float(r2(S[m], P[m]))
        out[f"forecast R2 | window {w} | block 1"] = float(r2(S[m][:, :2], P[m][:, :2]))
        out[f"forecast R2 | window {w} | persist (copy source)"] = float(r2(S[m], Pe[m]))
    return out
