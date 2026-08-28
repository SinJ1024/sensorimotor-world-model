"""
Model components for latent world models.

Inherited from Le-WM (https://github.com/lucas-maes/le-wm) with the addition of InverseModel.
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=ConditionalBlock,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)
        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)
        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, norm_fn=nn.LayerNorm, act_fn=nn.GELU):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class InverseModel(nn.Module):
    """Predicts action from consecutive latent embeddings: (z_t, z_{t+1}) -> a_t"""

    def __init__(self, embed_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z_t, z_tp1):
        """
        z_t:   (B, D) or (B, T, D)
        z_tp1: (B, D) or (B, T, D)
        Returns: predicted action, same leading dims
        """
        return self.net(torch.cat([z_t, z_tp1], dim=-1))


class _ResBlock(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, mult * dim), nn.GELU(),
            nn.Linear(mult * dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class _PolicyTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, mlp_dim=None):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim, mlp_dim or 4 * dim)

    def forward(self, x):
        x = x + self.attn(x, causal=True)
        return x + self.ff(x)


class _MambaBlock(nn.Module):
    """Minimal pure-PyTorch selective SSM (Mamba) block — no custom CUDA kernel,
    so it runs on any GPU; the scan is sequential (O(L), fine for short windows)."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * dim
        self.d_state = d_state
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, 2 * d_state + 1)
        self.dt_proj = nn.Linear(1, d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, dim)

    def forward(self, x):
        B, L, _ = x.shape
        res = x
        xz = self.in_proj(self.norm(x))
        xin, z = xz.chunk(2, dim=-1)                                    # (B,L,d_inner)
        xc = self.conv(xin.transpose(1, 2))[..., :L].transpose(1, 2)    # causal conv
        xc = F.silu(xc)
        dbl = self.x_proj(xc)
        dt, Bm, Cm = dbl[..., :1], dbl[..., 1:1 + self.d_state], dbl[..., 1 + self.d_state:]
        dt = F.softplus(self.dt_proj(dt))                              # (B,L,d_inner)
        A = -torch.exp(self.A_log)                                     # (d_inner,d_state)
        h = torch.zeros(B, xc.shape[-1], self.d_state, device=x.device, dtype=xc.dtype)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)                 # (B,d_inner,d_state)
            dBx = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1) * xc[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))            # (B,d_inner)
        y = torch.stack(ys, dim=1) + xc * self.D
        return res + self.out_proj(y * F.silu(z))


class PolicyModel(nn.Module):
    """Regularizer head: predict the next ``num_future`` actions a_{t+1..t+k}
    from a window of ``context`` latents ending at z_{t+1} (+ optional past
    actions within the window). ``arch`` selects the sequence model.

    arch = mlp | resmlp | gru | transformer | mamba
      mlp/resmlp : flatten the whole window -> MLP (context=2,k=1 == original head)
      gru/transformer/mamba : per-timestep token -> sequence model -> last token

    Predicting the *future* action only carries anti-collapse signal when the
    data's behaviour policy is state-conditioned. A stronger head predicts better
    but can also extract the answer from a poor z (less encoder pressure), so
    always track both policy_loss (down) AND effective_rank (up).
    """

    def __init__(self, embed_dim, action_dim, hidden_dim=256, use_action=True,
                 num_future=1, context=2, arch="mlp", depth=2, heads=4):
        super().__init__()
        self.use_action = bool(use_action)
        self.num_future = int(num_future)
        self.action_dim = int(action_dim)
        self.context = int(context)
        self.arch = str(arch).lower()
        out_dim = self.num_future * action_dim
        act_in = action_dim if self.use_action else 0

        if self.arch in ("mlp", "resmlp"):
            flat_in = self.context * embed_dim + (self.context - 1) * act_in
            self.proj = nn.Linear(flat_in, hidden_dim)
            if self.arch == "mlp":
                self.body = nn.Sequential(
                    nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
                )
            else:
                self.body = nn.Sequential(*[_ResBlock(hidden_dim) for _ in range(depth)])
            self.out = nn.Linear(hidden_dim, out_dim)
        elif self.arch in ("gru", "transformer", "mamba"):
            self.token = nn.Linear(embed_dim + act_in, hidden_dim)
            if self.arch == "gru":
                self.seq = nn.GRU(hidden_dim, hidden_dim, num_layers=depth, batch_first=True)
            elif self.arch == "transformer":
                self.pos = nn.Parameter(torch.randn(1, self.context, hidden_dim) * 0.02)
                self.seq = nn.ModuleList(
                    [_PolicyTransformerBlock(hidden_dim, heads=heads) for _ in range(depth)]
                )
            else:  # mamba
                self.seq = nn.ModuleList([_MambaBlock(hidden_dim) for _ in range(depth)])
            self.out = nn.Linear(hidden_dim, out_dim)
        else:
            raise ValueError(f"Unknown policy arch: {self.arch!r}")

    def forward(self, z_window, a_window=None):
        """
        z_window: (B, L, D) — L=context latents ending at z_{t+1}
        a_window: (B, L-1, A) — actions within the window (required iff use_action)
        Returns: (B, num_future, action_dim)
        """
        B, L = z_window.shape[:2]
        if self.arch in ("mlp", "resmlp"):
            parts = [z_window.reshape(B, -1)]
            if self.use_action:
                parts.append(a_window.reshape(B, -1))
            h = self.proj(torch.cat(parts, dim=-1))
            h = self.body(h)
        else:
            tok_in = z_window
            if self.use_action:
                a_pad = F.pad(a_window, (0, 0, 0, 1))    # zero for the last latent
                tok_in = torch.cat([z_window, a_pad], dim=-1)
            x = self.token(tok_in)                       # (B, L, H)
            if self.arch == "gru":
                x, _ = self.seq(x)
            elif self.arch == "transformer":
                x = x + self.pos[:, :L]
                for blk in self.seq:
                    x = blk(x)
            else:
                for blk in self.seq:
                    x = blk(x)
            h = x[:, -1]                                  # last-token summary
        return self.out(h).reshape(B, self.num_future, self.action_dim)


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()