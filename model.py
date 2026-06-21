
"""ConvNeXt-V2 U-Net that carries its own Jacobian-vector product.

Closed-form JVP, so the whole encoder/decoder tangent is plain differentiable ops:

    net.jvp(x, xd, c) == (net(x, c), J_net(x) @ xd)            (c held fixed)

Two reasons this exists rather than a forward-AD ``dual_level`` pass over the plain net:

1. It is *compilable*: ``torch.compile`` fuses ``jvp`` to ~plain-forward cost, versus the
   ~3-7x of the forward-AD dual.
2. It is *reverse-over-forward correct*: Seems like on certain version of pytorch
   stock ``F.layer_norm``'s forward-AD rule gives a wrong second derivative. (the sign sometimes changes)

Conditioning. By default ``c`` is a fixed per-sample vector (label / time embedding) and
carries no tangent: the adaLN-Zero shift/scale/gate are constants and the spatial tangent
rides through ``* (scale + 1)`` and ``* gate``. ``jvp`` also accepts an optional conditioning
tangent ``cd``; since adaLN is affine in ``c``, the tangent then also rides the modulation in
closed form (``net.jvp(x, xd, c, cd) == (net(x,c), J @ (xd, cd))``), giving the full ``d/dt`` of
a time-conditioned field (e.g. the MeanFlow bootstrap, ``cd = dc/dt``). With ``cond_dim = 0`` the
blocks are plain affine-norm ConvNeXt-V2 and ``c`` is ignored.

Shapes:
    Input:  x, xd [B, in_ch, H, W]   (H, W divisible by 4),   c [B, cond_dim] or None
    Output:        [B, out_ch, H, W]   (and the matching tangent from ``jvp``)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_I2 = 1 / math.sqrt(2.0)
_I2PI = 1 / math.sqrt(2 * math.pi)


def _gelu_prime(x: Tensor) -> Tensor:
    """Derivative of the exact (erf) GELU, matching ``F.gelu``'s default."""
    return 0.5 * (1 + torch.erf(x * _I2)) + x * torch.exp(-0.5 * x * x) * _I2PI


def _silu_prime(x: Tensor) -> Tensor:
    """Derivative of SiLU (the adaLN activation): s + x s (1-s),  s = sigmoid(x)."""
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


def _conv_jvp(conv: nn.Conv2d, x: Tensor, xd: Tensor) -> tuple[Tensor, Tensor]:
    """Primal + tangent of a conv. The weight has no tangent and the bias is an additive
    constant, so the tangent is the same convolution with the bias dropped."""
    y = conv(x)
    yd = F.conv2d(xd, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
    return y, yd


class LayerNorm2d(nn.Module):
    """Channel-dim LayerNorm over an NCHW map, from primitive ops.

    Matches ``backbones.convnext.LayerNorm2d`` numerically (biased variance, eps inside the
    rsqrt, optional affine over channels) and shares its ``weight``/``bias`` shapes, so the
    state dict is interchangeable. Built from mean/var/rsqrt rather than ``F.layer_norm`` so
    the reverse-over-forward second derivative of ``jvp`` is correct.

    JVP:  ẏ = (γ/σ)[ẋ − mean_c ẋ − x̂ · mean_c(x̂ ẋ)],   with x̂ the normalised input.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _normalise(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(x̂, rstd)`` over the channel dim (biased variance, eps in the rsqrt)."""
        d = x - x.mean(1, keepdim=True)
        rstd = torch.rsqrt((d * d).mean(1, keepdim=True) + self.eps)
        return d * rstd, rstd

    def _affine(self, xhat: Tensor) -> Tensor:
        if self.weight is None:
            return xhat
        return torch.addcmul(self.bias.view(1, -1, 1, 1), xhat, self.weight.view(1, -1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        xhat, _ = self._normalise(x)
        return self._affine(xhat)

    def jvp(self, x: Tensor, xd: Tensor) -> tuple[Tensor, Tensor]:
        xhat, rstd = self._normalise(x)
        xhatd = rstd * (xd - xd.mean(1, keepdim=True) - xhat * (xhat * xd).mean(1, keepdim=True))
        if self.weight is None:
            return xhat, xhatd
        w = self.weight.view(1, -1, 1, 1)
        return torch.addcmul(self.bias.view(1, -1, 1, 1), xhat, w), xhatd * w


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt-V2). Identity at init (gamma = beta = 0);
    the spatial L2 energy is accumulated in fp32 so the reduction is exact under bf16."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        gx = torch.norm(x.float(), p=2, dim=(2, 3), keepdim=True)
        nx = (gx / (gx.mean(dim=1, keepdim=True) + 1e-6)).to(x.dtype)
        return self.gamma * (x * nx) + self.beta + x

    def jvp(self, x: Tensor, xd: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        xf, xdf = x.float(), xd.float()
        gx = torch.norm(xf, p=2, dim=(2, 3), keepdim=True)
        mgx = gx.mean(1, keepdim=True) + eps
        nx = gx / mgx
        gxd = (xf * xdf).sum(dim=(2, 3), keepdim=True) / gx.clamp_min(1e-20)
        nxd = (gxd * mgx - gx * gxd.mean(1, keepdim=True)) / (mgx * mgx)
        nxt, nxdt = nx.to(x.dtype), nxd.to(x.dtype)
        y = self.gamma * (x * nxt) + self.beta + x
        yd = self.gamma * (xd * nxt + x * nxdt) + xd
        return y, yd


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-V2 block, optionally adaLN-Zero conditioned (see base backbone for the math).

    With ``cond_dim > 0`` the norm is non-affine and adaLN supplies shift/scale/gate; the
    zero-init gate makes the block the identity at init. In ``jvp`` the conditioning ``c`` is
    fixed, so shift/scale/gate are constants and the tangent rides through ``* (scale + 1)``
    after the norm and ``* gate`` before the residual add.
    """

    def __init__(self, dim: int, mlp_ratio: float = 3.0, cond_dim: int = 0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, affine=cond_dim == 0)
        self.pwconv1 = nn.Conv2d(dim, hidden, 1)
        self.grn = GRN(hidden)
        self.pwconv2 = nn.Conv2d(hidden, dim, 1)
        self.adaln = (nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * dim))
                      if cond_dim else None)

    def _mod(self, c: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.adaln(c)[:, :, None, None].chunk(3, dim=1)   # shift, scale, gate

    def _mod_jvp(self, c: Tensor, cd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Tangents (d shift, d scale, d gate) of the adaLN modulation along a c-tangent ``cd``.

        adaLN is ``Linear(SiLU(c))``, affine in its activation, so the tangent is
        ``W (SiLU'(c) . cd)`` (bias drops); chunked like ``_mod``."""
        d = (_silu_prime(c) * cd) @ self.adaln[1].weight.T               # (B, 3*dim)
        return d[:, :, None, None].chunk(3, dim=1)                       # each (B, dim, 1, 1)

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        h = self.norm(self.dwconv(x))
        if self.adaln is not None:
            shift, scale, gate = self._mod(c)
            h = torch.addcmul(shift, h, scale + 1)
        h = self.pwconv2(self.grn(F.gelu(self.pwconv1(h))))
        if self.adaln is not None:
            h = h * gate
        return x + h

    def jvp(self, x: Tensor, xd: Tensor, c: Tensor | None = None,
            cd: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``(forward(x,c), J@(xd,cd))`` -- spatial tangent ``xd`` and (optional) conditioning
        tangent ``cd``. ``cd=None`` holds ``c`` fixed (the divergence path), identical to before."""
        h, hd = _conv_jvp(self.dwconv, x, xd)
        h, hd = self.norm.jvp(h, hd)
        if self.adaln is not None:
            shift, scale, gate = self._mod(c)
            s1 = scale + 1
            if cd is not None:
                shift_d, scale_d, gate_d = self._mod_jvp(c, cd)
                h, hd = torch.addcmul(shift, h, s1), hd * s1 + h * scale_d + shift_d
            else:
                h, hd = torch.addcmul(shift, h, s1), hd * s1
        h, hd = _conv_jvp(self.pwconv1, h, hd)
        h, hd = F.gelu(h), _gelu_prime(h) * hd
        h, hd = self.grn.jvp(h, hd)
        h, hd = _conv_jvp(self.pwconv2, h, hd)
        if self.adaln is not None:
            if cd is not None:
                hd = hd * gate + h * gate_d          # product rule before overwriting h
            else:
                hd = hd * gate
            h = h * gate
        return x + h, xd + hd


class Downsample(nn.Module):
    """(B, in_ch, H, W) -> (B, out_ch, H/2, W/2): 3x3 conv to out_ch/4, then pixel-unshuffle."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.pixel_unshuffle(self.conv(x), 2)

    def jvp(self, x: Tensor, xd: Tensor) -> tuple[Tensor, Tensor]:
        h, hd = _conv_jvp(self.conv, x, xd)
        return F.pixel_unshuffle(h, 2), F.pixel_unshuffle(hd, 2)


class Upsample(nn.Module):
    """(B, in_ch, H, W) -> (B, out_ch, 2H, 2W): 3x3 conv to 4*out_ch, then pixel-shuffle."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.pixel_shuffle(self.conv(x), 2)

    def jvp(self, x: Tensor, xd: Tensor) -> tuple[Tensor, Tensor]:
        h, hd = _conv_jvp(self.conv, x, xd)
        return F.pixel_shuffle(h, 2), F.pixel_shuffle(hd, 2)


class ConvNeXtUNetJVP(nn.Module):
    """Fully convolutional U-Net with a built-in Jacobian-vector product.

    Constructor, parameters, and primal ``forward`` ``jvp`` adds the
    forward-mode tangent in closed form. ``cond_dim > 0`` threads a fixed per-sample vector
    through every block's adaLN-Zero and the final layer; ``zero_head=True`` zero-inits the
    head so the net outputs 0 at init.
    """

    def __init__(self, in_ch: int, out_ch: int, *, width: int = 128,
                 depths: tuple[int, int, int, int, int] = (2, 4, 8, 4, 2),
                 mlp_ratio: float = 3.0, cond_dim: int = 0, zero_head: bool = True):
        super().__init__()
        w1, w2, w3 = width, 2 * width, 4 * width
        self.cond_dim = cond_dim

        def blocks(dim: int, n: int) -> nn.ModuleList:
            return nn.ModuleList(ConvNeXtBlock(dim, mlp_ratio, cond_dim) for _ in range(n))

        self.stem = nn.Conv2d(in_ch, w1, kernel_size=3, padding=1)
        self.enc1 = blocks(w1, depths[0])
        self.down1 = Downsample(w1, w2)
        self.enc2 = blocks(w2, depths[1])
        self.down2 = Downsample(w2, w3)
        self.latent = blocks(w3, depths[2])
        self.up2 = Upsample(w3, w2)
        self.reduce2 = nn.Conv2d(2 * w2, w2, kernel_size=1)
        self.dec2 = blocks(w2, depths[3])
        self.up1 = Upsample(w2, w1)
        self.reduce1 = nn.Conv2d(2 * w1, w1, kernel_size=1)
        self.dec1 = blocks(w1, depths[4])

        self.out_conv = nn.Conv2d(w1, w1, kernel_size=3, padding=1)
        self.out_norm = LayerNorm2d(w1, affine=cond_dim == 0)
        self.out_adaln = (nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * w1))
                          if cond_dim else None)
        self.head = nn.Conv2d(w1, out_ch, kernel_size=3, padding=1)
        self.reset_parameters(zero_head)

    def reset_parameters(self, zero_head: bool) -> None:
        """Xavier-uniform convs/linears with zero bias; zero every adaLN projection
        (identity modulation at init); zero the head when ``zero_head``."""
        def _init(m: nn.Module) -> None:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)
        for m in self.modules():
            if isinstance(m, ConvNeXtBlock) and m.adaln is not None:
                nn.init.zeros_(m.adaln[-1].weight)
                nn.init.zeros_(m.adaln[-1].bias)
        if self.out_adaln is not None:
            nn.init.zeros_(self.out_adaln[-1].weight)
            nn.init.zeros_(self.out_adaln[-1].bias)
        if zero_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, c: Tensor | None = None,
                return_latent: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Primal pass. ``return_latent=True`` also returns the bottleneck map
        ``(B, 4*width, H/4, W/4)`` -- the representation tap for SSL objectives."""
        h1 = self.stem(x)
        for blk in self.enc1:
            h1 = blk(h1, c)
        h2 = self.down1(h1)
        for blk in self.enc2:
            h2 = blk(h2, c)
        h3 = self.down2(h2)
        for blk in self.latent:
            h3 = blk(h3, c)
        d2 = self.reduce2(torch.cat([self.up2(h3), h2], dim=1))
        for blk in self.dec2:
            d2 = blk(d2, c)
        d1 = self.reduce1(torch.cat([self.up1(d2), h1], dim=1))
        for blk in self.dec1:
            d1 = blk(d1, c)
        out = self.out_norm(self.out_conv(d1))
        if self.out_adaln is not None:
            shift, scale = self.out_adaln(c)[:, :, None, None].chunk(2, dim=1)
            out = torch.addcmul(shift, out, scale + 1)
        out = self.head(out)
        return (out, h3) if return_latent else out

    def jvp(self, x: Tensor, xd: Tensor, c: Tensor | None = None,
            cd: Tensor | None = None) -> tuple[Tensor, Tensor]:
        h1, h1d = _conv_jvp(self.stem, x, xd)
        for blk in self.enc1:
            h1, h1d = blk.jvp(h1, h1d, c, cd)
        h2, h2d = self.down1.jvp(h1, h1d)
        for blk in self.enc2:
            h2, h2d = blk.jvp(h2, h2d, c, cd)
        h3, h3d = self.down2.jvp(h2, h2d)
        for blk in self.latent:
            h3, h3d = blk.jvp(h3, h3d, c, cd)
        u, ud = self.up2.jvp(h3, h3d)
        d2, d2d = _conv_jvp(self.reduce2, torch.cat([u, h2], 1), torch.cat([ud, h2d], 1))
        for blk in self.dec2:
            d2, d2d = blk.jvp(d2, d2d, c, cd)
        u, ud = self.up1.jvp(d2, d2d)
        d1, d1d = _conv_jvp(self.reduce1, torch.cat([u, h1], 1), torch.cat([ud, h1d], 1))
        for blk in self.dec1:
            d1, d1d = blk.jvp(d1, d1d, c, cd)
        o, od = _conv_jvp(self.out_conv, d1, d1d)
        o, od = self.out_norm.jvp(o, od)
        if self.out_adaln is not None:
            shift, scale = self.out_adaln(c)[:, :, None, None].chunk(2, dim=1)
            s1 = scale + 1
            if cd is not None:
                d = (_silu_prime(c) * cd) @ self.out_adaln[1].weight.T
                shift_d, scale_d = d[:, :, None, None].chunk(2, dim=1)
                o, od = torch.addcmul(shift, o, s1), od * s1 + o * scale_d + shift_d
            else:
                o, od = torch.addcmul(shift, o, s1), od * s1
        o, od = _conv_jvp(self.head, o, od)
        return o, od
