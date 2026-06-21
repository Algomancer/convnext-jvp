# Closed-form JVP ConvNeXt-V2 U-Net 

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
```
Shapes:
    Input:  x, xd [B, in_ch, H, W]   (H, W divisible by 4),   c [B, cond_dim] or None
    Output:        [B, out_ch, H, W]   (and the matching tangent from ``jvp``)
```

```
@misc{algomancer2025,
  author = {@algomancer},
  title  = {JVP ConvNeXt-V2 U-Net Closed Form},
  year   = {2026}
}
```
