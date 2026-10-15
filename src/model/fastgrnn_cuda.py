"""Persistent CUDA kernels for the sigmoid/tanh FastGRNN recurrence.

Input projection stays in PyTorch. Each program owns a group and a batch tile,
keeps its recurrent state on the GPU, and loops over time without host launches.
Backward accumulates weight gradients in private tiles; no atomics are needed.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["T"])
def forward_kernel(
    X, H0, U, BG, BU, ZETA, NU, GAMMA, DRIFT, Y, Z, C,
    B: tl.constexpr, H: tl.constexpr, T, G: tl.constexpr,
    BB: tl.constexpr, HH: tl.constexpr, REVERSE: tl.constexpr, SAVE: tl.constexpr,
):
    # X/Y/Z/C: [time, group, batch, hidden]; U: [group, hidden, hidden].
    # T is a runtime scalar: variable audio lengths do not trigger compilation.
    g = tl.program_id(0)
    tile = tl.program_id(1)
    b = tile * BB + tl.arange(0, BB)
    k = tl.arange(0, HH)
    valid = (b[:, None] < B) & (k[None, :] < H)
    valid_u = (k[:, None] < H) & (k[None, :] < H)
    h = tl.load(H0 + (g * B + b[:, None]) * H + k[None, :], valid, 0)
    u = tl.load(U + g * H * H + k[:, None] * H + k[None, :], valid_u, 0)
    bg = tl.load(BG + g * H + k, k < H, 0)
    bu = tl.load(BU + g * H + k, k < H, 0)
    zeta = tl.load(ZETA + g)
    nu = tl.load(NU + g)
    gamma = tl.load(GAMMA + g)
    drift = tl.load(DRIFT + g)
    for step in range(T):
        t = T - 1 - step if REVERSE else step
        offset = ((t * G + g) * B + b[:, None]) * H + k[None, :]
        wx = tl.load(X + offset, valid, 0)
        pre = wx + tl.dot(h, u, input_precision="tf32x3")
        z = 1.0 / (1.0 + tl.exp(-(pre + bg[None, :])))
        c = libdevice.tanh(pre + bu[None, :])
        h = gamma * (z * h + (zeta * (1 - z) + nu) * c) + drift
        h = tl.where(k[None, :] < H, h, 0)
        tl.store(Y + offset, h, valid)
        if SAVE:
            tl.store(Z + offset, z, valid)
            tl.store(C + offset, c, valid)


@triton.jit(do_not_specialize=["T"])
def wide_forward_kernel(
    X, H0, U, BG, BU, ZETA, NU, GAMMA, DRIFT, Y, Z, C,
    B: tl.constexpr, H: tl.constexpr, T, G: tl.constexpr,
    BB: tl.constexpr, REVERSE: tl.constexpr, SAVE: tl.constexpr,
):
    # Splitting U into four 64x64 blocks keeps tf32x3 within SM86/89 shared
    # memory limits. A single padded 128x128 dot needs over 150 KiB.
    g = tl.program_id(0)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    k0 = tl.arange(0, 64)
    k1 = k0 + 64
    valid0 = (b[:, None] < B) & (k0[None, :] < H)
    valid1 = (b[:, None] < B) & (k1[None, :] < H)
    state0 = tl.load(H0 + (g * B + b[:, None]) * H + k0[None, :], valid0, 0)
    state1 = tl.load(H0 + (g * B + b[:, None]) * H + k1[None, :], valid1, 0)
    base_u = U + g * H * H
    u00 = tl.load(
        base_u + k0[:, None] * H + k0[None, :],
        (k0[:, None] < H) & (k0[None, :] < H), 0,
    )
    u10 = tl.load(
        base_u + k1[:, None] * H + k0[None, :],
        (k1[:, None] < H) & (k0[None, :] < H), 0,
    )
    u01 = tl.load(
        base_u + k0[:, None] * H + k1[None, :],
        (k0[:, None] < H) & (k1[None, :] < H), 0,
    )
    u11 = tl.load(
        base_u + k1[:, None] * H + k1[None, :],
        (k1[:, None] < H) & (k1[None, :] < H), 0,
    )
    bg0 = tl.load(BG + g * H + k0, k0 < H, 0)
    bg1 = tl.load(BG + g * H + k1, k1 < H, 0)
    bu0 = tl.load(BU + g * H + k0, k0 < H, 0)
    bu1 = tl.load(BU + g * H + k1, k1 < H, 0)
    zeta = tl.load(ZETA + g)
    nu = tl.load(NU + g)
    gamma = tl.load(GAMMA + g)
    drift = tl.load(DRIFT + g)
    for step in range(T):
        t = T - 1 - step if REVERSE else step
        offset = ((t * G + g) * B + b[:, None]) * H
        wx0 = tl.load(X + offset + k0[None, :], valid0, 0)
        wx1 = tl.load(X + offset + k1[None, :], valid1, 0)
        recurrent0 = tl.dot(state0, u00, input_precision="tf32x3") + tl.dot(
            state1, u10, input_precision="tf32x3"
        )
        recurrent1 = tl.dot(state0, u01, input_precision="tf32x3") + tl.dot(
            state1, u11, input_precision="tf32x3"
        )
        pre0 = wx0 + recurrent0
        pre1 = wx1 + recurrent1
        z0 = 1.0 / (1.0 + tl.exp(-(pre0 + bg0[None, :])))
        z1 = 1.0 / (1.0 + tl.exp(-(pre1 + bg1[None, :])))
        c0 = libdevice.tanh(pre0 + bu0[None, :])
        c1 = libdevice.tanh(pre1 + bu1[None, :])
        state0 = gamma * (z0 * state0 + (zeta * (1 - z0) + nu) * c0) + drift
        state1 = gamma * (z1 * state1 + (zeta * (1 - z1) + nu) * c1) + drift
        state0 = tl.where(k0[None, :] < H, state0, 0)
        state1 = tl.where(k1[None, :] < H, state1, 0)
        tl.store(Y + offset + k0[None, :], state0, valid0)
        tl.store(Y + offset + k1[None, :], state1, valid1)
        if SAVE:
            tl.store(Z + offset + k0[None, :], z0, valid0)
            tl.store(Z + offset + k1[None, :], z1, valid1)
            tl.store(C + offset + k0[None, :], c0, valid0)
            tl.store(C + offset + k1[None, :], c1, valid1)


@triton.jit(do_not_specialize=["T"])
def backward_kernel(
    DY, H0, U, ZETA, NU, GAMMA, Y, Z, C,
    DX, DH0, DU, DBG, DBU, DZETA, DNU, DGAMMA, DDRIFT,
    B: tl.constexpr, H: tl.constexpr, T, G: tl.constexpr,
    NT: tl.constexpr, BB: tl.constexpr, HH: tl.constexpr, REVERSE: tl.constexpr,
):
    g = tl.program_id(0)
    tile = tl.program_id(1)
    b = tile * BB + tl.arange(0, BB)
    k = tl.arange(0, HH)
    valid = (b[:, None] < B) & (k[None, :] < H)
    valid_u = (k[:, None] < H) & (k[None, :] < H)
    ut = tl.load(U + g * H * H + k[None, :] * H + k[:, None], valid_u, 0)
    zeta = tl.load(ZETA + g)
    nu = tl.load(NU + g)
    gamma = tl.load(GAMMA + g)
    dh = tl.full((BB, HH), 0, tl.float32)
    du = tl.full((HH, HH), 0, tl.float32)
    dbg = tl.full((BB, HH), 0, tl.float32)
    dbu = tl.full((BB, HH), 0, tl.float32)
    dzeta = tl.full((BB, HH), 0, tl.float32)
    dnu = tl.full((BB, HH), 0, tl.float32)
    dgamma = tl.full((BB, HH), 0, tl.float32)
    ddrift = tl.full((BB, HH), 0, tl.float32)
    for step in range(T):
        t = step if REVERSE else T - 1 - step
        offset = ((t * G + g) * B + b[:, None]) * H + k[None, :]
        dy = tl.load(DY + offset, valid, 0)
        z = tl.load(Z + offset, valid, 0)
        c = tl.load(C + offset, valid, 0)
        previous = t + 1 if REVERSE else t - 1
        if previous >= 0 and previous < T:
            hp = tl.load(Y + ((previous * G + g) * B + b[:, None]) * H + k[None, :], valid, 0)
        else:
            hp = tl.load(H0 + (g * B + b[:, None]) * H + k[None, :], valid, 0)
        # h = gamma * (z * hp + (zeta * (1-z) + nu) * c) + drift.
        # Both nonlinearities share preactivation, so their derivatives add.
        dy = tl.where(valid, dy + dh, 0)
        dq = gamma * dy
        zg = dq * (hp - zeta * c) * z * (1 - z)
        cg = dq * (zeta * (1 - z) + nu) * (1 - c * c)
        dp = zg + cg
        du += tl.dot(tl.trans(hp), dp, input_precision="ieee" if HH > 64 else "tf32x3")
        dh = dq * z + tl.dot(dp, ut, input_precision="ieee" if HH > 64 else "tf32x3")
        dh = tl.where(valid, dh, 0)
        dbg += zg
        dbu += cg
        dzeta += dq * (1 - z) * c
        dnu += dq * c
        dgamma += dy * (z * hp + (zeta * (1 - z) + nu) * c)
        ddrift += dy
        tl.store(DX + offset, dp, valid)
    tl.store(DH0 + (g * B + b[:, None]) * H + k[None, :], dh, valid)
    tl.store(DU + ((g * NT + tile) * H + k[:, None]) * H + k[None, :], du, valid_u)
    tl.store(DBG + (g * NT + tile) * H + k, tl.sum(dbg, 0), k < H)
    tl.store(DBU + (g * NT + tile) * H + k, tl.sum(dbu, 0), k < H)
    tl.store(DZETA + g * NT + tile, tl.sum(tl.sum(dzeta, 0), 0))
    tl.store(DNU + g * NT + tile, tl.sum(tl.sum(dnu, 0), 0))
    tl.store(DGAMMA + g * NT + tile, tl.sum(tl.sum(dgamma, 0), 0))
    tl.store(DDRIFT + g * NT + tile, tl.sum(tl.sum(ddrift, 0), 0))


class _PersistentGRNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h0, u, bg, bu, zeta, nu, gamma, drift, reverse, save):
        t, g, b, h = x.shape
        y = torch.empty_like(x)
        z = torch.empty_like(x) if save else y
        c = torch.empty_like(x) if save else y
        grid = (g, triton.cdiv(b, 16))
        if h > 64:
            wide_forward_kernel[grid](
                x, h0, u, bg, bu, zeta, nu, gamma, drift, y, z, c,
                b, h, t, g, 16, reverse, save,
                num_warps=4, num_stages=1, enable_fp_fusion=False,
            )
        else:
            forward_kernel[grid](
                x, h0, u, bg, bu, zeta, nu, gamma, drift, y, z, c,
                b, h, t, g, 16, max(16, triton.next_power_of_2(h)), reverse, save,
                num_warps=4, num_stages=1, enable_fp_fusion=False,
            )
        if save:
            ctx.save_for_backward(h0, u, zeta, nu, gamma, y, z, c)
            ctx.reverse = reverse
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        h0, u, zeta, nu, gamma, y, z, c = ctx.saved_tensors
        t, g, b, h = y.shape
        nt = triton.cdiv(b, 16)
        dx = torch.empty_like(y)
        dh0 = torch.empty_like(h0)
        du = u.new_empty(g, nt, h, h)
        dbg = u.new_empty(g, nt, h)
        dbu = torch.empty_like(dbg)
        ds = u.new_empty(4, g, nt)
        backward_kernel[(g, nt)](
            dy.contiguous(), h0, u, zeta, nu, gamma, y, z, c,
            dx, dh0, du, dbg, dbu, ds[0], ds[1], ds[2], ds[3],
            b, h, t, g, nt, 16, max(16, triton.next_power_of_2(h)), ctx.reverse,
            num_warps=8 if h > 64 else 4, num_stages=1, enable_fp_fusion=False,
        )
        ds = ds.sum(-1)
        return dx, dh0, du.sum(1), dbg.sum(1), dbu.sum(1), ds[0], ds[1], ds[2], ds[3], None, None


def persistent_grnn(x, h0, params, reverse=False):
    """Return [T, G, B, H] states; parameters are already sigmoid/clamp transformed.

    Materializing a low-rank U once preserves its parameterization and lets
    autograd propagate the dense gradient to both factors outside the kernel.
    tf32x3 uses three Tensor Core products to retain float32 accuracy, rather
    than the reduced precision of a single TF32 product.
    """
    g = h0.shape[0]
    u = params["u"].contiguous() if "u" in params else torch.bmm(params["u1"], params["u2"])
    bg = params["bias_gate"].reshape(g, -1)
    bu = params["bias_update"].reshape(g, -1)
    zeta = params["zeta"].reshape(g)
    nu = params["nu"].reshape(g)
    gamma = params["gamma"].reshape(g)
    drift = ((1 - params["gamma"]) * params["lambd"]).reshape(g)
    save = torch.is_grad_enabled() and any(
        p.requires_grad for p in (x, h0, u, bg, bu, zeta, nu, gamma, drift)
    )
    return _PersistentGRNN.apply(x, h0, u, bg, bu, zeta, nu, gamma, drift, reverse, save)
