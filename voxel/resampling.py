"""
Tensor-level sampling and resampling. `sample_tensor` fixes the local grid sampling
convention in one place, and `resample_tensor` resamples grid-free: a Triton kernel maps
each output voxel through a voxel-to-voxel matrix and interpolates the source in a single
pass, so no coordinate grid is ever materialized.
"""

from __future__ import annotations

import os
import torch
import voxel as vx

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

_INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'


def sample_tensor(
    tensor: torch.Tensor,
    local: torch.Tensor,
    mode: str = 'linear',
    padding_mode: str = 'zeros') -> torch.Tensor:
    """
    Sample a $(C, W, H, D)$ tensor at points given in the flipped [-1, 1] local
    coordinates of torch grid sampling (`align_corners=False`). This is the one place
    that fixes voxel's sampling convention. Floating-point inputs on one device are
    passed to the sampler without copies or host syncs, so the function suits hot
    optimization loops.

    Args:
        tensor (Tensor): Source features of shape $(C, W, H, D)$.
        local (Tensor): Local sampling coordinates of shape $(..., 3)$, e.g. from
            `AcquisitionGeometry.grid('local')` or `local_coordinate_transform`.
        mode (str, optional): Interpolation mode, 'linear' or 'nearest'.
        padding_mode (str, optional): Out-of-bounds handling, 'zeros', 'border'
            or 'reflection'.

    Returns:
        Tensor: Sampled floating-point features of shape $(C, ...)$, keeping the
            leading shape of the points.
    """
    if not tensor.is_floating_point():
        tensor = tensor.float()

    # up to 3 leading point dims are sampled in their natural shape, avoiding a
    # flattening round trip; higher-dim point sets fall back to flat
    lead = local.shape[:-1]
    points = local.reshape(-1, 3) if len(lead) > 3 else local
    grid = points.reshape(1, *points.shape[:-1], *([1] * (3 - (points.ndim - 1))), 3)

    sampled = torch.nn.functional.grid_sample(
        tensor.unsqueeze(0),
        grid.to(tensor.dtype),
        mode=('bilinear' if mode == 'linear' else mode),
        padding_mode=padding_mode,
        align_corners=False)
    return sampled.squeeze(0).reshape(tensor.shape[0], *lead)


def resample_tensor(
    tensor: torch.Tensor,
    baseshape: torch.Size,
    matrix: vx.AffineMatrix,
    mode: str = 'linear',
    padding_mode: str = 'zeros',
    fill: float = 0) -> torch.Tensor:
    """
    Resample a $(C, W, H, D)$ tensor onto a grid of spatial shape `baseshape`.

    Args:
        tensor (Tensor): Source features of shape $(C, W, H, D)$.
        baseshape (Size): Spatial shape of the output grid.
        matrix (AffineMatrix): Voxel-to-voxel transform mapping output voxel
            coordinates to source voxel coordinates.
        mode (str, optional): Interpolation mode, 'linear' or 'nearest'.
        padding_mode (str, optional): Out-of-bounds handling, 'zeros', 'border',
            'fill', or 'reflection' (torch fallback only).
        fill (float, optional): Value used by the 'fill' padding mode.

    Returns:
        Tensor: Resampled float32 features of shape $(C, *baseshape)$.
    """
    if triton is not None and (tensor.is_cuda or _INTERPRET) and mode in ('linear', 'nearest') \
       and padding_mode in ('zeros', 'border', 'fill'):
        return _resample_triton(tensor, baseshape, matrix, mode, padding_mode, fill)
    return _resample_torch(tensor, baseshape, matrix, mode, padding_mode, fill)


def _resample_torch(tensor, baseshape, matrix, mode, padding_mode, fill):
    grid = vx.volume.volume_grid(baseshape, transform=matrix, localshape=tensor.shape[1:],
                                 device=tensor.device)

    fill_out_of_bounds = padding_mode == 'fill'
    if fill_out_of_bounds:
        padding_mode = 'border'

    resampled = sample_tensor(tensor.float(), grid, mode=mode, padding_mode=padding_mode)

    if fill_out_of_bounds:
        out_of_bounds = (grid < -1).any(-1) | (grid > 1).any(-1)
        if isinstance(fill, torch.Tensor):
            fill = fill.type(resampled.dtype)
        resampled[:, out_of_bounds] = fill

    return resampled


def _resample_triton(tensor, baseshape, matrix, mode, padding_mode, fill):
    if tensor.dtype == torch.bool:
        tensor = tensor.view(torch.uint8)
    elif tensor.dtype == torch.bfloat16 and _INTERPRET:
        tensor = tensor.float()
    matrix = matrix.tensor[:3].to(device=tensor.device, dtype=torch.float32).contiguous()
    baseshape = tuple(int(s) for s in baseshape)
    return _Resample.apply(tensor.contiguous(), matrix, baseshape, mode, padding_mode, float(fill))


_BLOCK = 256
_WARPS = 4
_MODES = {'linear': 0, 'nearest': 1}
_PADS = {'zeros': 0, 'border': 1, 'fill': 2}


class _Resample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, source, matrix, baseshape, mode, padding_mode, fill):
        numel = baseshape[0] * baseshape[1] * baseshape[2]
        if numel >= 2 ** 31:
            raise ValueError('resampling target exceeds 2^31 voxels')
        out = torch.empty((source.shape[0], *baseshape), dtype=torch.float32, device=source.device)
        _resample_forward[(triton.cdiv(numel, _BLOCK),)](
            source, matrix, out, *source.shape, *baseshape, numel, fill,
            MODE=_MODES[mode], PAD=_PADS[padding_mode], BLOCK=_BLOCK, num_warps=_WARPS)
        ctx.save_for_backward(source, matrix)
        ctx.config = (mode, padding_mode, fill)
        return out

    @staticmethod
    def backward(ctx, grad):
        source, matrix = ctx.saved_tensors
        mode, padding_mode, fill = ctx.config
        need_source, need_matrix = ctx.needs_input_grad[:2]
        need_matrix = need_matrix and mode == 'linear'
        if not (need_source or need_matrix):
            return None, None, None, None, None, None

        grad = grad.contiguous()
        numel = grad[0].numel()
        programs = triton.cdiv(numel, _BLOCK)
        grad_source = torch.zeros(source.shape, dtype=torch.float32, device=source.device) \
                      if need_source else grad
        partials = torch.empty((programs, 12), dtype=torch.float32, device=source.device) \
                   if need_matrix else grad
        _resample_backward[(programs,)](
            source, matrix, grad, grad_source, partials, *source.shape, *grad.shape[1:], numel, fill,
            MODE=_MODES[mode], PAD=_PADS[padding_mode], NEED_SGRAD=need_source,
            NEED_MGRAD=need_matrix, BLOCK=_BLOCK, num_warps=_WARPS)

        grad_source = grad_source.to(source.dtype) if need_source else None
        grad_matrix = partials.sum(0).view(3, 4) if need_matrix else None
        return grad_source, grad_matrix, None, None, None, None


if triton is not None:

    _SIZES = ['C', 'W', 'H', 'D', 'OW', 'OH', 'OD', 'numel']

    @triton.jit
    def _coordinates(m_ptr, i, OH, OD):
        # output voxel index (z fastest) and its float source coordinates
        z = i % OD
        y = (i // OD) % OH
        x = i // (OD * OH)
        xf = x.to(tl.float32)
        yf = y.to(tl.float32)
        zf = z.to(tl.float32)
        sx = tl.load(m_ptr + 0) * xf + tl.load(m_ptr + 1) * yf + tl.load(m_ptr + 2) * zf + tl.load(m_ptr + 3)
        sy = tl.load(m_ptr + 4) * xf + tl.load(m_ptr + 5) * yf + tl.load(m_ptr + 6) * zf + tl.load(m_ptr + 7)
        sz = tl.load(m_ptr + 8) * xf + tl.load(m_ptr + 9) * yf + tl.load(m_ptr + 10) * zf + tl.load(m_ptr + 11)
        return xf, yf, zf, sx, sy, sz

    @triton.jit
    def _clamp(s, N):
        # grid_sample border clipping to [0, N - 1]
        return tl.minimum(tl.maximum(s, 0.0), (N - 1).to(tl.float32))

    @triton.jit
    def _round(s):
        # round half to even, as grid_sample's nearbyint
        r = tl.floor(s)
        f = s - r
        odd = (r.to(tl.int32) & 1) != 0
        return tl.where((f > 0.5) | ((f == 0.5) & odd), r + 1, r)

    @triton.jit
    def _index(f, N):
        # float grid index to a clamped int64 index (bounds are masked separately)
        return tl.minimum(tl.maximum(f, -2.0), (N + 1).to(tl.float32)).to(tl.int64)

    @triton.jit(do_not_specialize=_SIZES)
    def _resample_forward(src_ptr, m_ptr, out_ptr, C, W, H, D, OW, OH, OD, numel, fill,
                          MODE: tl.constexpr, PAD: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < numel
        xf, yf, zf, sx, sy, sz = _coordinates(m_ptr, i, OH, OD)
        if PAD == 2:
            oob = (sx < -0.5) | (sx > W - 0.5) | (sy < -0.5) | (sy > H - 0.5) | (sz < -0.5) | (sz > D - 0.5)
        if PAD != 0:
            sx, sy, sz = _clamp(sx, W), _clamp(sy, H), _clamp(sz, D)
        WHD = W.to(tl.int64) * H * D
        HD = H.to(tl.int64) * D
        numel64 = numel.to(tl.int64)

        if MODE == 0:
            fx, fy, fz = tl.floor(sx), tl.floor(sy), tl.floor(sz)
            x0, y0, z0 = _index(fx, W), _index(fy, H), _index(fz, D)
            # weights in grid_sample's form and multiplication order (innermost axis first)
            wx0, wx1 = (fx + 1) - sx, sx - fx
            wy0, wy1 = (fy + 1) - sy, sy - fy
            wz0, wz1 = (fz + 1) - sz, sz - fz
            bx0, bx1 = valid & (x0 >= 0) & (x0 < W), valid & (x0 + 1 >= 0) & (x0 + 1 < W)
            by0, by1 = (y0 >= 0) & (y0 < H), (y0 + 1 >= 0) & (y0 + 1 < H)
            bz0, bz1 = (z0 >= 0) & (z0 < D), (z0 + 1 >= 0) & (z0 + 1 < D)
            off = (x0 * H + y0) * D + z0
            for c in range(C):
                ptr = src_ptr + c * WHD
                acc = tl.zeros([BLOCK], dtype=tl.float32)
                m = bx0 & by0 & bz0
                acc += tl.where(m, (wz0 * wy0) * wx0 * tl.load(ptr + off, mask=m, other=0).to(tl.float32), 0.0)
                m = bx0 & by0 & bz1
                acc += tl.where(m, (wz1 * wy0) * wx0 * tl.load(ptr + off + 1, mask=m, other=0).to(tl.float32), 0.0)
                m = bx0 & by1 & bz0
                acc += tl.where(m, (wz0 * wy1) * wx0 * tl.load(ptr + off + D, mask=m, other=0).to(tl.float32), 0.0)
                m = bx0 & by1 & bz1
                acc += tl.where(m, (wz1 * wy1) * wx0 * tl.load(ptr + off + D + 1, mask=m, other=0).to(tl.float32), 0.0)
                m = bx1 & by0 & bz0
                acc += tl.where(m, (wz0 * wy0) * wx1 * tl.load(ptr + off + HD, mask=m, other=0).to(tl.float32), 0.0)
                m = bx1 & by0 & bz1
                acc += tl.where(m, (wz1 * wy0) * wx1 * tl.load(ptr + off + HD + 1, mask=m, other=0).to(tl.float32), 0.0)
                m = bx1 & by1 & bz0
                acc += tl.where(m, (wz0 * wy1) * wx1 * tl.load(ptr + off + HD + D, mask=m, other=0).to(tl.float32), 0.0)
                m = bx1 & by1 & bz1
                acc += tl.where(m, (wz1 * wy1) * wx1 * tl.load(ptr + off + HD + D + 1, mask=m, other=0).to(tl.float32), 0.0)
                if PAD == 2:
                    acc = tl.where(oob, fill, acc)
                tl.store(out_ptr + c * numel64 + i, acc, mask=valid)
        else:
            x0, y0, z0 = _index(_round(sx), W), _index(_round(sy), H), _index(_round(sz), D)
            m = valid & (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H) & (z0 >= 0) & (z0 < D)
            off = (x0 * H + y0) * D + z0
            for c in range(C):
                acc = tl.load(src_ptr + c * WHD + off, mask=m, other=0).to(tl.float32)
                if PAD == 2:
                    acc = tl.where(oob, fill, acc)
                tl.store(out_ptr + c * numel64 + i, acc, mask=valid)

    @triton.jit(do_not_specialize=_SIZES)
    def _resample_backward(src_ptr, m_ptr, grad_ptr, gsrc_ptr, part_ptr, C, W, H, D, OW, OH, OD, numel, fill,
                           MODE: tl.constexpr, PAD: tl.constexpr, NEED_SGRAD: tl.constexpr,
                           NEED_MGRAD: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < numel
        xf, yf, zf, sx, sy, sz = _coordinates(m_ptr, i, OH, OD)
        if PAD == 2:
            # filled voxels were overwritten, so no gradient flows through them
            valid &= ~((sx < -0.5) | (sx > W - 0.5) | (sy < -0.5) | (sy > H - 0.5) | (sz < -0.5) | (sz > D - 0.5))
        ux, uy, uz = sx, sy, sz
        if PAD != 0:
            sx, sy, sz = _clamp(sx, W), _clamp(sy, H), _clamp(sz, D)
        WHD = W.to(tl.int64) * H * D
        HD = H.to(tl.int64) * D
        numel64 = numel.to(tl.int64)

        if MODE == 0:
            fx, fy, fz = tl.floor(sx), tl.floor(sy), tl.floor(sz)
            x0, y0, z0 = _index(fx, W), _index(fy, H), _index(fz, D)
            wx0, wx1 = (fx + 1) - sx, sx - fx
            wy0, wy1 = (fy + 1) - sy, sy - fy
            wz0, wz1 = (fz + 1) - sz, sz - fz
            bx0, bx1 = valid & (x0 >= 0) & (x0 < W), valid & (x0 + 1 >= 0) & (x0 + 1 < W)
            by0, by1 = (y0 >= 0) & (y0 < H), (y0 + 1 >= 0) & (y0 + 1 < H)
            bz0, bz1 = (z0 >= 0) & (z0 < D), (z0 + 1 >= 0) & (z0 + 1 < D)
            m000, m001, m010, m011 = bx0 & by0 & bz0, bx0 & by0 & bz1, bx0 & by1 & bz0, bx0 & by1 & bz1
            m100, m101, m110, m111 = bx1 & by0 & bz0, bx1 & by0 & bz1, bx1 & by1 & bz0, bx1 & by1 & bz1
            off = (x0 * H + y0) * D + z0
            gx = tl.zeros([BLOCK], dtype=tl.float32)
            gy = tl.zeros([BLOCK], dtype=tl.float32)
            gz = tl.zeros([BLOCK], dtype=tl.float32)
            for c in range(C):
                g = tl.load(grad_ptr + c * numel64 + i, mask=valid, other=0)
                if NEED_SGRAD:
                    ptr = gsrc_ptr + c * WHD
                    tl.atomic_add(ptr + off, g * ((wz0 * wy0) * wx0), mask=m000, sem='relaxed')
                    tl.atomic_add(ptr + off + 1, g * ((wz1 * wy0) * wx0), mask=m001, sem='relaxed')
                    tl.atomic_add(ptr + off + D, g * ((wz0 * wy1) * wx0), mask=m010, sem='relaxed')
                    tl.atomic_add(ptr + off + D + 1, g * ((wz1 * wy1) * wx0), mask=m011, sem='relaxed')
                    tl.atomic_add(ptr + off + HD, g * ((wz0 * wy0) * wx1), mask=m100, sem='relaxed')
                    tl.atomic_add(ptr + off + HD + 1, g * ((wz1 * wy0) * wx1), mask=m101, sem='relaxed')
                    tl.atomic_add(ptr + off + HD + D, g * ((wz0 * wy1) * wx1), mask=m110, sem='relaxed')
                    tl.atomic_add(ptr + off + HD + D + 1, g * ((wz1 * wy1) * wx1), mask=m111, sem='relaxed')
                if NEED_MGRAD:
                    ptr = src_ptr + c * WHD
                    v000 = tl.load(ptr + off, mask=m000, other=0).to(tl.float32)
                    v001 = tl.load(ptr + off + 1, mask=m001, other=0).to(tl.float32)
                    v010 = tl.load(ptr + off + D, mask=m010, other=0).to(tl.float32)
                    v011 = tl.load(ptr + off + D + 1, mask=m011, other=0).to(tl.float32)
                    v100 = tl.load(ptr + off + HD, mask=m100, other=0).to(tl.float32)
                    v101 = tl.load(ptr + off + HD + 1, mask=m101, other=0).to(tl.float32)
                    v110 = tl.load(ptr + off + HD + D, mask=m110, other=0).to(tl.float32)
                    v111 = tl.load(ptr + off + HD + D + 1, mask=m111, other=0).to(tl.float32)
                    gx += g * ((v100 - v000) * (wz0 * wy0) + (v101 - v001) * (wz1 * wy0) +
                               (v110 - v010) * (wz0 * wy1) + (v111 - v011) * (wz1 * wy1))
                    gy += g * ((v010 - v000) * (wz0 * wx0) + (v011 - v001) * (wz1 * wx0) +
                               (v110 - v100) * (wz0 * wx1) + (v111 - v101) * (wz1 * wx1))
                    gz += g * ((v001 - v000) * (wy0 * wx0) + (v011 - v010) * (wy1 * wx0) +
                               (v101 - v100) * (wy0 * wx1) + (v111 - v110) * (wy1 * wx1))
            if NEED_MGRAD:
                if PAD != 0:
                    # clipped coordinates have no gradient (borders inclusive, as grid_sample)
                    gx = tl.where((ux <= 0) | (ux >= W - 1), 0.0, gx)
                    gy = tl.where((uy <= 0) | (uy >= H - 1), 0.0, gy)
                    gz = tl.where((uz <= 0) | (uz >= D - 1), 0.0, gz)
                ptr = part_ptr + tl.program_id(0) * 12
                tl.store(ptr + 0, tl.sum(gx * xf, 0))
                tl.store(ptr + 1, tl.sum(gx * yf, 0))
                tl.store(ptr + 2, tl.sum(gx * zf, 0))
                tl.store(ptr + 3, tl.sum(gx, 0))
                tl.store(ptr + 4, tl.sum(gy * xf, 0))
                tl.store(ptr + 5, tl.sum(gy * yf, 0))
                tl.store(ptr + 6, tl.sum(gy * zf, 0))
                tl.store(ptr + 7, tl.sum(gy, 0))
                tl.store(ptr + 8, tl.sum(gz * xf, 0))
                tl.store(ptr + 9, tl.sum(gz * yf, 0))
                tl.store(ptr + 10, tl.sum(gz * zf, 0))
                tl.store(ptr + 11, tl.sum(gz, 0))
        else:
            x0, y0, z0 = _index(_round(sx), W), _index(_round(sy), H), _index(_round(sz), D)
            m = valid & (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H) & (z0 >= 0) & (z0 < D)
            off = (x0 * H + y0) * D + z0
            for c in range(C):
                g = tl.load(grad_ptr + c * numel64 + i, mask=m, other=0)
                tl.atomic_add(gsrc_ptr + c * WHD + off, g, mask=m, sem='relaxed')
