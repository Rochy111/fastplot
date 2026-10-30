"""The GPU canvas (cupy): fields sampled, rasterized, coloured and blended
by CUDA kernels -- the same arithmetic as _raster, _color and _render's
numba kernels, compiled without FMA contraction, so the image is the CPU's.

    module()                        # the compiled kernels (once a process)
    c = GpuCanvas(H, W, rgba)       # _render.CpuCanvas's methods

Where the time goes on a map is per pixel and per grid point: sampling a
1M-point field onto 2M pixels, rasterizing a 3.7M-point curvilinear grid,
colouring. That is what runs on the card. Lines, polygons and text --
a few thousand segments and glyphs, a few ms on the CPU's tiles -- run on
the host: the canvas moves to host memory once, when the first of them is
drawn (fields usually come first, zorder 1), and back only if another
field follows. A typical frame is then one upload of the field and one
download of the image.
"""
import threading

import numpy as np

from . import _color, _proj, _render

_lock = threading.Lock()
_module = None

CORE = r'''
__device__ __forceinline__ double fp_pymod(double a, double b) {
    double r = fmod(a, b);
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
    return r;
}

__device__ __forceinline__ long long fp_wrap(long long k, long long n) {
    k = k % n;
    return k < 0 ? k + n : k;
}

/* _raster._frac: fractional index of u on an axis (params ap, coords). */
__device__ double fp_frac(double u, const double* ap, const double* coords, int m) {
    double x0 = ap[0], dx = ap[1], n = ap[2];
    bool uniform = ap[3] != 0.0, edges = ap[4] != 0.0, periodic = ap[5] != 0.0;
    bool angular = ap[6] != 0.0;
    double lo = ap[7];
    double f;
    if (angular) u = lo + fp_pymod(u - lo, 360.0);
    if (uniform) {
        f = (u - x0) / dx;
    } else if (u <= coords[0]) {
        f = (u - coords[0]) / (coords[1] - coords[0]);
    } else if (u >= coords[m - 1]) {
        f = (m - 1) + (u - coords[m - 1]) / (coords[m - 1] - coords[m - 2]);
    } else {
        int a = 0, b = m;                       /* searchsorted(side='right') */
        while (a < b) { int c = (a + b) >> 1; if (coords[c] <= u) a = c + 1; else b = c; }
        int k = a - 1;
        f = k + (u - coords[k]) / (coords[k + 1] - coords[k]);
    }
    if (periodic) return f;
    if (edges) { if (f < 0.0 || f >= n) return CUDART_NAN; }
    else if (f < -0.5 || f >= n - 0.5) return CUDART_NAN;
    return f;
}

extern "C" __global__ void sample_rect(float* out, int H, int W, int mkind, const double* mp,
        double x0, double dx, double y0, double dy, int skind, const double* sp,
        const double* xp, const double* xc, int nxc, const double* yp, const double* yc, int nyc,
        const float* z, long long zs0, long long zs1, int mode) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) return;
    long long nxv = (long long)xp[2], nyv = (long long)yp[2];
    bool xper = xp[5] != 0.0, yper = yp[5] != 0.0, xedge = xp[4] != 0.0, yedge = yp[4] != 0.0;
    float nanf_ = __int_as_float(0x7fc00000);
    double Y = y0 + (i + 0.5) * dy;
    double X = x0 + (j + 0.5) * dx;
    double lon, lat, sx, sy;
    fp_inv(mkind, mp, X, Y, &lon, &lat);
    long long o = (long long)i * W + j;
    if (!(isfinite(lon) && isfinite(lat))) { out[o] = nanf_; return; }
    fp_fwd(skind, sp, lon, lat, &sx, &sy);
    double fx = fp_frac(sx, xp, xc, nxc);
    double fy = fp_frac(sy, yp, yc, nyc);
    if (!(isfinite(fx) && isfinite(fy))) { out[o] = nanf_; return; }
    if (mode == 0) {
        long long kx = xedge ? (long long)floor(fx) : (long long)floor(fx + 0.5);
        long long ky = yedge ? (long long)floor(fy) : (long long)floor(fy + 0.5);
        if (xper) kx = fp_wrap(kx, nxv);
        if (yper) ky = fp_wrap(ky, nyv);
        out[o] = z[ky * zs0 + kx * zs1];
        return;
    }
    if (mode != 3 && ((!xper && (fx < 0.0 || fx > nxv - 1)) ||
                      (!yper && (fy < 0.0 || fy > nyv - 1)))) {
        out[o] = nanf_; return;
    }
    long long kx = (long long)floor(fx), ky = (long long)floor(fy);
    double tx = fx - kx, ty = fy - ky;
    long long kx1 = kx + 1, ky1 = ky + 1;
    if (xper) { kx = fp_wrap(kx, nxv); kx1 = fp_wrap(kx1, nxv); }
    else {
        if (kx < 0) { kx = 0; tx = 0.0; }
        if (kx1 > nxv - 1) { kx1 = nxv - 1; tx = 0.0; kx = kx < nxv - 1 ? kx : nxv - 1; }
    }
    if (yper) { ky = fp_wrap(ky, nyv); ky1 = fp_wrap(ky1, nyv); }
    else {
        if (ky < 0) { ky = 0; ty = 0.0; }
        if (ky1 > nyv - 1) { ky1 = nyv - 1; ty = 0.0; ky = ky < nyv - 1 ? ky : nyv - 1; }
    }
    double v00 = (double)z[ky * zs0 + kx * zs1], v01 = (double)z[ky * zs0 + kx1 * zs1];
    double v10 = (double)z[ky1 * zs0 + kx * zs1], v11 = (double)z[ky1 * zs0 + kx1 * zs1];
    double v = (v00 * (1.0 - tx) + v01 * tx) * (1.0 - ty) + (v10 * (1.0 - tx) + v11 * tx) * ty;
    out[o] = (float)v;
}

extern "C" __global__ void forward_nodes(int n, int mkind, const double* mp, double x0,
        double dx, double y0, double dy, int skind, const double* sp, const double* xs,
        const double* ys, double* sx, double* sy) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double lon, lat, X, Y;
    fp_inv(skind, sp, xs[k], ys[k], &lon, &lat);
    fp_fwd(mkind, mp, lon, lat, &X, &Y);
    sx[k] = (X - x0) / dx - 0.5;
    sy[k] = (Y - y0) / dy - 0.5;
}

/* _raster._tri */
__device__ void fp_tri(float* out, double ax, double ay, double bx, double by, double cx,
        double cy, float za, float zb, float zc, int mode, int kind, const float* zn, int H,
        int W) {
    double area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (area == 0.0 || !isfinite(area)) return;
    long long j0 = (long long)ceil(fmin(ax, fmin(bx, cx))); if (j0 < 0) j0 = 0;
    long long j1 = (long long)floor(fmax(ax, fmax(bx, cx))); if (j1 > W - 1) j1 = W - 1;
    long long i0 = (long long)ceil(fmin(ay, fmin(by, cy))); if (i0 < 0) i0 = 0;
    long long i1 = (long long)floor(fmax(ay, fmax(by, cy))); if (i1 > H - 1) i1 = H - 1;
    double inv = 1.0 / area;
    for (long long i = i0; i <= i1; i++) {
        double py = (double)i;
        for (long long j = j0; j <= j1; j++) {
            double px = (double)j;
            double w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * inv;
            double w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * inv;
            double w2 = 1.0 - w0 - w1;
            if (w0 < -1e-12 || w1 < -1e-12 || w2 < -1e-12) continue;
            if (mode == 1) {
                out[i * W + j] = (float)(w0 * za + w1 * zb + w2 * zc);
            } else {
                double s, t;
                if (kind == 0) { s = w1 + w2; t = w2; } else { s = w1; t = w1 + w2; }
                int q = (s >= 0.5 ? 1 : 0) + (t >= 0.5 ? 2 : 0);
                const int map[4] = {0, 1, 3, 2};
                out[i * W + j] = zn[map[q]];
            }
        }
    }
}

extern "C" __global__ void raster_cells(float* out, int H, int W, const double* sx,
        const double* sy, int ny, int nx, const float* z, int mode, double period) {
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (long long)(ny - 1) * (nx - 1)) return;
    int i = (int)(c / (nx - 1)), j = (int)(c % (nx - 1));
    double x00 = sx[(long long)i * nx + j], y00 = sy[(long long)i * nx + j];
    double x01 = sx[(long long)i * nx + j + 1], y01 = sy[(long long)i * nx + j + 1];
    double x11 = sx[(long long)(i + 1) * nx + j + 1], y11 = sy[(long long)(i + 1) * nx + j + 1];
    double x10 = sx[(long long)(i + 1) * nx + j], y10 = sy[(long long)(i + 1) * nx + j];
    if (!(isfinite(x00) && isfinite(x01) && isfinite(x11) && isfinite(x10))) return;
    if (period > 0.0) {
        double h = 0.5 * period;
        if (x01 - x00 > h) x01 -= period; else if (x01 - x00 < -h) x01 += period;
        if (x11 - x00 > h) x11 -= period; else if (x11 - x00 < -h) x11 += period;
        if (x10 - x00 > h) x10 -= period; else if (x10 - x00 < -h) x10 += period;
    }
    double lo = fmin(fmin(x00, x01), fmin(x11, x10));
    double hi = fmax(fmax(x00, x01), fmax(x11, x10));
    if ((hi < -1.0 && period == 0.0) || (lo > W && period == 0.0)) return;
    if (fmax(fmax(y00, y01), fmax(y11, y10)) < -1.0 || fmin(fmin(y00, y01), fmin(y11, y10)) > H)
        return;
    float z00, z01, z11, z10;
    if (mode == 2) { z00 = z01 = z11 = z10 = z[(long long)i * (nx - 1) + j]; }
    else {
        z00 = z[(long long)i * nx + j]; z01 = z[(long long)i * nx + j + 1];
        z11 = z[(long long)(i + 1) * nx + j + 1]; z10 = z[(long long)(i + 1) * nx + j];
    }
    float zn[4] = {z00, z01, z11, z10};
    double shifts[2] = {0.0, 0.0};
    int nshift = 1;
    if (period > 0.0) {
        if (lo < 0.0) { shifts[1] = period; nshift = 2; }
        else if (hi > W - 1) { shifts[1] = -period; nshift = 2; }
    }
    int tm = (mode == 2 || mode == 3) ? 1 : mode;
    for (int q = 0; q < nshift; q++) {
        double o = shifts[q];
        fp_tri(out, x00 + o, y00, x01 + o, y01, x11 + o, y11, z00, z01, z11, tm, 0, zn, H, W);
        fp_tri(out, x00 + o, y00, x11 + o, y11, x10 + o, y10, z00, z11, z10, tm, 1, zn, H, W);
    }
}

/* _raster._sample_sep: per-column and per-row tables (k0, k1, t, ok) */
extern "C" __global__ void sample_sep(float* out, int H, int W, const double* cols,
        const double* rows, const float* z, long long zs0, long long zs1, int mode) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) return;
    long long o = (long long)i * W + j;
    const double* rw = rows + 4 * (long long)i;
    const double* cl = cols + 4 * (long long)j;
    if (rw[3] == 0.0 || cl[3] == 0.0) { out[o] = __int_as_float(0x7fc00000); return; }
    long long ky0 = (long long)rw[0], ky1 = (long long)rw[1];
    long long kx0 = (long long)cl[0], kx1 = (long long)cl[1];
    if (mode == 0) { out[o] = z[ky0 * zs0 + kx0 * zs1]; return; }
    double tx = cl[2], ty = rw[2];
    double v00 = (double)z[ky0 * zs0 + kx0 * zs1], v01 = (double)z[ky0 * zs0 + kx1 * zs1];
    double v10 = (double)z[ky1 * zs0 + kx0 * zs1], v11 = (double)z[ky1 * zs0 + kx1 * zs1];
    double v = (v00 * (1.0 - tx) + v01 * tx) * (1.0 - ty) + (v10 * (1.0 - tx) + v11 * tx) * ty;
    out[o] = (float)v;
}

/* _draw.blend_px */
__device__ __forceinline__ void fp_blend(unsigned char* p, float r, float g, float b, float a,
                                         float m) {
    float f = (a / 255.0f) * m;
    float k = 1.0f - f;
    float da = (float)p[3];
    if (da == 255.0f) {
        p[0] = (unsigned char)(r * f + (float)p[0] * k + 0.5f);
        p[1] = (unsigned char)(g * f + (float)p[1] * k + 0.5f);
        p[2] = (unsigned char)(b * f + (float)p[2] * k + 0.5f);
        p[3] = (unsigned char)(255.0f * f + da * k + 0.5f);
        return;
    }
    float w = (da / 255.0f) * k;
    float oa = f + w;
    if (oa <= 0.0f) return;
    p[0] = (unsigned char)((r * f + (float)p[0] * w) / oa + 0.5f);
    p[1] = (unsigned char)((g * f + (float)p[1] * w) / oa + 0.5f);
    p[2] = (unsigned char)((b * f + (float)p[2] * w) / oa + 0.5f);
    p[3] = (unsigned char)(255.0f * oa + 0.5f);
}

/* _render._blend_image */
extern "C" __global__ void blend_image(unsigned char* canvas, int H, int W,
        const unsigned char* img, int h, int w, int y0, int x0, float alpha) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= h || j >= w) return;
    int ci = y0 + i, cj = x0 + j;
    if (ci < 0 || ci >= H || cj < 0 || cj >= W) return;
    const unsigned char* s = img + ((long long)i * w + j) * 4;
    unsigned char* p = canvas + ((long long)ci * W + cj) * 4;
    unsigned char a = s[3];
    if (a == 0) return;
    if (a == 255 && alpha == 1.0f) { p[0] = s[0]; p[1] = s[1]; p[2] = s[2]; p[3] = 255; return; }
    fp_blend(p, (float)s[0], (float)s[1], (float)s[2], (float)a, alpha);
}

extern "C" __global__ void fill_rect(unsigned char* canvas, int H, int W, int x0, int y0, int x1,
        int y1, float r, float g, float b, float a) {
    int j = x0 + blockIdx.x * blockDim.x + threadIdx.x;
    int i = y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= y1 || j >= x1) return;
    unsigned char* p = canvas + ((long long)i * W + j) * 4;
    if (a == 255.0f) { p[0] = (unsigned char)r; p[1] = (unsigned char)g; p[2] = (unsigned char)b;
                       p[3] = 255; return; }
    fp_blend(p, r, g, b, a, 1.0f);
}
'''


def module():
    """The RawModule of every kernel, compiled on first use."""
    global _module
    if _module is None:
        with _lock:
            if _module is None:
                import cupy as cp
                code = ('#include <math_constants.h>\n' + _proj.CUDA + '\n' + _color.CUDA
                        + '\n' + CORE)
                m = cp.RawModule(code=code, options=('--std=c++14', '--fmad=false'))
                m.compile()
                _module = m
    return _module


def _grid2(h, w, b=(32, 8)):
    return ((w + b[0] - 1) // b[0], (h + b[1] - 1) // b[1]), b


class GpuCanvas:
    """The canvas on the card while fields are drawn; in host memory for
    lines, polygons and text (see the module docstring)."""

    device = 'gpu'

    def __init__(self, H, W, rgba):
        import cupy as cp
        self.cp = cp
        self.m = module()
        self.H, self.W = H, W
        self.d = cp.empty((H, W, 4), cp.uint8)
        self.d[...] = cp.asarray(np.asarray(rgba, np.uint8))
        self.h = None
        self.on = 'gpu'
        self._cpu = None

    # -- residency
    def _to_gpu(self):
        if self.on == 'cpu':
            self.d.set(self.h)
            self.on = 'gpu'

    def _to_cpu(self):
        if self.on == 'gpu':
            if self.h is None:
                self.h = self.cp.cuda.alloc_pinned_memory(self.H * self.W * 4)
                self.h = np.frombuffer(self.h, np.uint8, self.H * self.W * 4).reshape(
                    self.H, self.W, 4)
            self.d.get(out=self.h)
            self.on = 'cpu'
        if self._cpu is None:
            c = _render.CpuCanvas.__new__(_render.CpuCanvas)
            c.H, c.W = self.H, self.W
            self._cpu = c
        self._cpu.a = self.h
        return self._cpu

    def _dev(self, a, dtype):
        cp = self.cp
        if isinstance(a, cp.ndarray):
            return a.astype(dtype, copy=False)
        return cp.asarray(np.asarray(a, dtype))

    # -- operations
    def fill_rect(self, x0, y0, x1, y1, rgba):
        if self.on == 'cpu':
            return self._to_cpu().fill_rect(x0, y0, x1, y1, rgba)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.W, int(x1)), min(self.H, int(y1))
        if x1 <= x0 or y1 <= y0 or rgba[3] == 0:
            return
        c = [np.float32(v) for v in np.asarray(rgba, np.float32)]
        g, b = _grid2(y1 - y0, x1 - x0)
        self.m.get_function('fill_rect')(g, b, (self.d, np.int32(self.H), np.int32(self.W),
                                                np.int32(x0), np.int32(y0), np.int32(x1),
                                                np.int32(y1), *c))

    def _shade_blend(self, box, values, coloring, alpha):
        cp = self.cp
        h, w = values.shape
        if coloring.bands:                              # contourf: the CPU's touched_bands
            from ._render import touched_bands_xp
            values = touched_bands_xp(cp, values, coloring.top)
        layer = cp.empty((h, w, 4), cp.uint8)
        plan = coloring.plan(None if coloring.bands else alpha)
        if plan.kernel is None:                         # a norm only the CPU has (FuncNorm)
            host = np.empty((h, w, 4), np.uint8)
            coloring.shade(values.get(), host, None if coloring.bands else alpha)
            layer.set(host)
        else:
            args = tuple(cp.asarray(a) if isinstance(a, np.ndarray) else a for a in plan.args)
            n = h * w
            self.m.get_function(plan.kernel)(((n + 255) // 256,), (256,),
                                             (values, np.int32(n)) + args + (layer,))
        a = np.float32(1.0 if (alpha is None or not coloring.bands) else alpha)
        g, b = _grid2(h, w)
        self.m.get_function('blend_image')(g, b, (self.d, np.int32(self.H), np.int32(self.W),
                                                  layer, np.int32(h), np.int32(w),
                                                  np.int32(box[1]), np.int32(box[0]), a))

    def field_rect(self, box, view, src, xaxis, yaxis, z, mode, coloring, alpha=None):
        cp = self.cp
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        self._to_gpu()
        zd = self._dev(z, np.float32)
        if yaxis.flip:
            zd = zd[::-1]
        if xaxis.flip:
            zd = zd[:, ::-1]
        isz = zd.itemsize
        values = cp.empty((h, w), cp.float32)
        g, b = _grid2(h, w)
        self.m.get_function('sample_rect')(g, b, (
            values, np.int32(h), np.int32(w), np.int32(view.kind), cp.asarray(view.params),
            np.float64(view.x0), np.float64(view.dx), np.float64(view.y0), np.float64(view.dy),
            np.int32(src.kind), cp.asarray(src.params), cp.asarray(xaxis.params()),
            cp.asarray(xaxis.coords), np.int32(xaxis.coords.size), cp.asarray(yaxis.params()),
            cp.asarray(yaxis.coords), np.int32(yaxis.coords.size), zd,
            np.int64(zd.strides[0] // isz), np.int64(zd.strides[1] // isz), np.int32(mode)))
        self._shade_blend(box, values, coloring, alpha)

    def field_cells(self, box, view, src, x2d, y2d, z, mode, coloring, alpha=None, period=0.0):
        cp = self.cp
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        self._to_gpu()
        xs = self._dev(x2d, np.float64)
        ys = self._dev(y2d, np.float64)
        ny, nx = xs.shape
        n = ny * nx
        sx = cp.empty((ny, nx), cp.float64)
        sy = cp.empty((ny, nx), cp.float64)
        self.m.get_function('forward_nodes')(((n + 255) // 256,), (256,), (
            np.int32(n), np.int32(view.kind), cp.asarray(view.params), np.float64(view.x0),
            np.float64(view.dx), np.float64(view.y0), np.float64(view.dy), np.int32(src.kind),
            cp.asarray(src.params), cp.ascontiguousarray(xs), cp.ascontiguousarray(ys), sx, sy))
        values = cp.full((h, w), np.nan, cp.float32)
        zd = cp.ascontiguousarray(self._dev(z, np.float32))
        nc = (ny - 1) * (nx - 1)
        self.m.get_function('raster_cells')(((nc + 255) // 256,), (256,), (
            values, np.int32(h), np.int32(w), sx, sy, np.int32(ny), np.int32(nx), zd,
            np.int32(mode), np.float64(period)))
        self._shade_blend(box, values, coloring, alpha)

    def field_sep(self, box, cols, rows, z, grid, mode, coloring, alpha=None):
        cp = self.cp
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        self._to_gpu()
        zd = self._dev(z, np.float32)
        if grid.yaxis.flip:
            zd = zd[::-1]
        if grid.xaxis.flip:
            zd = zd[:, ::-1]
        isz = zd.itemsize
        values = cp.empty((h, w), cp.float32)
        g, b = _grid2(h, w)
        self.m.get_function('sample_sep')(g, b, (
            values, np.int32(h), np.int32(w), cp.asarray(cols), cp.asarray(rows), zd,
            np.int64(zd.strides[0] // isz), np.int64(zd.strides[1] // isz), np.int32(mode)))
        self._shade_blend(box, values, coloring, alpha)

    def field_cells_px(self, box, sx, sy, z, mode, coloring, alpha=None):
        cp = self.cp
        h, w = box[3] - box[1], box[2] - box[0]
        if h <= 0 or w <= 0:
            return
        self._to_gpu()
        ny, nx = sx.shape
        values = cp.full((h, w), np.nan, cp.float32)
        zd = cp.ascontiguousarray(self._dev(z, np.float32))
        nc = (ny - 1) * (nx - 1)
        self.m.get_function('raster_cells')(((nc + 255) // 256,), (256,), (
            values, np.int32(h), np.int32(w), cp.asarray(np.ascontiguousarray(sx, np.float64)),
            cp.asarray(np.ascontiguousarray(sy, np.float64)), np.int32(ny), np.int32(nx), zd,
            np.int32(mode), np.float64(0.0)))
        self._shade_blend(box, values, coloring, alpha)

    def segments(self, segs, styles, clip, memo=None):
        self._to_cpu().segments(segs, styles, clip, memo)

    def polygons(self, xy, ring_offsets, rgba, clip, rule='evenodd'):
        self._to_cpu().polygons(xy, ring_offsets, rgba, clip, rule)

    def mask(self, alpha_mask, x0, y0, rgba):
        self._to_cpu().mask(alpha_mask, x0, y0, rgba)

    def image(self, rgba, x0, y0, alpha=1.0):
        self._to_cpu().image(rgba, x0, y0, alpha)

    def image_nearest(self, rgba, rows, cols, x0, y0, alpha=1.0):
        self._to_cpu().image_nearest(rgba, rows, cols, x0, y0, alpha)

    def field_values(self, box, values, coloring, alpha=None):
        self._to_cpu().field_values(box, values, coloring, alpha)

    def hatch(self, xy, ring_offsets, pattern, color, dpi, clip):
        self._to_cpu().hatch(xy, ring_offsets, pattern, color, dpi, clip)

    def stamp(self, pts, fill, stroke, width, face, edge, clip, cap, which=None):
        self._to_cpu().stamp(pts, fill, stroke, width, face, edge, clip, cap, which)

    def array(self, copy=True):
        """The image in host memory: a copy, or (copy=False) the page-locked
        download buffer itself, for an image used and dropped at once."""
        self._to_cpu()
        return np.array(self.h) if copy else self.h
