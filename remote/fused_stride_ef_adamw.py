"""Fused structured error-feedback + BF16-high AdamW CUDA backend.

The external and DFC kernels implement the same logical transition. The only
placement difference is that external-EF loads/stores a separate FP32 residual,
while DFC reconstructs/stores the same FP32 residual from the low 16-bit words
of the two physical FP32 Adam moment containers. The semantic Adam moments use
only their high 16 bits in both methods.

One launch performs residual decode, compensated-gradient construction,
structured 1/stride selection, FP16 transport quantization, residual update,
AdamW moment update, parameter update, and (for DFC) residual repacking. This
is intentionally the traffic-elimination realization: DFC does not launch a
separate decode/encode kernel and does not materialize an auxiliary residual.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import torch

_EXT = None

_CPP = r"""
#include <torch/extension.h>
void fused_external_cuda(torch::Tensor p, torch::Tensor g, torch::Tensor m, torch::Tensor v,
                         torch::Tensor r, int64_t stride, int64_t phase,
                         double lr, double beta1, double beta2, double eps,
                         double weight_decay, int64_t step);
void fused_dfc_cuda(torch::Tensor p, torch::Tensor g, torch::Tensor m, torch::Tensor v,
                    int64_t stride, int64_t phase,
                    double lr, double beta1, double beta2, double eps,
                    double weight_decay, int64_t step);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("external_step", &fused_external_cuda, "fused external EF + AdamW (CUDA)");
  m.def("dfc_step", &fused_dfc_cuda, "fused DFC EF + AdamW (CUDA)");
}
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

static inline void check_common(torch::Tensor p, torch::Tensor g, torch::Tensor m, torch::Tensor v) {
  TORCH_CHECK(p.is_cuda() && g.is_cuda() && m.is_cuda() && v.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(p.scalar_type() == at::kHalf && g.scalar_type() == at::kHalf, "p/g must be float16");
  TORCH_CHECK(m.scalar_type() == at::kFloat && v.scalar_type() == at::kFloat, "m/v must be float32");
  TORCH_CHECK(p.is_contiguous() && g.is_contiguous() && m.is_contiguous() && v.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(p.numel() == g.numel() && p.numel() == m.numel() && p.numel() == v.numel(), "size mismatch");
}

__device__ __forceinline__ uint32_t f2u(float x) { return __float_as_uint(x); }
__device__ __forceinline__ float u2f(uint32_t x) { return __uint_as_float(x); }

// Compile-time placement specialization keeps arithmetic order identical while
// allowing the DFC variant to omit the auxiliary residual memory transaction.
template <bool DFC>
__global__ void fused_kernel(__half* p, const __half* g, float* m, float* v, float* r,
                             int64_t n, int stride, int phase,
                             float lr, float beta1, float beta2, float eps,
                             float weight_decay, float step_size, float bias2_sqrt) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  uint32_t mb = f2u(m[i]);
  uint32_t vb = f2u(v[i]);
  float m_sem = u2f(mb & 0xFFFF0000u);
  float v_sem = u2f(vb & 0xFFFF0000u);

  float residual;
  if constexpr (DFC) {
    uint32_t rb = (mb & 0x0000FFFFu) | ((vb & 0x0000FFFFu) << 16);
    residual = u2f(rb);
  } else {
    residual = r[i];
  }

  float grad = __half2float(g[i]);
  float compensated = grad + residual;
  bool keep = ((int)(i % stride)) == phase;
  __half qh = keep ? __float2half_rn(compensated) : __float2half_rn(0.0f);
  float q = __half2float(qh);
  float residual_new = keep ? (compensated - q) : compensated;

  float m_new = beta1 * m_sem + (1.0f - beta1) * q;
  float v_new = beta2 * v_sem + (1.0f - beta2) * q * q;
  uint32_t mhi = f2u(m_new) & 0xFFFF0000u;
  uint32_t vhi = f2u(v_new) & 0xFFFF0000u;
  float m_use = u2f(mhi);
  float v_use = u2f(vhi);

  float denom = sqrtf(v_use) / bias2_sqrt + eps;

  __half ph = p[i];
  if (weight_decay != 0.0f) {
    ph = __float2half_rn(__half2float(ph) * (1.0f - lr * weight_decay));
  }
  __half uh = __float2half_rn((-step_size) * m_use / denom);
  p[i] = __float2half_rn(__half2float(ph) + __half2float(uh));

  if constexpr (DFC) {
    uint32_t rb_new = f2u(residual_new);
    m[i] = u2f(mhi | (rb_new & 0x0000FFFFu));
    v[i] = u2f(vhi | ((rb_new >> 16) & 0x0000FFFFu));
  } else {
    m[i] = u2f(mhi);
    v[i] = u2f(vhi);
    r[i] = residual_new;
  }
}

static inline dim3 blocks_for(int64_t n) { return dim3((unsigned int)((n + 255) / 256)); }

void fused_external_cuda(torch::Tensor p, torch::Tensor g, torch::Tensor m, torch::Tensor v,
                         torch::Tensor r, int64_t stride, int64_t phase,
                         double lr, double beta1, double beta2, double eps,
                         double weight_decay, int64_t step) {
  check_common(p, g, m, v);
  TORCH_CHECK(r.is_cuda() && r.scalar_type() == at::kFloat && r.is_contiguous(), "r must be contiguous CUDA float32");
  TORCH_CHECK(r.numel() == p.numel(), "residual size mismatch");
  TORCH_CHECK(stride > 0 && phase >= 0 && phase < stride && step > 0, "invalid stride/phase/step");
  c10::cuda::CUDAGuard guard(p.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  float step_size = (float)lr / (1.0f - std::pow((float)beta1, (float)step));
  float bias2_sqrt = std::sqrt(1.0f - std::pow((float)beta2, (float)step));
  fused_kernel<false><<<blocks_for(p.numel()), 256, 0, stream>>>(
      reinterpret_cast<__half*>(p.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(g.data_ptr<at::Half>()),
      m.data_ptr<float>(), v.data_ptr<float>(), r.data_ptr<float>(), p.numel(),
      (int)stride, (int)phase, (float)lr, (float)beta1, (float)beta2, (float)eps,
      (float)weight_decay, step_size, bias2_sqrt);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_dfc_cuda(torch::Tensor p, torch::Tensor g, torch::Tensor m, torch::Tensor v,
                    int64_t stride, int64_t phase,
                    double lr, double beta1, double beta2, double eps,
                    double weight_decay, int64_t step) {
  check_common(p, g, m, v);
  TORCH_CHECK(stride > 0 && phase >= 0 && phase < stride && step > 0, "invalid stride/phase/step");
  c10::cuda::CUDAGuard guard(p.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  float step_size = (float)lr / (1.0f - std::pow((float)beta1, (float)step));
  float bias2_sqrt = std::sqrt(1.0f - std::pow((float)beta2, (float)step));
  fused_kernel<true><<<blocks_for(p.numel()), 256, 0, stream>>>(
      reinterpret_cast<__half*>(p.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(g.data_ptr<at::Half>()),
      m.data_ptr<float>(), v.data_ptr<float>(), nullptr, p.numel(),
      (int)stride, (int)phase, (float)lr, (float)beta1, (float)beta2, (float)eps,
      (float)weight_decay, step_size, bias2_sqrt);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def load_backend(verbose: bool = False):
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from torch.utils.cpp_extension import load_inline
    cap = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}")
    build_root = Path(os.environ.get("DFC_FUSED_BUILD_DIR", "/tmp/dfc_fused_cuda"))
    build_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((_CPP + _CUDA + torch.__version__).encode()).hexdigest()[:12]
    _EXT = load_inline(
        name=f"dfc_fused_stride_{key}",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        with_cuda=True,
        build_directory=str(build_root),
        verbose=verbose,
    )
    return _EXT


def _count_selected(n: int, stride: int, phase: int) -> int:
    if phase >= n:
        return 0
    return 1 + (n - 1 - phase) // stride


class FusedStrideEFAdamW:
    """Minimal optimizer/state holder for the matched fused experiment."""
    def __init__(self, params: Iterable[torch.nn.Parameter], *, method: str,
                 lr: float = 3e-4, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01, stride: int = 8):
        if method not in ("external_fused", "dfc_fused"):
            raise ValueError(method)
        self.params = list(params)
        if not self.params:
            raise ValueError("no parameters")
        if any(p.dtype != torch.float16 or not p.is_cuda or not p.is_contiguous() for p in self.params):
            raise ValueError("fused optimizer requires contiguous CUDA float16 parameters")
        self.method = method
        self.lr = float(lr); self.betas = tuple(map(float, betas)); self.eps = float(eps)
        self.weight_decay = float(weight_decay); self.stride = int(stride); self.step_count = 0
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        self.m = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.v = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.residual = ([torch.zeros_like(p, dtype=torch.float32) for p in self.params]
                         if method == "external_fused" else None)
        self.backend = load_backend()

    @property
    def coordinates(self) -> int:
        return sum(p.numel() for p in self.params)

    @property
    def external_residual_bytes(self) -> int:
        return 4 * self.coordinates if self.residual is not None else 0

    @property
    def physical_state_bytes(self) -> int:
        return 8 * self.coordinates + self.external_residual_bytes

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self, phase: int | None = None) -> int:
        self.step_count += 1
        phase = (self.step_count - 1) % self.stride if phase is None else int(phase)
        if not 0 <= phase < self.stride:
            raise ValueError("invalid phase")
        sent = 0
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue
            if g.dtype != torch.float16 or not g.is_contiguous():
                raise ValueError("fused optimizer requires contiguous float16 gradients")
            if self.method == "external_fused":
                self.backend.external_step(p, g, self.m[i], self.v[i], self.residual[i],
                                           self.stride, phase, self.lr, b1, b2, self.eps,
                                           self.weight_decay, self.step_count)
            else:
                self.backend.dfc_step(p, g, self.m[i], self.v[i], self.stride, phase,
                                      self.lr, b1, b2, self.eps, self.weight_decay,
                                      self.step_count)
            sent += _count_selected(p.numel(), self.stride, phase)
        return sent

    @torch.no_grad()
    def semantic_moment_bits(self):
        mask = -65536
        return [(mi.view(torch.int32) & mask, vi.view(torch.int32) & mask)
                for mi, vi in zip(self.m, self.v)]

    @torch.no_grad()
    def logical_residuals(self):
        if self.residual is not None:
            return [r for r in self.residual]
        out = []
        for mi, vi in zip(self.m, self.v):
            mb = mi.view(torch.int32); vb = vi.view(torch.int32)
            bits = (mb & 65535) | ((vb & 65535) << 16)
            out.append(bits.view(torch.float32))
        return out

    def checkpoint_dict(self):
        return {
            "schema_version": 1, "method": self.method, "step": self.step_count,
            "m": self.m, "v": self.v, "residual": self.residual,
            "lr": self.lr, "betas": self.betas, "eps": self.eps,
            "weight_decay": self.weight_decay, "stride": self.stride,
        }
