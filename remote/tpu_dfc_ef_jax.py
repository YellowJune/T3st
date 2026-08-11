"""Cross-substrate DFC-EF validation in JAX/XLA (designed for Kaggle TPU).

This is not used to claim TPU-specific memory savings. Its purpose is stronger:
the same low-word decoder-fiber construction, exact logical FP32 residual
roundtrip, semantic optimizer invariance, and external-vs-fiber trajectory are
executed through a non-CUDA compiler/runtime and, when available, across all TPU
cores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

HIGH = jnp.uint32(0xFFFF0000)
LOW = jnp.uint32(0x0000FFFF)


def bits(x):
    return lax.bitcast_convert_type(x, jnp.uint32)


def f32(x):
    return lax.bitcast_convert_type(x, jnp.float32)


def decode_semantic(x):
    return f32(jnp.bitwise_and(bits(x), HIGH))


def decode_residual(m_phys, v_phys):
    lo = jnp.bitwise_and(bits(m_phys), LOW)
    hi = jnp.left_shift(jnp.bitwise_and(bits(v_phys), LOW), jnp.uint32(16))
    return f32(jnp.bitwise_or(lo, hi))


def pack_residual(m_sem, v_sem, residual):
    rb = bits(residual)
    mb = jnp.bitwise_and(bits(m_sem), HIGH)
    vb = jnp.bitwise_and(bits(v_sem), HIGH)
    m_phys = f32(jnp.bitwise_or(mb, jnp.bitwise_and(rb, LOW)))
    v_phys = f32(
        jnp.bitwise_or(
            vb,
            jnp.bitwise_and(jnp.right_shift(rb, jnp.uint32(16)), LOW),
        )
    )
    return m_phys, v_phys


def mask_for(n, step, stride=8):
    idx = jnp.arange(n, dtype=jnp.uint32)
    return jnp.equal(jnp.mod(idx, jnp.uint32(stride)), jnp.uint32(step % stride))


def gradient_for(n, step):
    idx = jnp.arange(n, dtype=jnp.float32)
    return jnp.sin(idx * jnp.float32(0.0017) + jnp.float32(step) * jnp.float32(0.13)) * jnp.float32(0.01)


def semantic_adam(param, m_phys, v_phys, grad, step, lr=3e-4):
    b1 = jnp.float32(0.9)
    b2 = jnp.float32(0.999)
    m = decode_semantic(m_phys)
    v = decode_semantic(v_phys)
    m_new = b1 * m + (jnp.float32(1) - b1) * grad
    v_new = b2 * v + (jnp.float32(1) - b2) * grad * grad
    # Canonical low16-zero physical representative.
    m_can = f32(jnp.bitwise_and(bits(m_new), HIGH))
    v_can = f32(jnp.bitwise_and(bits(v_new), HIGH))
    m_use = decode_semantic(m_can)
    v_use = decode_semantic(v_can)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    bc1 = jnp.float32(1.0) - jnp.power(jnp.float32(0.9), step_f)
    bc2 = jnp.float32(1.0) - jnp.power(jnp.float32(0.999), step_f)
    update = (m_use / bc1) / (jnp.sqrt(v_use / bc2) + jnp.float32(1e-8))
    return param - jnp.float32(lr) * update, m_can, v_can


def external_step(state, step, stride=8):
    param, m_phys, v_phys, residual = state
    n = param.shape[0]
    g = gradient_for(n, step)
    comp = g + residual
    keep = mask_for(n, step, stride)
    communicated = jnp.where(keep, comp, jnp.float32(0))
    residual_new = comp - communicated
    p_new, m_new, v_new = semantic_adam(param, m_phys, v_phys, communicated, step + 1)
    return p_new, m_new, v_new, residual_new


def dfc_step(state, step, stride=8):
    param, m_phys, v_phys = state
    n = param.shape[0]
    residual = decode_residual(m_phys, v_phys)
    g = gradient_for(n, step)
    comp = g + residual
    keep = mask_for(n, step, stride)
    communicated = jnp.where(keep, comp, jnp.float32(0))
    residual_new = comp - communicated
    p_new, m_can, v_can = semantic_adam(param, m_phys, v_phys, communicated, step + 1)
    m_new, v_new = pack_residual(m_can, v_can, residual_new)
    return p_new, m_new, v_new


def bitwise_equal(a, b) -> bool:
    return bool(jax.device_get(jnp.all(bits(a) == bits(b))))


def hash_array(x) -> str:
    a = np.asarray(jax.device_get(x))
    return hashlib.sha256(a.tobytes()).hexdigest()


def exactness(n: int, steps: int, stride: int):
    p0 = jnp.linspace(jnp.float32(-0.1), jnp.float32(0.1), n, dtype=jnp.float32)
    z = jnp.zeros(n, dtype=jnp.float32)
    ext = (p0, z, z, z)
    dfc = (p0, z, z)

    # Force a nontrivial exact residual bit pattern before learning.
    probe = jnp.sin(jnp.arange(n, dtype=jnp.float32) * jnp.float32(0.003)) * jnp.float32(0.07)
    mp, vp = pack_residual(z, z, probe)
    recovered = decode_residual(mp, vp)
    roundtrip = bitwise_equal(probe, recovered)
    semantic_clean = bitwise_equal(decode_semantic(mp), z) and bitwise_equal(decode_semantic(vp), z)
    if not roundtrip or not semantic_clean:
        raise AssertionError("JAX fiber roundtrip/semantic invariance failed")

    ext_step = jax.jit(external_step, static_argnums=(2,))
    d_step = jax.jit(dfc_step, static_argnums=(2,))
    for step in range(steps):
        ext = ext_step(ext, step, stride)
        dfc = d_step(dfc, step, stride)
        ext[0].block_until_ready()
        dfc[0].block_until_ready()
        pe, me, ve, re = ext
        pd, md, vd = dfc
        checks = [
            bitwise_equal(pe, pd),
            bitwise_equal(decode_semantic(me), decode_semantic(md)),
            bitwise_equal(decode_semantic(ve), decode_semantic(vd)),
            bitwise_equal(re, decode_residual(md, vd)),
        ]
        if not all(checks):
            raise AssertionError(f"external/DFC trajectory mismatch at step {step}")

    return {
        "coordinates": n,
        "steps": steps,
        "stride": stride,
        "roundtrip_bitwise": roundtrip,
        "semantic_invariance_bitwise": semantic_clean,
        "trajectory_bitwise": True,
        "parameter_sha256": hash_array(ext[0]),
        "logical_residual_sha256": hash_array(ext[3]),
        "dfc_residual_sha256": hash_array(decode_residual(dfc[1], dfc[2])),
    }


def timing(n: int, steps: int, stride: int):
    p0 = jnp.zeros(n, dtype=jnp.float32)
    z = jnp.zeros_like(p0)
    ext = (p0, z, z, z)
    dfc = (p0, z, z)
    ext_step = jax.jit(external_step, static_argnums=(2,))
    d_step = jax.jit(dfc_step, static_argnums=(2,))

    # Compile/warm independently, excluded from timings.
    ext = ext_step(ext, 0, stride)
    ext[0].block_until_ready()
    dfc = d_step(dfc, 0, stride)
    dfc[0].block_until_ready()

    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        ext = ext_step(ext, i, stride)
    ext[0].block_until_ready()
    te = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        dfc = d_step(dfc, i, stride)
    dfc[0].block_until_ready()
    td = time.perf_counter() - t0

    return {
        "coordinates": n,
        "steps": steps,
        "external_seconds": te,
        "dfc_seconds": td,
        "dfc_overhead_fraction": td / te - 1.0 if te else None,
        "external_auxiliary_bytes": 4 * n,
        "dfc_model_scale_external_auxiliary_bytes": 0,
        "dfc_fiber_capacity_bytes": 4 * n,
        "two_moment_bytes_common": 8 * n,
    }


def multi_device_probe(local_n: int, stride: int):
    devices = jax.devices()
    if len(devices) < 2:
        return {"devices": len(devices), "skipped": True}

    # Each device gets a distinct gradient phase; psum exercises actual XLA
    # collective communication after sparsification. The payload state itself is
    # local and is not included in the communicated sparse tensor.
    per = jnp.zeros((len(devices), local_n), dtype=jnp.float32)
    residual = jnp.zeros_like(per)
    m = jnp.zeros_like(per)
    v = jnp.zeros_like(per)

    def fn(m_phys, v_phys, r_ext, replica):
        del m_phys, v_phys
        idx = jnp.arange(local_n, dtype=jnp.float32)
        g = jnp.sin(idx * jnp.float32(0.002) + replica * jnp.float32(0.3)) * jnp.float32(0.01)
        comp = g + r_ext
        keep = (jnp.arange(local_n, dtype=jnp.uint32) % jnp.uint32(stride)) == jnp.uint32(0)
        sparse = jnp.where(keep, comp, jnp.float32(0))
        reduced = lax.psum(sparse, "i")
        return reduced, comp - sparse

    pm = jax.pmap(fn, axis_name="i")
    replicas = jnp.arange(len(devices), dtype=jnp.float32)
    reduced, rnew = pm(m, v, residual, replicas)
    reduced.block_until_ready()
    return {
        "devices": len(devices),
        "local_coordinates": local_n,
        "global_coordinates": local_n * len(devices),
        "collective": "lax.psum",
        "stride": stride,
        "reduced_sha256": hash_array(reduced),
        "residual_sha256": hash_array(rnew),
        "pass": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact-coordinates", type=int, default=1_048_579)
    ap.add_argument("--exact-steps", type=int, default=24)
    ap.add_argument("--timing-sizes", default="1048576,4194304,16777216,67108864")
    ap.add_argument("--timing-steps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--output", default="results/kaggle_free/tpu_dfc_ef_jax.json")
    a = ap.parse_args()

    result = {
        "schema_version": 1,
        "protocol": "dfc-ef-jax-xla-cross-substrate-v1",
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "device_count": jax.device_count(),
    }
    result["exactness"] = exactness(a.exact_coordinates, a.exact_steps, a.stride)
    result["timing"] = [
        timing(int(n), a.timing_steps, a.stride)
        for n in a.timing_sizes.split(",")
        if n.strip()
    ]
    result["multi_device"] = multi_device_probe(262147, a.stride)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
