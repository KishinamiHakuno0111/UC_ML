"""Bounded shared fields and the historical nine-condition perturbation formula.

Supply the frozen bus groups, zero-bus weights and source split. This module does
not discover groups, choose seeds based on accuracy, solve UC, or select labels.
"""

import numpy as np
from scipy.special import ndtr


def rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def ar1(generator, samples, channels):
    value = generator.standard_normal((samples, 36, channels))
    for t in range(1, 36):
        value[:, t] = .9 * value[:, t-1] + np.sqrt(1 - .9**2) * value[:, t]
    return value


def field(generator, samples, groups):
    groups = np.asarray(groups, dtype=int)
    common = ar1(generator, samples, 1)
    regional = ar1(generator, samples, int(groups.max()) + 1)
    local = ar1(generator, samples, len(groups))
    value = np.sqrt(.1)*common + np.sqrt(.4)*regional[:, :, groups] + np.sqrt(.5)*local
    return 2*ndtr(value)-1


def system_field(generator, samples):
    return (2*ndtr(ar1(generator, samples, 1))-1)[:, :, 0]


def project(q, weights, lower=-.05, upper=.05, target=0.):
    q, weights = np.broadcast_arrays(np.asarray(q, float), np.asarray(weights, float))
    if np.any(weights < 0) or np.any(weights.sum(-1) <= 0) or not lower <= target <= upper:
        raise ValueError("Invalid bounded projection")
    weights = weights / weights.sum(-1, keepdims=True)
    left, right = np.min(q-upper, axis=-1), np.max(q-lower, axis=-1)
    lam = np.clip(np.sum(weights*q, -1)-target, left, right)
    for _ in range(80):
        shifted = q-lam[..., None]
        answer = np.clip(shifted, lower, upper)
        residual = np.sum(weights*answer, -1)-target
        done = np.abs(residual) <= 2e-15
        if np.all(done):
            return answer
        left = np.where((residual > 0) & ~done, lam, left)
        right = np.where((residual < 0) & ~done, lam, right)
        slope = np.sum(weights*((shifted > lower) & (shifted < upper)), -1)
        candidate = lam + residual/np.maximum(slope, 1e-300)
        candidate = np.where((candidate > left) & (candidate < right), candidate, (left+right)/2)
        lam = np.where(done, lam, candidate)
    raise RuntimeError("Projection did not converge")


def layout(seed, zero_count):
    value = rng(seed, 0).uniform(.5, 1.5, zero_count)/zero_count
    return project(value, np.ones(zero_count), .5/zero_count, 1.5/zero_count, 1/zero_count)


def perturb(base_mw, factor, a_sign, b_sign, active, zero, weights, zero_field, system, eta):
    """Reuse the SAME frozen fields across the nine cells and amplitudes.

    a_sign/b_sign are -1, 0, +1; delta=0.05*factor is the total-load
    mean-shift magnitude. Zero factor is exactly the source, including zero buses.
    """
    if a_sign not in (-1, 0, 1) or b_sign not in (-1, 0, 1) or not 0 <= factor <= 8:
        raise ValueError("Invalid cell or amplitude")
    base = np.asarray(base_mw, dtype=np.float64)
    active, zero = np.asarray(active), np.asarray(zero)
    if not np.array_equal(np.sort(np.concatenate([active, zero])), np.arange(base.shape[-1])):
        raise ValueError("Active and zero nodes must partition the buses")
    if np.any(base[:, :, zero] != 0) or np.any(base[:, :, active] <= 0):
        raise ValueError("Source active/zero partition changed")
    if (np.any(np.asarray(weights) < .5/len(zero)-1e-12)
            or np.any(np.asarray(weights) > 1.5/len(zero)+1e-12)
            or not np.isclose(np.sum(weights), 1, atol=1e-12, rtol=0)
            or np.max(abs(zero_field)) > 1 or np.max(abs(system)) > 1 or np.max(abs(eta)) > .05+1e-9):
        raise ValueError("Field or weight bounds violated")
    if np.max(abs((eta*base[:, :, active]).sum(-1))) > 1e-6:
        raise ValueError("eta is not load-weighted zero mean")
    total = base.sum(-1)
    coefficient = np.asarray(weights)*factor*(.025*b_sign + .05*zero_field)
    correction = .05*factor*a_sign + .01*factor*system - coefficient.sum(-1)
    result = np.empty_like(base)
    result[:, :, zero] = total[:, :, None]*coefficient
    result[:, :, active] = base[:, :, active]*(1+correction[:, :, None]+factor*eta)
    expected = total*(1+.05*factor*a_sign+.01*factor*system)
    if not np.isfinite(result).all() or np.max(abs(result.sum(-1)-expected)) > 1e-6:
        raise ValueError("Input formula balance failed")
    if factor <= 5 and np.any(result[:, :, active] < 0):
        raise ValueError("Negative original-load node in the nonnegative domain")
    return result
