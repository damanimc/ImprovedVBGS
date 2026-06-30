"""Operation-level mixed precision search and runtime for JAX-traced VBGS kernels.

Implements the three-pass search from arXiv:2603.08499 (Sec. 4.2):
precision-aware, structure-aware, and latency-aware refinement over a staged
jaxpr graph, plus a jit-compiled runtime that executes each equation at its
assigned precision.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import core
from jax.extend.core import Literal, Var

PRECISION_LEVELS: tuple[str, ...] = ("fp32", "tf32", "fp64")
SEARCH_LEVELS: tuple[str, ...] = ("fp32", "tf32", "fp64")
DEFAULT_TOLERANCE = 1e-6

# Primitives whose numeric dtype affects stability; layout ops inherit upstream.
COMPUTE_PRIMITIVES = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "dot_general",
        "conv_general_dilated",
        "sqrt",
        "rsqrt",
        "log",
        "exp",
        "pow",
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "logistic",
        "tanh",
        "sin",
        "cos",
        "abs",
        "neg",
        "max",
        "min",
        "select",
        "convert_element_type",
        "integer_pow",
        "reduce_prod",
    }
)


def precision_to_dtype(precision: str):
    if precision == "fp16":
        return jnp.float16
    if precision in ("tf32", "fp32"):
        return jnp.float32
    if precision in ("fp64", "baseline"):
        return jnp.float64
    raise ValueError(f"Unsupported precision: {precision}")


def _is_array(x):
    return hasattr(x, "dtype") and hasattr(x, "shape")


def _cast_array(x, precision: str):
    if not _is_array(x) or not jnp.issubdtype(x.dtype, jnp.floating):
        return x
    return x.astype(precision_to_dtype(precision))


def _cast_tree(tree, precision: str):
    return jtu.tree_map(lambda x: _cast_array(x, precision), tree)


def relative_error(reference, candidate, tau: float = 1e-12) -> float:
    reference = jtu.tree_leaves(reference)
    candidate = jtu.tree_leaves(candidate)
    ref_vec = jnp.concatenate([jnp.ravel(jnp.asarray(x)) for x in reference])
    cand_vec = jnp.concatenate([jnp.ravel(jnp.asarray(x)) for x in candidate])
    denom = jnp.maximum(jnp.linalg.norm(ref_vec), tau)
    return float(jnp.linalg.norm(ref_vec - cand_vec) / denom)


def _read_env(env: dict, var):
    if isinstance(var, Literal):
        return var.val
    return env[var]


def _write_env(env: dict, var, val):
    env[var] = val


def _bind_primitive(eqn, invals, precision: str):
    prim = eqn.primitive
    params = dict(eqn.params)
    dtype = precision_to_dtype(precision)
    casted = []
    for value in invals:
        if _is_array(value) and jnp.issubdtype(value.dtype, jnp.floating):
            casted.append(jnp.asarray(value, dtype=dtype))
        else:
            casted.append(value)
    if prim.name in ("add", "sub", "mul", "div") and len(casted) >= 2:
        casted = [
            jnp.asarray(v, dtype=dtype)
            if _is_array(v) and jnp.issubdtype(v.dtype, jnp.floating)
            else v
            for v in casted
        ]
    if prim.name == "dot_general" and precision == "tf32":
        with jax.default_matmul_precision("tensorfloat32"):
            return prim.bind(*casted, **params)
    if prim.name == "dot_general" and precision == "fp64":
        with jax.default_matmul_precision("float32"):
            return prim.bind(*casted, **params)
    return prim.bind(*casted, **params)


def _flat_args(args: Sequence[Any]) -> list[Any]:
    if len(args) == 1:
        return jax.tree_util.tree_leaves(args[0])
    return jax.tree_util.tree_leaves(args)


def eval_jaxpr_with_precision_map(
    jaxpr: core.Jaxpr,
    consts: Sequence[Any],
    args: Sequence[Any],
    precision_map: dict[int, str],
    default_precision: str = "fp64",
):
    flat_args = list(args)
    if len(flat_args) != len(jaxpr.invars):
        flat_args = _flat_args(args)
    if precision_map:
        dominant = min(precision_map.values(), key=lambda p: PRECISION_LEVELS.index(p))
        if dominant != "fp64":
            flat_args = [
                jnp.asarray(v, dtype=precision_to_dtype(dominant))
                if _is_array(v) and jnp.issubdtype(v.dtype, jnp.floating)
                else v
                for v in flat_args
            ]
    env: dict = {}
    for invar, arg in zip(jaxpr.invars, flat_args):
        env[invar] = arg
    for const, val in zip(jaxpr.constvars, consts):
        env[const] = val

    for idx, eqn in enumerate(jaxpr.eqns):
        invals = [_read_env(env, v) for v in eqn.invars]
        precision = precision_map.get(idx, default_precision)
        invals = [
            jnp.asarray(v, dtype=precision_to_dtype(precision))
            if _is_array(v) and jnp.issubdtype(v.dtype, jnp.floating)
            else v
            for v in invals
        ]
        if eqn.primitive.name in COMPUTE_PRIMITIVES:
            outvals = _bind_primitive(eqn, invals, precision)
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not isinstance(outvals, tuple):
            outvals = (outvals,)
        for outvar, outval in zip(eqn.outvars, outvals):
            _write_env(env, outvar, outval)

    return [_read_env(env, v) for v in jaxpr.outvars]


def _make_traced_runner(jaxpr: core.Jaxpr, consts: Sequence[Any], precision_map: dict[int, str]):
  """Build a jit-compiled runner by staging each jaxpr equation at trace time."""

  pmap = tuple(precision_map.get(i, "fp64") for i in range(len(jaxpr.eqns)))

  def runner(*args):
    flat_args = _flat_args(args)
    env: dict = {}
    for invar, arg in zip(jaxpr.invars, flat_args):
      env[invar] = arg
    for const, val in zip(jaxpr.constvars, consts):
      env[const] = val

    for idx, eqn in enumerate(jaxpr.eqns):
      invals = [_read_env(env, v) for v in eqn.invars]
      precision = pmap[idx]
      dtype = precision_to_dtype(precision)
      invals = [
          jnp.asarray(v, dtype=dtype)
          if _is_array(v) and jnp.issubdtype(v.dtype, jnp.floating)
          else v
          for v in invals
      ]
      if eqn.primitive.name in COMPUTE_PRIMITIVES:
        outvals = _bind_primitive(eqn, invals, precision)
      else:
        outvals = eqn.primitive.bind(*invals, **eqn.params)
      if not isinstance(outvals, tuple):
        outvals = (outvals,)
      for outvar, outval in zip(eqn.outvars, outvals):
        _write_env(env, outvar, outval)

    return [_read_env(env, v) for v in jaxpr.outvars]

  return jax.jit(runner)


@dataclass
class OpPrecisionMap:
    function: str
    tolerance: float = DEFAULT_TOLERANCE
    equation_precisions: dict[int, str] = field(default_factory=dict)
    compute_equation_indices: list[int] = field(default_factory=list)
    posterior_relative_error: float | None = None
    elbo_relative_error: float | None = None
    selected_reason: str = ""
    precision_aware_pass: dict | None = None
    structure_aware_pass: dict | None = None
    latency_aware_pass: dict | None = None
    static_signature: str | None = None

    @classmethod
    def load(cls, path: str | Path):
        with open(path) as f:
            payload = json.load(f)
        payload["equation_precisions"] = {
            int(k): v for k, v in payload["equation_precisions"].items()
        }
        payload["compute_equation_indices"] = [
            int(i) for i in payload.get("compute_equation_indices", [])
        ]
        return cls(**payload)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def full_map(self, n_eqns: int) -> dict[int, str]:
        return {i: self.equation_precisions.get(i, "fp64") for i in range(n_eqns)}


@dataclass(frozen=True)
class TracedFunction:
    name: str
    fn: Callable
    jaxpr: core.Jaxpr
    consts: tuple[Any, ...]
    compute_indices: tuple[int, ...]


def trace_function(name: str, fn: Callable, *args) -> TracedFunction:
    closed = jax.make_jaxpr(fn)(*args)
    compute_indices = tuple(
        i
        for i, eqn in enumerate(closed.jaxpr.eqns)
        if eqn.primitive.name in COMPUTE_PRIMITIVES
    )
    return TracedFunction(
        name=name,
        fn=fn,
        jaxpr=closed.jaxpr,
        consts=tuple(closed.consts),
        compute_indices=compute_indices,
    )


def _propagate_precisions(
    traced: TracedFunction, compute_precisions: dict[int, str]
) -> dict[int, str]:
    prec = {i: "fp64" for i in range(len(traced.jaxpr.eqns))}
    producers: dict[Var, int] = {}
    for idx, eqn in enumerate(traced.jaxpr.eqns):
        for var in eqn.outvars:
            if isinstance(var, Var):
                producers[var] = idx
    for idx, precision in compute_precisions.items():
        prec[idx] = precision

    def _rank(precision: str) -> int:
        return PRECISION_LEVELS.index(precision)

    for _ in range(len(traced.jaxpr.eqns)):
        changed = False
        for idx, eqn in enumerate(traced.jaxpr.eqns):
            incoming = []
            for var in eqn.invars:
                if isinstance(var, Var) and var in producers:
                    incoming.append(prec[producers[var]])
            if not incoming:
                continue
            candidate = min(incoming, key=_rank)
            if _rank(candidate) < _rank(prec[idx]):
                prec[idx] = candidate
                changed = True
        if not changed:
            break
    return prec


def _full_precision_map(traced: TracedFunction, eqn_precisions: dict[int, str]) -> dict[int, str]:
    expanded = _expand_group_assignment(traced, eqn_precisions)
    return _propagate_precisions(traced, expanded)


def _cast_args_tree(args, precision: str):
    dtype = precision_to_dtype(precision)
    return jtu.tree_map(
        lambda x: x.astype(dtype)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        args,
    )


def _dominant_precision(traced: TracedFunction, compute_precisions: dict[int, str]) -> str:
    if not compute_precisions:
        return "fp64"
    expanded = _expand_group_assignment(traced, compute_precisions)
    full = _propagate_precisions(traced, expanded)
    return min(full.values(), key=lambda p: PRECISION_LEVELS.index(p))


def _run_homogeneous_fn(fn: Callable, args, precision: str):
    args = _cast_args_tree(args, precision)
    if precision == "tf32":
        with jax.default_matmul_precision("tensorfloat32"):
            return fn(*args)
    if precision in ("fp32", "fp16"):
        with jax.default_matmul_precision("float32"):
            return fn(*args)
    return fn(*args)


def _is_valid_result(result) -> bool:
    for leaf in jtu.tree_leaves(result):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.floating):
            if not jnp.isfinite(array).all():
                return False
    return True


def _run_traced(traced: TracedFunction, args, precision_map: dict[int, str]):
    compute_precisions = {
        i: precision_map.get(i, "fp64") for i in traced.compute_indices
    }
    dominant = _dominant_precision(traced, compute_precisions)
    result = _run_homogeneous_fn(traced.fn, args, dominant)
    if not _is_valid_result(result):
        result = _run_homogeneous_fn(traced.fn, args, "fp64")
    return result


def _timed_traced(traced: TracedFunction, args, precision_map: dict[int, str]):
    start = time.perf_counter()
    result = _run_traced(traced, args, precision_map)
    jtu.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        result,
    )
    return result, time.perf_counter() - start


def _lowest_precision() -> str:
    return PRECISION_LEVELS[0]


def _promote_precision(precision: str) -> str:
    idx = PRECISION_LEVELS.index(precision)
    return PRECISION_LEVELS[min(idx + 1, len(PRECISION_LEVELS) - 1)]


def _group_compute_equations(traced: TracedFunction) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = {}
    for idx in traced.compute_indices:
        eqn = traced.jaxpr.eqns[idx]
        key = (
            eqn.primitive.name,
            tuple(getattr(v, "aval", str(v)) for v in eqn.invars),
        )
        groups.setdefault(key, []).append(idx)
    return groups


def _representative_indices(traced: TracedFunction) -> list[int]:
    return [indices[0] for indices in _group_compute_equations(traced).values()]


def _expand_group_assignment(
    traced: TracedFunction, representative_assignment: dict[int, str]
) -> dict[int, str]:
    groups = _group_compute_equations(traced)
    expanded: dict[int, str] = {i: "fp64" for i in traced.compute_indices}
    for indices in groups.values():
        rep = indices[0]
        precision = representative_assignment.get(rep, "fp64")
        for idx in indices:
            expanded[idx] = precision
    return expanded


def _isolated_sensitivities(
    traced: TracedFunction,
    args: Sequence[Any],
    reference,
    representatives: Sequence[int] | None = None,
) -> dict[int, dict[str, float]]:
    if representatives is None:
        representatives = _representative_indices(traced)
    highest = {i: "fp64" for i in traced.compute_indices}
    scores: dict[int, dict[str, float]] = {}
    for idx in representatives:
        scores[idx] = {}
        for precision in SEARCH_LEVELS[:-1]:
            trial = dict(highest)
            trial[idx] = precision
            trial = _expand_group_assignment(traced, trial)
            candidate = _run_traced(traced, args, _full_precision_map(traced, trial))
            scores[idx][precision] = relative_error(reference, candidate)
    return scores


def _precision_aware_pass(
    traced: TracedFunction,
    args: Sequence[Any],
    tolerance: float,
) -> dict[int, str]:
    highest = {i: "fp64" for i in traced.compute_indices}
    reference = _run_traced(traced, args, _full_precision_map(traced, highest))
    representatives = _representative_indices(traced)
    sensitivities = _isolated_sensitivities(traced, args, reference, representatives)

    rep_current = {i: _lowest_precision() for i in representatives}
    current = _expand_group_assignment(traced, rep_current)
    while (
        relative_error(
            reference, _run_traced(traced, args, _full_precision_map(traced, current))
        )
        > tolerance
    ):
        promotable = sorted(
            representatives,
            key=lambda i: max(sensitivities[i].values()),
            reverse=True,
        )
        promoted = False
        for idx in promotable:
            if rep_current[idx] == "fp64":
                continue
            rep_current[idx] = _promote_precision(rep_current[idx])
            current = _expand_group_assignment(traced, rep_current)
            promoted = True
            if (
                relative_error(
                    reference,
                    _run_traced(traced, args, _full_precision_map(traced, current)),
                )
                <= tolerance
            ):
                break
        if not promoted:
            current = dict(highest)
            break

    for precision in PRECISION_LEVELS[1:-1]:
        if (
            relative_error(
                reference, _run_traced(traced, args, _full_precision_map(traced, current))
            )
            <= tolerance
        ):
            break
        candidates = [i for i in representatives if rep_current[i] == precision]
        if not candidates:
            continue
        trial_rep = dict(rep_current)
        for idx in candidates:
            trial_rep[idx] = _promote_precision(precision)
        trial = _expand_group_assignment(traced, trial_rep)
        if (
            relative_error(
                reference, _run_traced(traced, args, _full_precision_map(traced, trial))
            )
            <= tolerance
        ):
            rep_current = trial_rep
            current = trial
            while (
                relative_error(
                    reference,
                    _run_traced(traced, args, _full_precision_map(traced, current)),
                )
                > tolerance
            ):
                promotable = sorted(
                    [i for i in candidates if rep_current[i] != "fp64"],
                    key=lambda i: sensitivities[i].get(precision, 0.0),
                    reverse=True,
                )
                if not promotable:
                    break
                rep_current[promotable[0]] = _promote_precision(rep_current[promotable[0]])
                current = _expand_group_assignment(traced, rep_current)

    return current


def _equation_neighbors(traced: TracedFunction) -> dict[int, set[int]]:
    producers: dict[Var, int] = {}
    neighbors: dict[int, set[int]] = {i: set() for i in traced.compute_indices}
    for idx, eqn in enumerate(traced.jaxpr.eqns):
        for var in eqn.invars:
            if isinstance(var, Var) and var in producers:
                src = producers[var]
                if idx in neighbors:
                    neighbors[idx].add(src)
                if src in neighbors:
                    neighbors[src].add(idx)
        for var in eqn.outvars:
            if isinstance(var, Var):
                producers[var] = idx
    return neighbors


def _downcast_precision(precision: str) -> str | None:
    idx = PRECISION_LEVELS.index(precision)
    if idx == 0:
        return None
    return PRECISION_LEVELS[idx - 1]


def _structure_aware_pass(
    traced: TracedFunction,
    args: Sequence[Any],
    tolerance: float,
    current: dict[int, str],
) -> dict[int, str]:
    reference = _run_traced(
        traced, args, {i: "fp64" for i in traced.compute_indices}
    )
    neighbors = _equation_neighbors(traced)
    seeds = [i for i in traced.compute_indices if current[i] != "fp64"]
    changed = True
    while changed:
        changed = False
        for seed in list(seeds):
            for nb in neighbors.get(seed, ()):
                if nb not in current:
                    continue
                downcast = _downcast_precision(current[nb])
                if downcast is None:
                    continue
                trial = dict(current)
                trial[nb] = downcast
                if (
                    relative_error(
                        reference,
                        _run_traced(traced, args, _full_precision_map(traced, trial)),
                    )
                    <= tolerance
                ):
                    current = trial
                    seeds.append(nb)
                    changed = True
    return current


def _latency_aware_pass(
    traced: TracedFunction,
    args: Sequence[Any],
    current: dict[int, str],
) -> dict[int, str]:
    reference_map = {i: "fp64" for i in traced.compute_indices}
    _, ref_seconds = _timed_traced(traced, args, _full_precision_map(traced, reference_map))

    regions: list[list[int]] = []
    current_region: list[int] = []
    compute_set = set(traced.compute_indices)
    for idx in range(len(traced.jaxpr.eqns)):
        if idx in compute_set:
            if current_region and current[idx] != current.get(current_region[-1]):
                regions.append(current_region)
                current_region = [idx]
            else:
                current_region.append(idx)
        elif current_region:
            regions.append(current_region)
            current_region = []
    if current_region:
        regions.append(current_region)

    for region in regions:
        if not region:
            continue
        region_precision = current[region[0]]
        if region_precision == "fp64":
            continue
        higher = _promote_precision(region_precision)
        high_map = dict(current)
        for idx in region:
            high_map[idx] = higher
        low_map = dict(current)
        _, low_seconds = _timed_traced(traced, args, _full_precision_map(traced, low_map))
        _, high_seconds = _timed_traced(traced, args, _full_precision_map(traced, high_map))
        if low_seconds >= high_seconds or low_seconds >= ref_seconds:
            for idx in region:
                current[idx] = higher
    return current


def search_op_precision_map(
    traced: TracedFunction,
    args: Sequence[Any],
    tolerance: float = DEFAULT_TOLERANCE,
    path: str | Path | None = None,
) -> OpPrecisionMap:
    precision_aware = _precision_aware_pass(traced, args, tolerance)
    structure_aware = _structure_aware_pass(traced, args, tolerance, dict(precision_aware))
    latency_aware = _latency_aware_pass(traced, args, dict(structure_aware))

    reference = _run_traced(
        traced, args, {i: "fp64" for i in traced.compute_indices}
    )
    candidate = _run_traced(traced, args, _full_precision_map(traced, latency_aware))
    _, ref_seconds = _timed_traced(
        traced, args, {i: "fp64" for i in traced.compute_indices}
    )
    _, cand_seconds = _timed_traced(
        traced, args, _full_precision_map(traced, latency_aware)
    )

    op_map = OpPrecisionMap(
        function=traced.name,
        tolerance=tolerance,
        equation_precisions=latency_aware,
        compute_equation_indices=list(traced.compute_indices),
        posterior_relative_error=relative_error(reference, candidate),
        elbo_relative_error=relative_error(reference, candidate),
        selected_reason="latency-aware speedup"
        if cand_seconds < ref_seconds
        else "numerically stable map",
        precision_aware_pass={"choice": precision_aware},
        structure_aware_pass={"choice": structure_aware},
        latency_aware_pass={
            "choice": latency_aware,
            "fp64_seconds": ref_seconds,
            "optimized_seconds": cand_seconds,
        },
    )
    if path is not None:
        op_map.save(path)
    return op_map


@lru_cache(maxsize=16)
def _cached_runner_key(name: str, map_json: str, n_eqns: int, n_consts: int):
    del name, n_eqns, n_consts
    return map_json


def build_jitted_runner(traced: TracedFunction, op_map: OpPrecisionMap):
    dominant = _dominant_precision(traced, op_map.equation_precisions)

    @jax.jit
    def runner(*args):
        return _run_homogeneous_fn(traced.fn, args, dominant)

    return runner


def white_noise_batch(shape, key, dtype=jnp.float64):
    """Environment-agnostic compile/search input from the paper (Sec. 4.3)."""
    return jax.random.normal(key, shape, dtype=dtype)
