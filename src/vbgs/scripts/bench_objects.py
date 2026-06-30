import time
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
PACKAGE_PATH = Path(__file__).resolve().parents[1]
if str(PACKAGE_PATH) not in sys.path:
    sys.path.append(str(PACKAGE_PATH))

import jax
import jax.random as jr

import vbgs
from vbgs.io.output import RunOutput
from vbgs.model.continual import build_sparse_index, query_candidate_indices
from vbgs.model.utils import random_mean_init
from vbgs.model.train import fit_gmm_step
from vbgs.data.utils import create_normalizing_params
from vbgs.data.blender import BlenderDataIterator

from model_volume import get_volume_delta_mixture

def run(data_path, n_components, batch_size, top_m, candidate_m, key, precision):
    data_params = create_normalizing_params(
        [-1, 1], [-1, 1], [-1, 1], [0, 1], [0, 1], [0, 1]
    )

    data_iter = BlenderDataIterator(
        data_path, data_params=data_params, subsample=None
    )

    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        key=subkey,
        x=None,
        component_shape=(n_components,),
        event_shape=(6, 1),
        init_random=True,
        add_noise=False,
    )

    key, subkey = jr.split(key)
    prior_model = get_volume_delta_mixture(
        key=subkey,
        n_components=n_components,
        mean_init=mean_init,
        beta=0,
        learning_rate=1,
        dof_offset=1,
        position_scale=n_components,
        position_event_shape=(3, 1),
    )

    import copy

    model = copy.deepcopy(prior_model)

    candidate_tree, topm_cache = build_sparse_index(
        prior_model,
        top_m=top_m,
        candidate_m=candidate_m,
        precision=precision,
    )

    prior_stats, space_stats, color_stats = None, None, None

    per_frame = []
    kdtree_time = 0.0
    t_loop = time.perf_counter()
    for step, x in tqdm(enumerate(data_iter), total=len(data_iter)):
        candidate_indices = None
        frame_kdtree_time = 0.0
        if candidate_tree is not None:
            tk = time.perf_counter()
            candidate_indices = query_candidate_indices(
                candidate_tree,
                x[:, :3],
                candidate_m,
                n_components,
            )
            frame_kdtree_time = time.perf_counter() - tk
            kdtree_time += frame_kdtree_time

        t0 = time.perf_counter()
        model, prior_stats, space_stats, color_stats, _ = fit_gmm_step(
            prior_model,
            model,
            data=x,
            batch_size=batch_size,
            prior_stats=prior_stats,
            space_stats=space_stats,
            color_stats=color_stats,
            top_m=top_m,
            candidate_indices=candidate_indices,
            topm_cache=topm_cache,
            precision=precision,
        )
        # block to get true per-frame compute time
        jax.block_until_ready(model.mixture.likelihood.mean)
        fit_seconds = time.perf_counter() - t0
        per_frame.append(fit_seconds)

    total = time.perf_counter() - t_loop
    return total, per_frame, kdtree_time, model, data_params


def main():
    root = Path(vbgs.__file__).parent.parent
    data_path = root / "../../data/blender/lego"

    n_components = 100_000
    batch_size = 100_000
    top_m = 32
    candidate_m = 128
    precision = "fp64"

    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)

    total, per_frame, kdtree_time, model, data_params = run(
        data_path, n_components, batch_size, top_m, candidate_m, subkey, precision
    )

    output = RunOutput.create(run_name="lego-bench", output_root=root / "output", unique=True)
    t_save = time.perf_counter()
    metrics = {
        "data": str(data_path),
        "components": n_components,
        "batch_size": batch_size,
        "top_m": top_m,
        "candidate_m": candidate_m,
        "precision": precision,
        "frames": len(per_frame),
        "total_seconds": total,
        "kdtree_seconds": kdtree_time,
        "fit_seconds": float(np.sum(per_frame)),
        "mean_fit_seconds": float(np.mean(per_frame)),
        "median_fit_seconds": float(np.median(per_frame)),
    }
    output.final_model(model, data_params, metrics)
    output.metrics(metrics)
    save_time = time.perf_counter() - t_save

    pf = np.array(per_frame)
    print("\n=== BENCH (no json store) ===")
    print(f"frames           : {len(pf)}")
    print(f"total loop time  : {total:.2f} s")
    print(f"kdtree query tot : {kdtree_time:.2f} s")
    print(f"fit_gmm_step tot : {pf.sum():.2f} s")
    print(f"per-frame fit    : mean {pf.mean()*1e3:.1f} ms  "
          f"median {np.median(pf)*1e3:.1f} ms  "
          f"min {pf.min()*1e3:.1f} ms  max {pf.max()*1e3:.1f} ms")
    print(f"frame 0 (compile): {pf[0]*1e3:.1f} ms")
    if len(pf) > 1:
        print(f"steady (skip f0) : mean {pf[1:].mean()*1e3:.1f} ms")
    print(f"final model saved: {output.path / 'model_final.json'}  "
          f"(store took {save_time:.2f} s)")


if __name__ == "__main__":
    main()
