# Copyright 2024 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/vbgs/blob/main/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import hydra

import copy
import os
import sys
import time

from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
PACKAGE_PATH = Path(__file__).resolve().parents[1]
if str(PACKAGE_PATH) not in sys.path:
    sys.path.append(str(PACKAGE_PATH))

import jax
import jax.random as jr

import vbgs
from vbgs.io.output import RunOutput
from vbgs.model.continual import (
    build_sparse_index,
    default_densify_stats,
    initialize_mean,
    query_candidate_indices,
    reset_component_stats,
    strict_unused_count,
)
from vbgs.model.densify import densify_from_frame
from vbgs.model.train import fit_gmm_step
from vbgs.data.utils import create_normalizing_params, normalize_data
from vbgs.data.blender import BlenderDataIterator

from model_volume import get_volume_delta_mixture


# Densify stats are numpy scalars; cast the ones we serialize so the metrics
# JSON stays plain Python. Anything not listed is passed through unchanged.
_FRAME_INT_KEYS = (
    "available_components",
    "strict_unused_components",
    "component_offset",
    "densify_points",
    "densified_components",
    "unseen_points",
)


def frame_record(step, x, *, densify_stats, **timings):
    record = {"frame": int(step), "points": int(x.shape[0]), **timings}
    for key in _FRAME_INT_KEYS:
        record[key] = int(densify_stats[key])
    record["unseen_fraction"] = float(densify_stats["unseen_fraction"])
    record["recycle_mode"] = bool(densify_stats["recycle_mode"])
    for key in (
        "densify_distance_threshold",
        "densify_distance_mean",
        "unseen_distance_threshold",
        "densify_skipped_reason",
        "reassign_fraction",
    ):
        record[key] = densify_stats[key]
    return record


def fit_continual(
    data_path,
    n_components,
    subsample=None,
    init_random=False,
    key=None,
    batch_size=5000,
    top_m=None,
    candidate_m=None,
    save_every=None,
    precision="fp64",
    init_first_frame=False,
    init_downsample_factor=None,
    densify=False,
    densify_point_ratio=0.0,
    densify_unseen_distance_threshold=None,
    densify_min_unseen_fraction=0.0,
    densify_min_unseen_points=0,
    densify_reassign_if_full=False,
    reassign_fraction=0.05,
    output=None,
):
    if key is None:
        key = jr.PRNGKey(0)
    output = output or RunOutput.create(output_dir=Path.cwd())
    densify_point_ratio = float(densify_point_ratio or 0.0)
    if densify_unseen_distance_threshold is not None:
        densify_unseen_distance_threshold = float(densify_unseen_distance_threshold)
    densify_min_unseen_fraction = float(densify_min_unseen_fraction or 0.0)
    densify_min_unseen_points = int(densify_min_unseen_points or 0)
    densify_reassign_if_full = bool(densify_reassign_if_full)
    reassign_fraction = float(reassign_fraction)
    if reassign_fraction < 0:
        raise ValueError("reassign_fraction must be non-negative")
    if init_downsample_factor is not None:
        init_downsample_factor = int(init_downsample_factor)
        if init_downsample_factor <= 0:
            raise ValueError("init_downsample_factor must be positive or null")

    data_iter = BlenderDataIterator(
        data_path,
        data_params=None,
        subsample=subsample,
    )
    if init_first_frame:
        x_data = None
        data_params = create_normalizing_params(
            [-1, 1], [-1, 1], [-1, 1], [0, 1], [0, 1], [0, 1]
        )
        init_iter = BlenderDataIterator(
            data_path,
            data_params=data_params,
            subsample=subsample,
        )
        x_data = init_iter._get_frame(0)
        if init_downsample_factor is not None and init_downsample_factor > 1:
            x_data = x_data[::init_downsample_factor]
        init_random = False
    elif not init_random:
        # Essentially, if not init random, we load n_components points from the
        # point cloud, to initialize the model components on. Then we can just
        # do either the continual or non continual learning scheme with this
        # script. Note that in a real continual setting, init on data won't be
        # possible, hence we need a proper create_normalizing params
        data = np.zeros((0, 6))
        for d in data_iter:
            if isinstance(d, tuple):
                d = d[0]
            data = np.concatenate([data, d])
        data, data_params = normalize_data(data)

        np.random.seed(0)
        idcs = np.arange(data.shape[0])
        np.random.shuffle(idcs)
        x_data = data[idcs[:n_components]]
        del data

    else:
        x_data = None
        data_params = create_normalizing_params(
            [-1, 1], [-1, 1], [-1, 1], [0, 1], [0, 1], [0, 1]
        )

    data_iter = BlenderDataIterator(
        data_path,
        data_params=data_params,
        subsample=subsample,
    )

    key, subkey = jr.split(key)
    mean_init, init_metrics = initialize_mean(
        key=subkey,
        init_data=x_data,
        n_components=n_components,
        event_shape=(6, 1),
        init_random=init_random,
        init_first_frame=init_first_frame,
        add_noise=not init_first_frame,
    )
    del x_data

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

    model = copy.deepcopy(prior_model)

    candidate_tree, topm_cache = build_sparse_index(
        prior_model,
        top_m=top_m,
        candidate_m=candidate_m,
        precision=precision,
    )
    if save_every is not None:
        save_every = int(save_every)
        if save_every <= 0:
            raise ValueError("save_every must be positive or null")

    metrics = {
        "components": int(n_components),
        "batch_size": int(batch_size),
        "top_m": top_m,
        "candidate_m": candidate_m,
        "precision": precision,
        "init_first_frame": bool(init_first_frame),
        "init_downsample_factor": init_downsample_factor,
        **init_metrics,
        "densify_point_ratio": float(densify_point_ratio),
        "densify_unseen_distance_threshold": densify_unseen_distance_threshold,
        "densify_min_unseen_fraction": densify_min_unseen_fraction,
        "densify_min_unseen_points": densify_min_unseen_points,
        "densify_reassign_if_full": densify_reassign_if_full,
        "reassign_fraction": reassign_fraction,
        "densify": bool(densify),
        "frames": [],
    }
    prior_stats, space_stats, color_stats = None, None, None
    for step, x in tqdm(enumerate(data_iter), total=len(data_iter)):
        start = time.perf_counter()
        densify_seconds = 0.0
        sparse_rebuild_seconds = 0.0
        densify_stats = default_densify_stats(densify_unseen_distance_threshold)
        if densify and not (init_first_frame and step == 0):
            densify_start = time.perf_counter()
            available = strict_unused_count(model, prior_model)
            densify_stats["available_components"] = available
            densify_stats["strict_unused_components"] = available
            if available > 0 or densify_reassign_if_full:
                densify_candidates = query_candidate_indices(
                    candidate_tree,
                    x[:, :3],
                    candidate_m,
                    n_components,
                )
                prior_model, densify_stats, inserted_components = densify_from_frame(
                    prior_model,
                    model,
                    x,
                    candidate_indices=densify_candidates,
                    point_ratio=densify_point_ratio if densify_point_ratio > 0 else None,
                    unseen_distance_threshold=densify_unseen_distance_threshold,
                    min_unseen_fraction=densify_min_unseen_fraction,
                    min_unseen_points=densify_min_unseen_points,
                    recycle_if_full=densify_reassign_if_full,
                    reassign_fraction=reassign_fraction,
                    debug=True,
                    return_indices=True,
                )
                prior_stats = reset_component_stats(prior_stats, inserted_components)
                space_stats = reset_component_stats(space_stats, inserted_components)
                color_stats = reset_component_stats(color_stats, inserted_components)
            densify_seconds = time.perf_counter() - densify_start
            if densify_stats["densified_components"] > 0:
                rebuild_start = time.perf_counter()
                candidate_tree, topm_cache = build_sparse_index(
                    prior_model,
                    top_m=top_m,
                    candidate_m=candidate_m,
                    precision=precision,
                )
                sparse_rebuild_seconds = time.perf_counter() - rebuild_start
        candidate_indices = None
        kdtree_seconds = 0.0
        if candidate_tree is not None:
            tk = time.perf_counter()
            candidate_indices = query_candidate_indices(
                candidate_tree,
                x[:, :3],
                candidate_m,
                n_components,
            )
            kdtree_seconds = time.perf_counter() - tk
        fit_start = time.perf_counter()
        model, prior_stats, space_stats, color_stats, _semantic_stats = fit_gmm_step(
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
        jax.block_until_ready(model.mixture.likelihood.mean)
        fit_seconds = time.perf_counter() - fit_start
        metrics["frames"].append(
            frame_record(
                step,
                x,
                seconds=time.perf_counter() - start,
                fit_seconds=fit_seconds,
                densify_seconds=densify_seconds,
                kdtree_seconds=kdtree_seconds,
                sparse_rebuild_seconds=sparse_rebuild_seconds,
                densify_stats=densify_stats,
            )
        )

        if save_every is not None and (step + 1) % save_every == 0:
            output.checkpoint(model, data_params, f"model_{step:02d}.json")

    output.final_model(model, data_params, metrics)

    return metrics


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="blender",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    jax.config.update("jax_default_device", jax.devices()[int(cfg.device)])
    output = RunOutput.create(output_dir=Path.cwd())

    root_path = Path(vbgs.__file__).parent.parent

    _, subkey = jr.split(jr.PRNGKey(0))
    results = fit_continual(
        key=subkey,
        n_components=cfg.model.n_components,
        data_path=root_path / Path(cfg.data.data_path),
        subsample=cfg.data.subsample_factor,
        init_random=cfg.model.init_random,
        batch_size=cfg.train.batch_size,
        top_m=getattr(cfg.train, "top_m", None),
        candidate_m=getattr(cfg.train, "candidate_m", None),
        save_every=getattr(cfg.train, "save_every", None),
        precision=getattr(cfg.train, "precision", "fp64"),
        init_first_frame=getattr(cfg.model, "init_first_frame", False),
        init_downsample_factor=getattr(cfg.model, "init_downsample_factor", None),
        densify=getattr(cfg.train, "densify", False),
        densify_point_ratio=getattr(cfg.train, "densify_point_ratio", 0.0),
        densify_unseen_distance_threshold=getattr(
            cfg.train,
            "densify_unseen_distance_threshold",
            None,
        ),
        densify_min_unseen_fraction=getattr(
            cfg.train,
            "densify_min_unseen_fraction",
            0.0,
        ),
        densify_min_unseen_points=getattr(cfg.train, "densify_min_unseen_points", 0),
        densify_reassign_if_full=getattr(cfg.train, "densify_reassign_if_full", False),
        reassign_fraction=getattr(cfg.train, "reassign_fraction", 0.05),
        output=output,
    )
    results.update({"config": OmegaConf.to_container(cfg)})

    output.metrics(results)


if __name__ == "__main__":
    main()
