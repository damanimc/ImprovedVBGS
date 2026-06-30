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

import hydra
from omegaconf import DictConfig, OmegaConf

import json
import copy
import rich
import sys

from pathlib import Path
from tqdm import tqdm

import numpy as np
import random

import jax
import jax.numpy as jnp
import jax.random as jr

PACKAGE_PATH = Path(__file__).resolve().parents[1]
if str(PACKAGE_PATH) not in sys.path:
    sys.path.append(str(PACKAGE_PATH))

import vbgs
from vbgs.io.output import RunOutput
from vbgs.data.utils import create_normalizing_params, normalize_data
from vbgs.model.utils import random_mean_init
from vbgs.model.train import fit_gmm_step

from vbgs.data.habitat import HabitatDataIterator
from vbgs.model.reassign import reassign
from vbgs.model.precision_training import ensure_op_precision_runtime
from vbgs.model.precision import PrecisionMap


from model_volume import get_volume_delta_mixture

def fit_continual(
    data_path,
    data_index,
    n_components,
    init_random=False,
    key=None,
    eval_every=1,
    batch_size=5000,
    use_reassign=True,
    reassign_fraction=0.05,
    shuffle=False,
    n_frames=None,
    seed=0,
    precision_mode="op",
    precision_tolerance=1e-6,
    precision_search_frame=1,
    output=None,
):
    np.random.seed(seed)
    random.seed(seed)

    if key is None:
        key = jr.PRNGKey(0)
    output = output or RunOutput.create(output_dir=Path.cwd())

    if use_reassign:
        def reassign_fn(prior, model, data, batch_size, fraction, precision="fp64"):
            return reassign(
                prior, model, data, batch_size, fraction, precision=precision
            )
    else:
        reassign_fn = lambda prior, model, *args, **kwargs: prior

    # ============
    # Some subsampling
    data_iter = HabitatDataIterator(data_path, data_index, None)

    idcs = np.arange(len(data_iter._frames))
    if shuffle:
        idcs = np.random.choice(idcs, n_frames)
    else:
        time_subsample = 1
        if n_frames is not None:
            time_subsample = len(data_iter) // n_frames
        idcs = idcs[::time_subsample][:n_frames]

    with open("train_idcs.json", "w") as f:
        json.dump({"idcs": idcs.tolist()}, f, indent=2)

    data_iter._frames = [data_iter._frames[i] for i in idcs]

    x_data = None
    data_params = None
    if not init_random:
        all_data = []
        for frame in data_iter:
            all_data.append(frame)

        x_data = jnp.concatenate(all_data, axis=0)

        x_data, data_params = normalize_data(x_data, None)
        print("Normalizing data parameters: ")
        print(data_params, end="\n\n")

    if data_params is None:
        data_params = create_normalizing_params(
            [-5, 5], [-5, 5], [-5, 5], [0, 1], [0, 1], [0, 1]
        )
    data_iter = HabitatDataIterator(data_path, data_index, data_params)

    # Subsample the selected indices
    data_iter._frames = [data_iter._frames[i] for i in idcs]

    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        key=subkey,
        x=x_data,
        component_shape=(n_components,),
        event_shape=(6, 1),
        init_random=init_random,
        add_noise=True,
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

    metrics = dict({})
    prior_stats, space_stats, color_stats = None, None, None
    precision = "fp64" if precision_mode == "auto" else precision_mode
    precision_map = PrecisionMap(tolerance=precision_tolerance)
    for step, x in tqdm(enumerate(data_iter), total=len(data_iter)):
        if precision_mode == "op" or (
            precision_mode == "auto" and step >= precision_search_frame
        ):
            precision, precision_map = ensure_op_precision_runtime(
                mode=precision_mode,
                frame_idx=step,
                search_frame=precision_search_frame,
                prior_model=prior_model,
                normalized=x,
                batch_size=batch_size,
                output_dir=output.path,
                tolerance=precision_tolerance,
                white_noise_key=jr.PRNGKey(seed + 1),
            )
        prior_model = reassign_fn(
            prior_model, model, x, batch_size, reassign_fraction, precision=precision
        )
        model, prior_stats, space_stats, color_stats, _semantic_stats = fit_gmm_step(
            prior_model,
            model,
            data=x,
            batch_size=batch_size,
            prior_stats=prior_stats,
            space_stats=space_stats,
            color_stats=color_stats,
            precision=precision,
        )

        if step % eval_every == 0:
            output.checkpoint(model, data_params, f"model_{step:02d}.json")

    output.final_model(model, data_params, metrics)
    return metrics


def run_experiment(
    key,
    data_path,
    data_index,
    n_components,
    init_random,
    batch_size,
    use_reassign,
    reassign_fraction,
    shuffle,
    n_frames,
    precision_mode="op",
    precision_tolerance=1e-6,
    precision_search_frame=1,
    output=None,
):
    # Fit continual VBEM
    key, subkey = jr.split(key)
    metrics = fit_continual(
        data_path,
        data_index,
        n_components,
        key=subkey,
        init_random=init_random,
        batch_size=batch_size,
        use_reassign=use_reassign,
        reassign_fraction=reassign_fraction,
        shuffle=shuffle,
        n_frames=n_frames,
        precision_mode=precision_mode,
        precision_tolerance=precision_tolerance,
        precision_search_frame=precision_search_frame,
        output=output,
    )
    rich.print(metrics)

    return metrics


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="habitat",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    jax.config.update("jax_default_device", jax.devices()[int(cfg.device)])
    output = RunOutput.create(output_dir=Path.cwd())

    root_path = Path(vbgs.__file__).parent.parent

    results = run_experiment(
        key=jr.PRNGKey(0),
        n_components=cfg.model.n_components,
        data_path=root_path / Path(cfg.data.data_path),
        data_index=cfg.data.data_index,
        init_random=cfg.model.init_random,
        batch_size=cfg.train.batch_size,
        use_reassign=cfg.model.use_reassign,
        reassign_fraction=float(cfg.model.reassign_fraction),
        shuffle=cfg.data.shuffle,
        n_frames=cfg.data.n_frames,
        precision_mode=getattr(cfg.precision, "mode", "op"),
        precision_tolerance=float(getattr(cfg.precision, "tolerance", 1e-6)),
        precision_search_frame=int(getattr(cfg.precision, "search_frame", 1)),
        output=output,
    )
    results.update({"config": OmegaConf.to_container(cfg)})

    output.metrics(results)


if __name__ == "__main__":
    main()
