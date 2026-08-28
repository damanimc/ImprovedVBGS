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

"""RGB-D scene loader used by scripts/train.py.

Accepts either a preprocessed standard scene (manifest.json + metric
``*_depth_da3.npy``) or a stock NeRF Synthetic Blender folder.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from PIL import Image

from vbgs.camera import transform_uvd_to_points
from vbgs.data.utils import normalize_data


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _intrinsics_from_dict(raw: dict, width: int, height: int) -> jnp.ndarray:
    fx = float(raw["fx"])
    fy = float(raw.get("fy", fx))
    cx = float(raw.get("cx", width / 2.0))
    cy = float(raw.get("cy", height / 2.0))
    mat = jnp.eye(4)
    mat = mat.at[0, 0].set(fx)
    mat = mat.at[1, 1].set(fy)
    mat = mat.at[0, 2].set(cx)
    mat = mat.at[1, 2].set(cy)
    return mat


def _intrinsics_from_angle_x(angle_x: float, width: int, height: int) -> jnp.ndarray:
    # Match VersesTech blender.py: square-image fx from height.
    fx = height / (2 * jnp.tan(angle_x / 2))
    mat = jnp.eye(4)
    mat = mat.at[0, 0].set(fx)
    mat = mat.at[1, 1].set(fx)
    mat = mat.at[0, 2].set(width / 2)
    mat = mat.at[1, 2].set(height / 2)
    return mat


class SceneDataIterator:
    """Iterate RGB-D frames as (N, 6[+C]) xyz-rgb(+semantic) point clouds."""

    def __init__(
        self,
        data_path,
        split="train",
        data_params=None,
        subsample=None,
    ):
        self._data_params = data_params
        self._subsample = subsample
        self._data_path = Path(data_path)
        self._split = split
        self._index = 0
        self.key = jr.PRNGKey(0)

        transform_path = self._data_path / f"transforms_{split}.json"
        if not transform_path.exists():
            raise FileNotFoundError(transform_path)
        data = _load_json(transform_path)
        self._frames = list(data.get("frames", []))
        if not self._frames:
            self._intrinsics = jnp.eye(4)
            self._r = None
            return

        first = self._frames[0]
        color_path = self._data_path / f"{first['file_path']}.png"
        image = Image.open(color_path)
        width, height = image.size

        manifest = {}
        manifest_path = self._data_path / "manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)

        intrinsics_raw = data.get("intrinsics") or manifest.get("intrinsics")
        angle_x = float(data.get("camera_angle_x", 0.6911112070083618))
        if isinstance(intrinsics_raw, dict) and "fx" in intrinsics_raw:
            self._intrinsics = _intrinsics_from_dict(intrinsics_raw, width, height)
        else:
            self._intrinsics = _intrinsics_from_angle_x(angle_x, width, height)

        shape = (height, width, 3)
        self._r = self._compute_distance_to_depth(angle_x, shape)

    @staticmethod
    def _compute_distance_to_depth(angle_x, shape):
        uv = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]))
        uv = jnp.concatenate(
            [jnp.expand_dims(u, -1) for u in uv]
            + [jnp.ones(shape=(*shape[:2], 1))],
            axis=-1,
        )
        uv = uv - shape[0] / 2
        uv = uv.at[..., 0].set(uv[..., 0] * angle_x / 2)
        uv = uv.at[..., 1].set(uv[..., 1] * angle_x / 2)
        uvr = uv.reshape(-1, 3)
        uvr = uvr / jnp.linalg.norm(uvr, axis=-1, keepdims=True)
        fwd = jnp.array([0, 0, -1])
        fwd = fwd / jnp.linalg.norm(fwd)
        r = jax.vmap(partial(jnp.dot, fwd))(uvr)
        return r.reshape(uv.shape[:2])

    def __len__(self):
        return len(self._frames)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self._frames):
            item = self._get_frame(self._index)
            self._index += 1
            return item
        raise StopIteration

    def preload_frames(self):
        for i in range(len(self._frames)):
            self._get_frame(i)

    def _depth_for_frame(self, frame: dict) -> np.ndarray:
        rel = frame["file_path"]
        npy = self._data_path / f"{rel}_depth_da3.npy"
        if npy.exists():
            depth = np.load(npy).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            return depth

        pngs = list(self._data_path.glob(f"{rel}_depth_*.png"))
        if not pngs:
            raise FileNotFoundError(
                f"No depth for {rel}: expected {npy} or {rel}_depth_*.png"
            )
        depth_im = np.asarray(Image.open(pngs[0]))
        depth = 8 * (1.0 - (depth_im[..., 0] / 255.0))
        depth = depth * np.asarray(self._r)
        depth = depth * (depth_im[..., 0] > 0)
        return depth.astype(np.float32)

    def _materialize_frame(self, i: int) -> np.ndarray:
        frame = self._frames[i]
        color = np.asarray(Image.open(self._data_path / f"{frame['file_path']}.png"))
        depth = self._depth_for_frame(frame)
        camera_to_world = np.array(frame["transform_matrix"])
        points, rgb = transform_uvd_to_points(
            color[..., :3],
            jnp.asarray(depth),
            camera_to_world,
            self._intrinsics,
            from_opengl=True,
            filter_zero=True,
        )
        return np.asarray(jnp.concatenate([points, rgb], axis=1))

    def _compute_cloud(self, i):
        return self._materialize_frame(i)

    def _get_frame(self, i):
        frame = self._frames[i]
        cloud_path = self._data_path / f"{frame['file_path']}.npz"
        if cloud_path.exists():
            data = np.load(cloud_path)["arr_0"]
        else:
            data = self._materialize_frame(i)
            cloud_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cloud_path, data)

        if self._data_params is not None:
            data, _ = normalize_data(data, self._data_params)

        if self._subsample is not None:
            self.key, subkey = jr.split(self.key)
            data = jr.permutation(subkey, data, independent=False)
            data = data[: self._subsample]

        return np.array(data)
