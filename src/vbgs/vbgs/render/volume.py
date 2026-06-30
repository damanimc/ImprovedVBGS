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
import json
import math

from pathlib import Path

import jax
import jax.numpy as jnp

import vbgs
from vbgs.io.paths import add_gaussian_splatting_to_syspath

root_path = Path(vbgs.__file__).parent.parent

# Gaussian splatting imports
import torch

add_gaussian_splatting_to_syspath()
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render, network_gui
from scene.dataset_readers import readCamerasFromTransforms, CameraInfo
from utils.camera_utils import loadCam
from arguments import PipelineParams
from gaussian_renderer import render as render_cuda
from argparse import ArgumentParser
from utils.sh_utils import RGB2SH, SH2RGB
from utils.general_utils import inverse_sigmoid


parser = ArgumentParser(description="Training script parameters")
pipe = PipelineParams(parser)


class CustomArgs:
    resolution = -1
    data_device = "cuda:0"


cargs = CustomArgs()


def render_img(model, cams, idx, bg=0, scale=1.0):
    custom_cam = loadCam(cargs, id=0, cam_info=cams[idx], resolution_scale=1.0)
    net_image = render_cuda(
        custom_cam, model, pipe, bg * torch.ones(3).to("cuda:0"), scale
    )["render"]
    img_ours_cu = net_image.detach().cpu().permute(1, 2, 0).numpy()
    return img_ours_cu.clip(0, 1)


def model_xyz_bounds(model_path):
    with open(model_path, "r") as f:
        d = json.load(f)
    xyz = np.asarray(d["mu"], dtype=np.float32)[:, :3]
    return xyz.min(axis=0), xyz.max(axis=0)


def make_topdown_camera(
    lo,
    hi,
    variant="below_z",
    width=1200,
    height=1200,
    fov_deg=55.0,
):
    from PIL import Image

    center = (lo + hi) / 2.0
    span = np.maximum(hi - lo, 1e-6)
    xy_span = float(max(span[0], span[1]))
    z_span = float(span[2])
    fov = math.radians(fov_deg)
    dist = (0.5 * xy_span) / math.tan(0.5 * fov) + z_span + 0.5
    blank = Image.new("RGB", (width, height), (0, 0, 0))

    def make_cam(name, right, down, forward, pos):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = np.asarray(right, dtype=np.float32)
        c2w[:3, 1] = np.asarray(down, dtype=np.float32)
        c2w[:3, 2] = np.asarray(forward, dtype=np.float32)
        c2w[:3, 3] = np.asarray(pos, dtype=np.float32)
        w2c = np.linalg.inv(c2w)
        return CameraInfo(
            uid=0,
            R=w2c[:3, :3].T,
            T=w2c[:3, 3],
            FovY=fov,
            FovX=fov,
            image=blank,
            image_path="",
            image_name=name,
            width=width,
            height=height,
        )

    variants = {
        "below_z": make_cam(
            "topdown_below_z",
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [center[0], center[1], lo[2] - dist],
        ),
        "above_negz": make_cam(
            "topdown_above_negz",
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
            [center[0], center[1], hi[2] + dist],
        ),
    }
    return variants[variant]


def render_topdown(model, model_path, variant="below_z", bg=0, scale=1.0, **camera_kwargs):
    lo, hi = model_xyz_bounds(model_path)
    cam = make_topdown_camera(lo, hi, variant=variant, **camera_kwargs)
    return render_img(model, [cam], 0, bg=bg, scale=scale)


def rot_mat_to_quat(matrix):
    trace = np.trace(matrix)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * s,
            (matrix[2, 1] - matrix[1, 2]) / s,
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[1, 0] - matrix[0, 1]) / s,
        ]
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-8)) * 2.0
            quat = [
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            ]
        elif axis == 1:
            s = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-8)) * 2.0
            quat = [
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            ]
        else:
            s = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-8)) * 2.0
            quat = [
                (matrix[1, 0] - matrix[0, 1]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                0.25 * s,
            ]
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat)
    return quat / max(norm, 1e-8)


def covariance_to_scaling_rotation(covariance):
    scales = []
    rotations = []
    eye = np.eye(3, dtype=np.float32)
    for cov in covariance:
        cov = 0.5 * (cov + cov.T)
        jitter = 1e-10
        for _ in range(5):
            try:
                mat_l = np.linalg.cholesky(cov + eye * jitter)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            values, vectors = np.linalg.eigh(cov)
            values = np.clip(values, 1e-10, None)
            mat_l = vectors @ np.diag(np.sqrt(values))

        scale = np.linalg.norm(mat_l, axis=-1)
        scale = np.clip(scale, 1e-10, None)
        rotation = mat_l / scale[:, None]
        scales.append(scale)
        rotations.append(rot_mat_to_quat(rotation))
    return np.asarray(scales, dtype=np.float32), np.asarray(rotations, dtype=np.float32)


def alpha_to_opacity(alpha, threshold=1e-6, max_opacity=0.95):
    alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
    opacity = np.where(alpha > threshold, max_opacity, 1e-4)
    return opacity.astype(np.float32)


def construct_covariance(lower, device="cuda:0"):
    cov = torch.zeros((lower.shape[0], 3, 3), device=device)

    # fill in lower triangle
    cov[:, 0:3, 0] = lower[:, :3]
    cov[:, 1:3, 1] = lower[:, 3:5]
    cov[:, 2:3, 2] = lower[:, 5:]

    # make symmetrical
    cov[:, 0, 1:3] = cov[:, 1:3, 0]
    cov[:, 1, 2] = cov[:, 2, 1]

    return cov


def vbgs_model_to_splat(
    model_path,
    device="cuda:0",
    dtype=torch.float32,
    opacity_threshold=1e-6,
    max_opacity=0.95,
    max_scale_percentile=None,
    scale_multiplier=1.41,
    color_mode="rgb",
):
    with open(model_path, "r") as f:
        d = json.load(f)

    mu, si = np.array(d["mu"]), np.array(d["si"])
    alpha = np.array(d["alpha"])
    n_semantic = int(d.get("n_semantic", 0))

    scaling, rotation = covariance_to_scaling_rotation(si[:, :3, :3])
    scaling = scaling * float(scale_multiplier)
    if max_scale_percentile is not None:
        scale_cap = max(float(np.percentile(scaling, max_scale_percentile)), 1e-6)
        scaling = np.clip(scaling, 1e-5, scale_cap)
    scaling = scaling.astype(np.float32)
    opacity = alpha_to_opacity(
        alpha, threshold=opacity_threshold, max_opacity=max_opacity
    )
    mask = np.isfinite(mu[:, :3]).all(axis=1) & np.isfinite(scaling).all(axis=1)

    if color_mode == "semantic" and n_semantic > 0:
        from vbgs.semantic.palette import class_color_rgb, label_from_onehot

        sem = mu[:, 6 : 6 + n_semantic]
        labels = label_from_onehot(sem)
        display_rgb = class_color_rgb(labels, n_semantic)
    else:
        display_rgb = mu[:, 3:6].clip(0, 1)

    model = GaussianModel(3)
    model.max_sh_degree = 0
    model._xyz = torch.tensor(mu[mask, :3], dtype=dtype, device=device)
    model._features_dc = torch.tensor(
        RGB2SH(display_rgb[mask]), dtype=dtype, device=device
    ).unsqueeze(1)
    model._features_rest = torch.zeros(
        (int(mask.sum()), 0, 3), dtype=dtype, device=device
    )
    model._opacity = inverse_sigmoid(
        torch.tensor(opacity[mask, None], dtype=dtype, device=device)
    )
    model._scaling = torch.tensor(scaling[mask], dtype=dtype, device=device)
    model.scaling_activation = lambda x: x
    model._rotation = torch.tensor(rotation[mask], dtype=dtype, device=device)
    return model


def save_inria_splat_ply(model_path, ply_path, **kwargs):
    """Write INRIA 3DGS PLY for viewers like SuperSplat (log-scales, logit opacity)."""
    model = vbgs_model_to_splat(model_path, **kwargs)
    model._scaling = torch.log(torch.clamp(model._scaling, min=1e-6))
    model.scaling_activation = torch.exp
    model.save_ply(str(ply_path))
