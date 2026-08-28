"""Fixed-viewpoint preview renders during / after training."""

from __future__ import annotations

import math
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

from vbgs.io.paths import add_gaussian_splatting_to_syspath
from vbgs.model.feature_layout import model_n_semantic
from vbgs.render.volume import arrays_to_splat, render_img

add_gaussian_splatting_to_syspath()

from arguments import PipelineParams  # noqa: E402
from scene.dataset_readers import CameraInfo  # noqa: E402
from utils.camera_utils import loadCam  # noqa: E402
from utils.graphics_utils import focal2fov  # noqa: E402
from diff_gaussian_rasterization import GaussianRasterizationSettings  # noqa: E402


def _geer_installed() -> bool:
    return "tan_theta" in getattr(GaussianRasterizationSettings, "__annotations__", {})


def camera_info_from_transforms(data_path, split_json: str, index: int) -> CameraInfo:
    import json
    from pathlib import Path

    data_path = Path(data_path)
    with open(data_path / split_json) as f:
        meta = json.load(f)
    frames = meta["frames"]
    if index < 0 or index >= len(frames):
        raise IndexError(f"preview index {index} out of range for {split_json}")
    frame = frames[index]
    angle_x = float(meta["camera_angle_x"])
    image_path = data_path / f"{frame['file_path']}.png"
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    fovy = focal2fov(0.5 * width / math.tan(0.5 * angle_x), height)
    return CameraInfo(
        uid=index,
        R=w2c[:3, :3].T,
        T=w2c[:3, 3],
        FovY=fovy,
        FovX=angle_x,
        image=image,
        image_path=str(image_path),
        image_name=image_path.stem,
        width=width,
        height=height,
    )


def _attach_geer_ph(cam, sample_step=0.002):
    w = int(cam.image_width)
    h = int(cam.image_height)
    cam.focal_x = w / (2.0 * math.tan(cam.FoVx * 0.5))
    cam.focal_y = h / (2.0 * math.tan(cam.FoVy * 0.5))
    cam.principal_x = w / 2.0
    cam.principal_y = h / 2.0
    cam.render_model = 2
    cam.distortion_coeffs = None
    cam.raymap = None

    def fov_sample2ray(fovx, fovy, interval):
        theta_arr = torch.arange(interval / 2, fovx, interval)
        theta_arr, _ = torch.sort(torch.cat((-theta_arr, theta_arr)))
        phi_arr = torch.arange(interval / 2, fovy, interval)
        phi_arr, _ = torch.sort(torch.cat((-phi_arr, phi_arr)))
        return theta_arr.float(), phi_arr.float()

    def mirror_transform(m, z, xi=0.0):
        return m / (1 + xi * (z / (torch.abs(z))) * (1 + m**2) ** 0.5)

    arr_theta, arr_phi = fov_sample2ray(cam.FoVx / 2, cam.FoVy / 2, sample_step)
    cos_theta = torch.cos(arr_theta)
    cos_phi = torch.cos(arr_phi)
    cos_theta = torch.where(
        torch.abs(cos_theta) < 1e-7, torch.full_like(cos_theta, 1e-7), cos_theta
    )
    cos_phi = torch.where(
        torch.abs(cos_phi) < 1e-7, torch.full_like(cos_phi, 1e-7), cos_phi
    )
    device = cam.world_view_transform.device
    cam.tan_theta = torch.tan(arr_theta).to(device)
    cam.tan_phi = torch.tan(arr_phi).to(device)
    cam.mirror_transformed_tan_theta = mirror_transform(
        cam.tan_theta, cos_theta.to(device)
    ).float()
    cam.mirror_transformed_tan_phi = mirror_transform(
        cam.tan_phi, cos_phi.to(device)
    ).float()
    return cam


def _render_geer(viewpoint_camera, pc, pipe, bg_color):
    from diff_gaussian_rasterization import GaussianRasterizer

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        mirror_transformed_tan_theta=viewpoint_camera.mirror_transformed_tan_theta,
        mirror_transformed_tan_phi=viewpoint_camera.mirror_transformed_tan_phi,
        tan_theta=viewpoint_camera.tan_theta,
        tan_phi=viewpoint_camera.tan_phi,
        focal_x=float(viewpoint_camera.focal_x),
        focal_y=float(viewpoint_camera.focal_y),
        principal_x=float(viewpoint_camera.principal_x),
        principal_y=float(viewpoint_camera.principal_y),
        distortion_coeffs=torch.empty(0, device="cuda"),
        raymap=torch.empty(0, device="cuda"),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=bool(getattr(pipe, "debug", False)),
        antialiasing=bool(getattr(pipe, "antialiasing", False)),
        render_mode=int(viewpoint_camera.render_model),
        near_threshold=0.2,
        asso_mode=0,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, *_ = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
    )
    return rendered_image.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()


class FixedViewPreview:
    """Render the evolving model from one fixed Blender camera each train step."""

    def __init__(self, data_path, split_json="transforms_val.json", index=0):
        self.cam_info = camera_info_from_transforms(data_path, split_json, index)
        self._pipe = PipelineParams(ArgumentParser())
        if not hasattr(self._pipe, "antialiasing"):
            self._pipe.antialiasing = False
        self._cargs = type("CArgs", (), {"resolution": -1, "data_device": "cuda:0"})()
        self._use_geer = _geer_installed()
        self._loaded = loadCam(
            self._cargs, id=0, cam_info=self.cam_info, resolution_scale=1.0
        )
        if self._use_geer:
            self._loaded = _attach_geer_ph(self._loaded)
        self._bg = torch.zeros(3, device="cuda")

    def render_rgb(self, model, data_params) -> np.ndarray:
        mu, si = model.denormalize(data_params, clip_val=None)
        alpha = np.asarray(model.prior.alpha).reshape(-1)
        splat = arrays_to_splat(
            np.asarray(mu),
            np.asarray(si),
            alpha,
            n_semantic=model_n_semantic(model),
        )
        if self._use_geer:
            return np.clip(_render_geer(self._loaded, splat, self._pipe, self._bg), 0, 1)
        return np.clip(render_img(splat, [self.cam_info], 0, bg=0), 0, 1)

    def save_png(self, model, data_params, path):
        rgb = self.render_rgb(model, data_params)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
