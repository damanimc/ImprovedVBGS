"""Ray-space Variational Bayes Gaussian Splatting (RVBGS).

Reference implementation of the sharpened derivation in
``docs/theory/ray_space_vbgs.md``. Includes geometry-only and colored
coordinate ascent, Blender/Lego loaders, and a CPU EWA renderer for PSNR.
"""

from vbgs.ray_space.blender_data import (
    blender_intrinsics,
    decode_blender_depth,
    load_frame_rays,
    load_ray_dataset,
)
from vbgs.ray_space.colored import (
    ColoredRayState,
    fit_colored_ray_vbgs,
    state_to_model_dict,
)
from vbgs.ray_space.depth_posterior import (
    expected_quadratic_energy,
    gaussian_depth_posterior,
    ray_quadratic_coeffs,
    truncated_normal_moments,
)
from vbgs.ray_space.geometry import (
    point_moments,
    point_on_ray,
    rays_from_c2w,
    rays_from_w2c,
)
from vbgs.ray_space.reduction import (
    ReductionCheck,
    check_proposition_1,
    ray_uncertainty_inflates_scatter,
)
from vbgs.ray_space.render_cpu import eval_split_psnr, psnr, render_view
from vbgs.ray_space.updates import (
    NIWParams,
    RayVBGSState,
    coordinate_ascent_step,
    fit_ray_vbgs,
    update_depths,
    update_responsibilities,
)

__all__ = [
    "ColoredRayState",
    "NIWParams",
    "RayVBGSState",
    "ReductionCheck",
    "blender_intrinsics",
    "check_proposition_1",
    "coordinate_ascent_step",
    "decode_blender_depth",
    "eval_split_psnr",
    "expected_quadratic_energy",
    "fit_colored_ray_vbgs",
    "fit_ray_vbgs",
    "gaussian_depth_posterior",
    "load_frame_rays",
    "load_ray_dataset",
    "point_moments",
    "point_on_ray",
    "psnr",
    "ray_quadratic_coeffs",
    "ray_uncertainty_inflates_scatter",
    "rays_from_c2w",
    "rays_from_w2c",
    "render_view",
    "state_to_model_dict",
    "truncated_normal_moments",
    "update_depths",
    "update_responsibilities",
]
