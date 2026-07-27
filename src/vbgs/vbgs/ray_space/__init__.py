"""Ray-space Variational Bayes Gaussian Splatting (RVBGS).

Reference implementation of the sharpened derivation in
``docs/theory/ray_space_vbgs.md``. Geometry-only coordinate ascent with
closed-form Gaussian (optionally truncated) depth posteriors.
"""

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
from vbgs.ray_space.updates import (
    NIWParams,
    RayVBGSState,
    coordinate_ascent_step,
    fit_ray_vbgs,
    update_depths,
    update_responsibilities,
)

__all__ = [
    "NIWParams",
    "RayVBGSState",
    "ReductionCheck",
    "check_proposition_1",
    "coordinate_ascent_step",
    "expected_quadratic_energy",
    "fit_ray_vbgs",
    "gaussian_depth_posterior",
    "point_moments",
    "point_on_ray",
    "ray_quadratic_coeffs",
    "ray_uncertainty_inflates_scatter",
    "rays_from_c2w",
    "rays_from_w2c",
    "truncated_normal_moments",
    "update_depths",
    "update_responsibilities",
]
