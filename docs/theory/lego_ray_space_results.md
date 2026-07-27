# Lego experiment (Ray-space VBGS)

CPU reference run on NeRF Synthetic **Lego** (depth from public
`nerfbaselines` mirror test split; eval on `transforms_val.json`).

## Command

```bash
cd src/vbgs
PYTHONPATH=. python scripts/train_ray_space_lego.py \
  --data-path /path/to/lego \
  --output-dir output/lego_ray_space \
  --train-split test \
  --eval-split val \
  --frames 50 \
  --eval-frames 20 \
  --components 64 \
  --iters 5 \
  --image-scale 0.25 \
  --max-pixels-per-frame 10000 \
  --splat-points 12000 \
  --fix-depth \
  --depth-prior-precision 250
```

Data: HuggingFace `nerfbaselines/nerfbaselines-data` → `blender/lego.zip`
(includes `*_depth_*.png` on the test split).

## Results

| Setup | Val mean PSNR | Notes |
| --- | ---: | --- |
| **RVBGS fixed depth (Dirac / Prop. 1)** | **15.20 dB** | 20 val views, scale 0.25, CPU EWA |
| GT-depth point splat baseline | 15.52 dB | same renderer / budget |
| Noisy depth, no refine (σ=0.4) | 12.41 dB | ablation |
| Noisy depth + free `q(λ)` | 11.46 dB | under-tuned free-depth; see note |

Fixed-depth metrics file: `src/vbgs/output/lego_ray_space/val_psnr.json`

```json
{
  "mean_psnr": 15.1976,
  "std_psnr": 0.7571,
  "n": 20,
  "split": "val",
  "image_scale": 0.25,
  "n_train_frames": 50,
  "n_rays": 498045,
  "n_points": 12000,
  "renderer": "cpu_ewa_alpha"
}
```

## Notes

- Full ImprovedVBGS CUDA 3DGS eval is not used here (no GPU in this
  environment). PSNR is from the CPU EWA alpha renderer in
  `vbgs.ray_space.render_cpu`.
- Paper-scale ImprovedVBGS reports ~21 dB on Lego with 100k components and
  CUDA rasterization; this run is a scaled CPU proof that RVBGS trains and
  evaluates on Lego.
- Free-depth `q(λ)` is implemented and runs; quality on Lego still needs
  stronger priors / less mixture collapse before it beats a noisy-depth
  point cloud.
