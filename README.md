# Improved Variational Bayes Gaussian Splatting

**Paper:** [ImprovedVBGS: Real-time Continual Variational Bayes Gaussian Splatting](https://arxiv.org/abs/2607.15542) ([arXiv:2607.15542](https://arxiv.org/abs/2607.15542))

ImprovedVBGS accelerates continual VBGS for on-the-fly reconstruction via
**(i) spatially truncated variational inference** (KD-tree nearest-neighbour
candidate pruning) and **(ii) improved reassignment**. On an RTX
3070 Ti, NeRF Synthetic mean per-frame latency drops from 84.0 s/frame to 0.050 s/frame
(1680×) without reassignment.

![Lego reconstruction](lego.gif)

NeRF Synthetic (paper Table 1): 100k components, 200 frames, mean 0.133 s/frame,
21.42 dB validation PSNR.

## Install

Requirements: Python 3.11, CUDA GPU, JAX CUDA build.

```bash
cd src/vbgs
conda create -n improved-vbgs python=3.11 -y
conda activate improved-vbgs
bash install_deps.sh
pip install -e .[gpu]
```

External repositories belong under `third_party/`, not `src/`.

## Outputs

Generated runs go under `src/vbgs/output/`. Do not write experiment outputs to
`src/vbgs/scripts/data/`; output path validation rejects that layout. Run names
are shortened automatically.

## Blender Objects

Download NeRF Synthetic Blender data:

```bash
mkdir -p data/blender
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_synthetic.zip \
  -O data/blender/nerf_synthetic.zip
unzip data/blender/nerf_synthetic.zip -d data/blender
```

Convert Lego to the standard scene format:

```bash
python src/preprocess/preprocess.py blender \
  --input data/blender/lego \
  --output data/scenes/lego
```

Convert all Blender scenes:

```bash
for scene in chair drums ficus hotdog lego materials mic ship; do
  python src/preprocess/preprocess.py blender \
    --input "data/blender/${scene}" \
    --output "data/scenes/${scene}"
done
```

Train Lego with the truncated E-step (paper: C=4 KD-tree candidates) and
static-shape reassignment before fit:

```bash
cd src/vbgs/scripts
python train.py \
  --data-path ../../../data/blender/lego \
  --run-name lego \
  --components 100000 \
  --frames 200 \
  --batch-size 250000 \
  --top-m 4 \
  --candidate-m 4 \
  --init random \
  --no-densify \
  --reassign \
  --reassign-before-fit \
  --reassign-every 1 \
  --reassign-fraction 0.05 \
  --precision fp64 \
  --no-eval
```

`--candidate-m` is the KD-tree truncation width C. `--top-m` is how many of
those candidates the E-step scores (set equal to C to match the paper).

Train all preprocessed Blender scenes:

```bash
for scene in chair drums ficus hotdog lego materials mic ship; do
  python train.py \
    --data-path "../../../data/scenes/${scene}" \
    --run-name "${scene}" \
    --components 100000 \
    --frames 200 \
    --batch-size 250000 \
    --top-m 4 \
    --candidate-m 4 \
    --init random \
    --no-densify \
    --reassign \
    --reassign-before-fit \
    --reassign-every 1 \
    --reassign-fraction 0.05 \
    --precision fp64 \
    --eval
done
```

Evaluate and save renders:

```bash
python eval.py \
  ../output/lego/model_final.json \
  --data-path ../../data/blender/lego \
  --save-images
```


## Standard Scene Format

The unified trainer accepts preprocessed RGB-D scenes:

```text
data/scenes/<scene>/
  manifest.json
  transforms_train.json
  transforms_val.json
  train/
    frame_000000.png
    frame_000000_depth_da3.npy
  val/
    frame_000008.png
    frame_000008_depth_da3.npy
```

Convert Blender or TUM into that format with `src/preprocess/preprocess.py`.
For MP4 inputs, the preprocessor uses Depth Anything 3 to estimate both
per-frame depth and camera poses:

```bash
cd src/vbgs
pip install -e ".[depth]"
cd ../..

python src/preprocess/preprocess.py video \
  --input data/videos/scene.mp4 \
  --output data/scenes/scene_from_video \
  --frames 200 \
  --stride 1 \
  --depth-source da3 \
  --pose-source da3 \
  --da3-model depth-anything/DA3-BASE
```

Depth Anything 3 estimates relative depth and camera extrinsics/intrinsics from
the extracted video frames. The preprocessor rescales each depth map to
`--depth-median` meters, which defaults to `1.0`, and converts DA3 camera poses
into the Blender-style camera-to-world matrices used by the trainer. Add
`--da3-use-ray-pose` for DA3's slower ray-head pose path.

If you already have estimates, use:

```bash
python src/preprocess/preprocess.py video \
  --input data/videos/scene.mp4 \
  --output data/scenes/scene_from_video \
  --depth-source dir \
  --depth-dir data/videos/scene_depth \
  --pose-source file \
  --pose-file data/videos/scene_transforms.json
```

`--depth-source placeholder` or `--pose-source stationary` are smoke-test modes
only and should not be used for reconstruction results.

## Citation

If you use this code, please cite:

```bibtex
@misc{mgunicoker2026improvedvbgsrealtimecontinualvariational,
      title={ImprovedVBGS: Real-time Continual Variational Bayes Gaussian Splatting},
      author={Damani Mguni-Coker},
      year={2026},
      eprint={2607.15542},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.15542},
}
```

## References

- ImprovedVBGS: [arXiv:2607.15542](https://arxiv.org/abs/2607.15542)
- VBGS: [arXiv:2410.03592](https://arxiv.org/abs/2410.03592)
- VBGS optimization study: [arXiv:2603.08499](https://arxiv.org/abs/2603.08499)

## License

`src/vbgs` remains under the [VERSES Academic Research License](src/vbgs/LICENSE.txt).
