# Improved Variational Bayes Gaussian Splatting

ImprovedVBGS is a continual updates rather than gradients the create scenes. It imrpoves upon VBGS by adding sparse top-M responsibility inference, and supports adaptive Gaussian insertion/reassignment for new observations.

## Install

Requirements: Python 3.11, CUDA-capable GPU, JAX CUDA build.

```bash
cd src/vbgs
conda create -n improved-vbgs python=3.11 -y
conda activate improved-vbgs
bash install_deps.sh
pip install -e .[gpu]
```



## Blender Objects

Run one NeRF Synthetic scene:

```bash
cd src/vbgs/scripts
python train_objects.py \
  experiment_name=lego-covis \
  data.model_name=lego \
  model.n_components=100000 \
  model.init_first_frame=false \
  model.init_random=true \
  train.densify=false
```

Run all Blender scenes:

```bash
cd src/vbgs/scripts
for scene in chair drums ficus hotdog lego materials mic ship; do
  python train_objects.py \
    experiment_name="${scene}-covis" \
    data.model_name="$scene" \
    model.n_components=100000 \
    model.init_first_frame=false \
    model.init_random=true \
    train.densify=false \
done
```

Evaluate Blender validation PSNR:

```bash
python eval.py ../output/lego-covis/model_final.json \
  --data-path ../../data/blender/lego \
  --save-images
```

## TUM RGB-D

Preprocess TUM frames:

```bash
python src/preprocess/preprocess.py tum-rgbd \
  --input data/tum/rgbd_dataset_freiburg1_desk \
  --output src/vbgs/output/preprocessed/tum_freiburg1_desk \
  --frames 200
```

Train TUM:

```bash
python src/vbgs/scripts/train_tum.py \
  --data-path src/vbgs/output/preprocessed/tum_freiburg1_desk \
  --run-name tum-covis \
  --components 100000 \
  --frames 200 \
  --batch-size 50000 \
  --top-m 32 \
  --candidate-m 128 \
  --init-first-frame \
  --densify \
  --densify-point-ratio 1.0 \
  --densify-unseen-distance-threshold 0.04 \
  --densify-min-unseen-fraction 0.01 \
  --densify-reassign-if-full \
  --reassign-fraction 0.005 \
  --no-reassign \
  --eval
```

## Semantic Variant

Semantic data appends continuous SAM/CLIP feature vectors after `[xyz, rgb]`.
These features are modeled as another VBGS feature block and updated by CAVI with
the same responsibilities as color.

```bash
cd src/vbgs
pip install -e ".[semantic]"
cd ../..

python src/preprocess/preprocess.py tum-rgbd \
  --input data/tum/rgbd_dataset_freiburg1_desk \
  --output src/vbgs/output/preprocessed/tum_freiburg1_desk_sem16 \
  --frames 200 \
  --semantic \
  --semantic-classes 16

python src/vbgs/scripts/train_tum.py \
  --data-path src/vbgs/output/preprocessed/tum_freiburg1_desk_sem16 \
  --run-name tum-semantic \
  --components 100000 \
  --frames 200 \
  --semantic-classes 16 \
  --batch-size 50000 \
  --top-m 32 \
  --candidate-m 128 \
  --init-first-frame \
  --no-densify \
  --no-reassign \
  --eval
```

## References

- VBGS: [arXiv:2410.03592](https://arxiv.org/abs/2410.03592)
- VBGS Follow Up: [arXiv:2603.08499](https://arxiv.org/abs/2603.08499)

## License

`src/vbgs` remains under the [VERSES Academic Research License](src/vbgs/LICENSE.txt).
