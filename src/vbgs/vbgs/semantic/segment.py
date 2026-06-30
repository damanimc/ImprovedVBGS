"""SAM + CLIP semantic segmentation for RGB-D point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_INDOOR_PROMPTS = (
    "background",
    "desk surface",
    "table",
    "chair",
    "computer monitor",
    "keyboard",
    "computer mouse",
    "book",
    "paper",
    "coffee mug",
    "bottle",
    "person",
    "wall",
    "floor",
    "cabinet",
    "other object",
)

_SAM = None
_CLIP = None
_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)


def _checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[2] / "checkpoints" / _CHECKPOINT_NAME


def _ensure_checkpoint(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    print(f"downloading SAM checkpoint -> {path}")
    urllib.request.urlretrieve(_CHECKPOINT_URL, path)
    return path


def _load_sam(device: str = "cuda"):
    global _SAM
    if _SAM is not None:
        return _SAM
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    ckpt = _ensure_checkpoint(_checkpoint_path())
    dev = device if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt))
    sam.to(device=dev)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=400,
    )
    _SAM = (generator, dev)
    return _SAM


def _load_clip(device: str = "cuda"):
    global _CLIP
    if _CLIP is not None:
        return _CLIP
    import torch
    from transformers import CLIPModel, CLIPProcessor

    dev = device if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True
    )
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval().to(dev)
    _CLIP = (model, processor, dev)
    return _CLIP


def class_names_for_model(prompts: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if prompts is None:
        return DEFAULT_INDOOR_PROMPTS
    return tuple(prompts)


def _mask_crops(rgb_u8: np.ndarray, masks: list[dict]):
    h, w = rgb_u8.shape[:2]
    crops = []
    for mask in masks:
        seg = mask["segmentation"]
        ys, xs = np.nonzero(seg)
        if ys.size == 0:
            crops.append(None)
            continue
        y0, y1 = max(int(ys.min()), 0), min(int(ys.max()) + 1, h)
        x0, x1 = max(int(xs.min()), 0), min(int(xs.max()) + 1, w)
        crop = rgb_u8[y0:y1, x0:x1]
        crops.append(crop if crop.size else None)
    return crops


def _clip_text_features(model, text_inputs):
    ids = {k: text_inputs[k] for k in ("input_ids", "attention_mask") if k in text_inputs}
    pooled = model.text_model(**ids).pooler_output
    return model.text_projection(pooled)


def _clip_image_features(model, image_inputs):
    pooled = model.vision_model(pixel_values=image_inputs["pixel_values"]).pooler_output
    return model.visual_projection(pooled)


def _clip_assign_masks(
    rgb: np.ndarray,
    masks: list[dict],
    prompts: tuple[str, ...],
    device: str,
) -> np.ndarray:
    import torch
    from PIL import Image

    model, processor, dev = _load_clip(device)
    h, w = rgb.shape[:2]
    label_map = np.zeros((h, w), dtype=np.int64)
    if not masks:
        return label_map

    text_inputs = processor(text=list(prompts), return_tensors="pt", padding=True)
    text_inputs = {k: v.to(dev) for k, v in text_inputs.items()}
    with torch.inference_mode():
        text_feat = _clip_text_features(model, text_inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    ordered = sorted(masks, key=lambda m: int(m["area"]))
    crops = _mask_crops(rgb, ordered)
    valid_idx = [i for i, c in enumerate(crops) if c is not None]
    if not valid_idx:
        return label_map

    images = [Image.fromarray(crops[i]) for i in valid_idx]
    image_inputs = processor(images=images, return_tensors="pt", padding=True)
    image_inputs = {k: v.to(dev) for k, v in image_inputs.items()}
    with torch.inference_mode():
        img_feat = _clip_image_features(model, image_inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = img_feat @ text_feat.T
        class_ids = torch.argmax(logits, dim=-1).cpu().numpy()

    for local_i, mask_i in enumerate(valid_idx):
        label_map[ordered[mask_i]["segmentation"]] = int(class_ids[local_i])
    return label_map


def segment_rgb(
    rgb: np.ndarray,
    *,
    prompts: tuple[str, ...] | None = None,
    max_masks: int = 48,
    device: str = "cuda",
) -> np.ndarray:
    """Segment image with SAM; label each mask via batched CLIP text prompts."""
    prompt_list = tuple(prompts) if prompts is not None else DEFAULT_INDOOR_PROMPTS
    rgb_u8 = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)

    generator, _ = _load_sam(device)
    masks = generator.generate(rgb_u8)
    if len(masks) > max_masks:
        masks = sorted(masks, key=lambda m: int(m["area"]), reverse=True)[:max_masks]
    return _clip_assign_masks(rgb_u8, masks, prompt_list, device)


def labels_to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    labels = np.clip(labels, 0, num_classes - 1)
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def attach_semantic_features(
    rgb_image: np.ndarray,
    depth: np.ndarray,
    *,
    num_classes: int | None = None,
    prompts: tuple[str, ...] | None = None,
    max_masks: int = 48,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Per valid-depth pixel: class id and one-hot semantic feature vector."""
    prompt_list = tuple(prompts) if prompts is not None else DEFAULT_INDOOR_PROMPTS
    n_cls = num_classes or len(prompt_list)
    if n_cls > len(prompt_list):
        raise ValueError(
            f"num_classes={n_cls} exceeds number of prompts ({len(prompt_list)})"
        )
    prompt_list = prompt_list[:n_cls]
    seg = segment_rgb(
        rgb_image, prompts=prompt_list, max_masks=max_masks, device=device
    )
    v_idx, u_idx = np.nonzero(depth > 0)
    labels = seg[v_idx, u_idx]
    onehot = labels_to_onehot(labels, num_classes=n_cls)
    return labels.astype(np.int64), onehot
