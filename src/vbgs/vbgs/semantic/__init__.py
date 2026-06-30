from vbgs.semantic.palette import class_color_rgb, label_from_onehot, semantic_palette
from vbgs.semantic.segment import (
    DEFAULT_INDOOR_PROMPTS,
    attach_semantic_features,
    class_names_for_model,
    segment_rgb,
)

__all__ = [
    "DEFAULT_INDOOR_PROMPTS",
    "attach_semantic_features",
    "class_color_rgb",
    "class_names_for_model",
    "label_from_onehot",
    "segment_rgb",
    "semantic_palette",
]
