from .feature_scale import feature_scale
from .intersection import intersection
from .lr import hetero_lr
from .reader import reader

BUILDIN_COMPONENTS = [
    hetero_lr,
    reader,
    feature_scale,
    intersection,
]
