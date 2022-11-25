from .feature_scale import feature_scale
from .hetero_lr import hetero_lr
from .intersection import intersection
from .reader import reader

BUILDIN_COMPONENTS = [
    hetero_lr,
    reader,
    feature_scale,
    intersection,
]
