from .feature_scale import feature_scale
from .lr import hetero_lr
from .reader import reader
from .intersection import intersection

BUILDIN_COMPONENTS = {
    "hetero_lr": hetero_lr,
    "reader": reader,
    "feature_scale": feature_scale,
    "intersection": intersection
}