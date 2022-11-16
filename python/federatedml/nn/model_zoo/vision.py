import torch as t
from torchvision.models import get_model


class TorchVisionModels(t.nn.Module):
    """
    This Class provides ALL torchvision classification models,
    instantiate models and using pretrained weights by providing string model name and weight names

    Parameters
    ----------
    vision_model_name: str, name of models provided by torchvision.models, for all available vision model, see:
                       https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
    pretrain_weights: str, name of pretrained weight, for available vision weights, see:
                      https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
    """

    def __init__(self, vision_model_name: str, pretrain_weights: str = None):
        super(TorchVisionModels, self).__init__()
        self.model = get_model(vision_model_name, weights=pretrain_weights)

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return self.model.__repr__()
