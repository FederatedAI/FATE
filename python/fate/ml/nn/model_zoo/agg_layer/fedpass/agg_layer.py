from typing import Union, Literal
from fate.ml.nn.model_zoo.agg_layer.agg_layer import AggLayerHost
from fate.ml.nn.model_zoo.agg_layer.fedpass._passport_block import ConvPassportBlock
from dataclasses import dataclass, asdict


@dataclass
class LinearConfig:

    input_features: int
    output_features: int

    def to_dict(self):
        d = asdict(self)
        d['layer_type'] = 'linear'
        return d

    @classmethod
    def from_dict(cls, params):
        params.pop('layer_type', None)
        return cls(**params)

@dataclass
class ConvConfig:

    in_channels: int
    out_channels: int
    kernel_size: Union[int, tuple] = 3
    stride: Union[int, tuple] = 1
    padding: int = 0

    def to_dict(self):
        d = asdict(self)
        d['layer_type'] = 'conv'
        return d

    @classmethod
    def from_dict(cls, params):
        params.pop('layer_type', None)
        return cls(**params)


def _recover_config(config_or_dict):
    if isinstance(config_or_dict, dict):
        layer_type = config_or_dict.get('layer_type')
        if layer_type == 'linear':
            config = LinearConfig.from_dict(config_or_dict)
        elif layer_type == 'conv':
            config = ConvConfig.from_dict(config_or_dict)
        else:
            raise ValueError(f"Unsupported layer type in dictionary: {layer_type}")
    else:
        config = config_or_dict
        assert isinstance(config, (LinearConfig, ConvConfig)), f"Unsupported config type: {type(config)}"

    return config


def _prepare_model(config, passport_distribute: Literal['gaussian', 'uniform'] = 'gaussian',
                 passport_mode: Literal['single', 'multi'] = 'single',
                 loc=-1.0, scale=1.0,
                 low=-1.0, high=1.0,
                 num_passport=1,
                 ae_in=None, ae_out=None,
                 activation: Literal['relu', 'tanh', 'sigmoid'] = "relu"
                ):
    if isinstance(config, LinearConfig):
        raise RuntimeError('currently not support FedPass Linear')
    elif isinstance(config, ConvConfig):
        model = ConvPassportBlock(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            passport_distribute=passport_distribute,
            passport_mode=passport_mode,
            loc=loc, scale=scale,
            low=low, high=high,
            num_passport=num_passport,
            ae_in=ae_in, ae_out=ae_out,
            activation=activation
        )
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")
    return model


class FedPassAggLayerHost(AggLayerHost):

    def __init__(self,
                 config_or_dict: Union[LinearConfig, ConvConfig, dict],
                 activation: Literal['relu', 'tanh', 'sigmoid'] = "relu",
                 passport_distribute: Literal['gaussian', 'uniform'] = 'gaussian',
                 passport_mode: Literal['single', 'multi'] = 'single',
                 loc=-1.0, scale=1.0,
                 low=-1.0, high=1.0,
                 num_passport=1,
                 ae_in=None, ae_out=None):

        super(FedPassAggLayerHost, self).__init__()

        config = _recover_config(config_or_dict)
        self._model = _prepare_model(
            config=config,
            passport_distribute=passport_distribute,
            passport_mode=passport_mode,
            loc=loc, scale=scale,
            low=low, high=high,
            num_passport=num_passport,
            ae_in=ae_in, ae_out=ae_out,
            activation=activation
        )
