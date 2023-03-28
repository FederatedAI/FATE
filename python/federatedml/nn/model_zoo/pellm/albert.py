from transformers import AlbertConfig, AutoConfig
from federatedml.nn.model_zoo.pellm.parameter_efficient_llm import PELLM


class ALBert(PELLM):

    config_class = AlbertConfig

    def __init__(self, config: dict=None, 
                 pretrained_path: str=None,
                 adapter_type: str=None,
                 adapter_config: dict=None,
                 ) -> None:
        
        if pretrained_path is not None:
            self.check_config(pretain_path=pretrained_path)
        if config is None and pretrained_path is None:
            config = AlbertConfig().to_dict()  # use default model setting
        super().__init__(config=config, pretrained_path=pretrained_path, adapter_type=adapter_type, adapter_config=adapter_config)

    def check_config(self, pretain_path):
        config = AutoConfig.from_pretrained(pretain_path)
        assert type(config) == AlbertConfig, 'The config of pretrained model must be AlbertConfig, but got {}'.format(type(config))
