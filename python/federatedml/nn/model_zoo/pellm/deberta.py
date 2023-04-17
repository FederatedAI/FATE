from transformers import DebertaConfig, AutoConfig
from federatedml.nn.model_zoo.pellm.parameter_efficient_llm import PELLM


class DeBERTa(PELLM):

    config_class = DebertaConfig

    def __init__(self, config: dict = None,
                 pretrained_path: str = None,
                 adapter_type: str = None,
                 adapter_config: dict = None) -> None:

        if pretrained_path is not None:
            self.check_config(pretrain_path=pretrained_path)
        if config is None and pretrained_path is None:
            config = DebertaConfig().to_dict()
        super().__init__(config=config, pretrained_path=pretrained_path, adapter_type=adapter_type, adapter_config=adapter_config)

    def check_config(self, pretrain_path):
        config = AutoConfig.from_pretrained(pretrain_path)
        assert isinstance(
            config, DebertaConfig), 'The config of pretrained model must be DebertaConfig, but got {}'.format(
            type(config))
