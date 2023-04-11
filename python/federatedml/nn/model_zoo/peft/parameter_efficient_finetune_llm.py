import peft
import torch
from transformers import AutoModel
from transformers.configuration_utils import PretrainedConfig
from federatedml.util import LOGGER


AVAILABLE_CONFIG = list(filter(lambda peft_type: peft_type.endswith("Config"), dir(peft)))


class PEFTLM(torch.nn.Module):

    def __init__(self, pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 fp16=True,
                 ) -> None:
        super().__init__()
        self._peft_lm: AutoModel = None
        self.config_path = pretrained_path
        self.peft_type = peft_type
        self.peft_config = peft_config
        self.fp16 = fp16
        assert self.peft_type is not None, "The peft_type must be set.Available peft_type: {}".format(
            AVAILABLE_CONFIG)
        self._init_peft_lm()

    def _init_peft_lm(self):
        if self.config_path is not None:
            LOGGER.info(
                'Prioritize reading from the pretrained model path. Loading pretrained model from {}'.format(
                    self.config_path))
            self._peft_lm = AutoModel.from_pretrained(self.config_path, trust_remote_code=True)
            if self.fp16:
                self._peft_lm.half()
        else:
            raise ValueError(
                'config_path to pretrained model folder cannot be None')

        assert self.peft_type in AVAILABLE_CONFIG, 'peft name {} not in availabe config {}'.format(
            self.peft_type, AVAILABLE_CONFIG)

        if self.peft_config is None:
            config = getattr(peft, self.peft_type)()
        else:
            config = getattr(peft, self.peft_type)(**self.peft_config)

        self._peft_lm = peft.get_peft_model(self._peft_lm, config)

    def model_summary(self):
        summary = self._peft_lm.print_trainable_parameters()
        LOGGER.debug('PEFT_LM model summary: \n{}'.format(summary))

    def forward(self, tokenized_data: dict):
        return self._peft_lm(**tokenized_data)
