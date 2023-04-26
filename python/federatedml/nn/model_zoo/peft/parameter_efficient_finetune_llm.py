import peft
import torch
from transformers import AutoConfig, AutoModel
from transformers.configuration_utils import PretrainedConfig
from federatedml.util import LOGGER


AVAILABLE_CONFIG = list(filter(lambda peft_type: peft_type.endswith("Config"), dir(peft)))


class PEFTLM(torch.nn.Module):

    def __init__(self,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 fp16=True,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.peft_lm: peft.PeftModel = None
        self.config_path = pretrained_path
        self.peft_type = peft_type
        self.peft_config = peft_config
        self.fp16 = fp16
        self.config = None

        self._init_peft_lm()

    def _init_peft_lm(self):
        self.init_config()
        self.init_lm()
        self.add_peft()

    def init_config(self):
        self.config = AutoConfig.from_pretrained(self.config_path, trust_remote_code=True)

    def init_lm(self):
        if self.config is not None:
            self.peft_lm = AutoModel.from_pretrained(self.config_path, config=self.config, trust_remote_code=True)
        elif self.config_path is not None:
            self.peft_lm = AutoModel.from_pretrained(self.config_path, trust_remote_code=True)
        else:
            raise ValueError(
                'config_path to pretrained model folder cannot be None')
        if self.fp16:
            self.peft_lm.half()

    def add_peft(self):
        assert self.peft_type in AVAILABLE_CONFIG, 'peft name {} not in availabe config {}'.format(
            self.peft_type, AVAILABLE_CONFIG)

        if self.peft_config is None:
            config = getattr(peft, self.peft_type)()
        else:
            config = getattr(peft, self.peft_type)(**self.peft_config)

        self.peft_lm = peft.get_peft_model(self.peft_lm, config)

    def model_summary(self):
        summary = self.peft_lm.print_trainable_parameters()
        LOGGER.debug('PEFT_LM model summary: \n{}'.format(summary))

    def forward(self, tokenized_data: dict):
        return self.peft_lm(**tokenized_data)

    def save_pretrained(self, path):
        from pathlib import Path

        state_dict = {
            k: p.to("cpu") for k, p in self.peft_lm.named_parameters() if p.requires_grad
        }
        Path.mkdir(Path(path), exist_ok=True)
        torch.save(state_dict, Path(path).joinpath("adapter_model.bin"))
