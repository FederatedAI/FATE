from federatedml.nn.model_zoo.pellm.parameter_efficient_llm import PELLM
from transformers import AutoConfig


class ChatGLMForConditionalGeneration(PELLM):
    enable_save_pretrained = True

    def __init__(self,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 fp16: bool = True,
                 pre_seq_len: int = None,
                 prefix_projection: bool = False) -> None:

        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.fp16 = fp16

        super().__init__(pretrained_path=pretrained_path,
                         peft_type=peft_type,
                         peft_config=peft_config)

    def init_config(self):
        self.config = AutoConfig.from_pretrained(self.config_path, trust_remote_code=True)
        self.config.pre_seq_len = self.pre_seq_len
        self.config.prefix_projection = self.prefix_projection

    def init_base_lm(self):
        super(ChatGLMForConditionalGeneration, self).init_base_lm(trust_remote_code=True)
        if self.fp16:
            self._pe_lm.half()

    def add_peft(self):
        if self.pre_seq_len:
            self._pe_lm.half()
            self._pe_lm.transformer.prefix_encoder.float()
        else:
            super(ChatGLMForConditionalGeneration, self).add_peft()
