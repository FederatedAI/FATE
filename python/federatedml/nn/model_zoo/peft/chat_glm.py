from federatedml.nn.model_zoo.peft.parameter_efficient_finetune_llm import PEFTLM


class ChatGLMForConditionalGeneration(PEFTLM):
    def __init__(self,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 fp16: bool = True,
                 pre_seq_len: int = None,
                 prefix_projection: bool = False) -> None:

        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection

        super().__init__(pretrained_path=pretrained_path,
                         peft_type=peft_type,
                         peft_config=peft_config,
                         fp16=fp16)

    def init_config(self):
        super(ChatGLMForConditionalGeneration, self).init_config()
        self.config.pre_seq_len = self.pre_seq_len
        self.config.prefix_projection = self.prefix_projection

    def add_peft(self):
        if self.pre_seq_len:
            self.peft_lm.half()
            self.peft_lm.transformer.prefix_encoder.float()
        else:
            super(ChatGLMForConditionalGeneration, self).add_peft()
