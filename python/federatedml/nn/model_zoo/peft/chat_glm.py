from federatedml.nn.model_zoo.peft.parameter_efficient_finetune_llm import PEFTLM


class ChatGLMForConditionalGeneration(PEFTLM):
    def __init__(self,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 fp16: bool = True) -> None:

        super().__init__(pretrained_path=pretrained_path,
                         peft_type=peft_type,
                         peft_config=peft_config,
                         fp16=fp16)
