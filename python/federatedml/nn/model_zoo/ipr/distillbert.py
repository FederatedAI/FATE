from torch.nn import Module
from transformers import DistilBertForSequenceClassification
from federatedml.nn.model_zoo.ipr.sign_block import recursive_replace_layernorm


class SignDistilBert(Module):

    def __init__(self, model_path=None) -> None:
        super().__init__()
        if model_path is None:
            model_path = 'distilbert-base-uncased'
        
        self.model_path = model_path
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)

        # replace layernorm by SignatureLayerNorm
        sub_distilbert = self.model.distilbert.transformer.layer[3:]
        recursive_replace_layernorm(sub_distilbert, layer_name_set={'output_layer_norm'})

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
