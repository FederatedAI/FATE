from transformers.models.bert import BertModel
from torch.nn import Module
from federatedml.util import LOGGER


class PretrainedBert(Module):

    def __init__(self, pretrained_model_name_or_path: str = 'bert-base-uncased', freeze_weight=False):
        """
        A pretrained Bert Model based on transformers
        Parameters
        ----------
        pretrained_model_name_or_path: string, specify the version of bert pretrained model,
            for all available bert model, see:
                https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.#model-variations
            or it can be a path to downloaded bert model
        freeze_weight: bool, freeze weight or not when training. if True, bert model will not be added to parameters,
                       and skip grad calculation
        """

        super(PretrainedBert, self).__init__()
        self.pretrained_model_str = pretrained_model_name_or_path
        self.freeze_weight = freeze_weight
        LOGGER.info(
            'if you are using non-local models, it will download the pretrained model and will take'
            'some time')
        self.model = BertModel.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_str)
        if self.freeze_weight:
            self.model.requires_grad_(False)

    def forward(self, x):
        return self.model(x)

    def parameters(self, recurse: bool = True):
        if self.freeze_weight:
            return (),
        else:
            return self.model.parameters(recurse=recurse)
