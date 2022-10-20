import torch as t
from torch import nn
from transformers import BertModel

class BertClassifier(t.nn.Module):
    
    def __init__(self, ):
        super(BertClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained('/data/projects/cwj/standalone_fate_install_1.9.0_release/bert/bert')
        self.bert_model = self.bert_model.requires_grad_(False)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        text_embeddings = self.bert_model(x)[1]
        return self.sigmoid(self.classifier(text_embeddings))
    
    def parameters(self):
        # only classifier is learnable
        return self.classifier.parameters()
    