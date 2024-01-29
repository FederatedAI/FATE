from fate.components.components.nn.loader import ModelLoader
from fate.components.components.nn.torch.base import Sequential, load_seq
from fate.components.components.nn.torch import nn
from fate.ml.nn.trainer.trainer_base import TrainingArguments
from transformers import Seq2SeqTrainingArguments


loader = ModelLoader('multi_model', 'Multi')

b = Sequential(
    nn.Linear(10, 10),
    nn.Sigmoid()
)

a = Sequential(
    ModelLoader('multi_model', 'Multi')
)