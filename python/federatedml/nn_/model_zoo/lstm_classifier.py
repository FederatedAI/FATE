import torch as t
from torch import nn

class LSTMClassifier(t.nn.Module):
    
    def __init__(self, vocab_size=10000, embed_size=64, lstm_layer=3, out_unit=1, padding_idx=0):
        super(LSTMClassifier, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=embed_size, num_layers=lstm_layer, bias=True, batch_first=True)
        self.classifier = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, (_, _) = self.lstm(self.word_embedding(x))
        embed = nn.ReLU()(out.sum(axis=1))
        return self.sigmoid(self.classifier(embed))