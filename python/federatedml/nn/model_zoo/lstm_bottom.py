from torch import nn
import torch as t
from torch.nn import functional as F

class LSTMBottom(nn.Module):
    
    def __init__(self, vocab_size):
        super(LSTMBottom, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=16, padding_idx=0)
        self.lstm = t.nn.Sequential(
            nn.LSTM(input_size=16, hidden_size=16, num_layers=2, batch_first=True)
        )
        self.act = nn.ReLU()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        embeddings = self.word_embed(x)
        lstm_fw, _ = self.lstm(embeddings)
        
        return self.act(self.linear(lstm_fw.sum(dim=1)))     
