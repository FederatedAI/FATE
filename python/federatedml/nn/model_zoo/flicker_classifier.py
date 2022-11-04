import torch as t
from torch import nn

class FlickerClassifier(nn.Module):
    
    def __init__(self, vocab_size,word_embed_size=8):
        super(FlickerClassifier, self).__init__()
        
        # 图像部分
        self.cv_seq = t.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3),
            nn.AvgPool2d(kernel_size=5)
        )
        self.fc = t.nn.Sequential(
            nn.Linear(1176, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        # NLP部分
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embed_size)
        self.lstm_seq = nn.LSTM(input_size=word_embed_size, hidden_size=word_embed_size, batch_first=True)
        # 分类器
        self.classifier_seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(word_embed_size + 8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        image_feat = x[0]
        word_feat = x[1]
        
        image_feat = self.fc(self.cv_seq(image_feat).flatten(start_dim=1))
        word_feat, _ = self.lstm_seq(self.word_embedding(x[1]))
        word_feat = word_feat.sum(dim=1)
        return self.classifier_seq(t.cat([image_feat, word_feat], axis=1)).flatten()
        
