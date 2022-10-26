import torch as t
from torch import nn

class NCF(nn.Module):
    
    def __init__(self, user_num, item_num, embed_dim=16):
        super(NCF, self).__init__()
        self.u_embed = nn.Embedding(num_embeddings=user_num, embedding_dim=embed_dim)
        self.i_embed = nn.Embedding(num_embeddings=item_num, embedding_dim=embed_dim)
        
        self.seq = nn.Sequential(
            nn.Linear(embed_dim*2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        u_idx, i_idx = x
        u_embed = self.u_embed(u_idx)
        i_embed = self.i_embed(i_idx)
        in_embed = t.concat([u_embed, i_embed], dim=1)
        out = self.seq(in_embed)
        return out.flatten()
