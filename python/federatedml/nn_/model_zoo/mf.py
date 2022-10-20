import torch as t


class MatrixFactorization(t.nn.Module):

    def __init__(self, u_num, i_num, embd_dim=16):

        super(MatrixFactorization, self).__init__()
        self.u_embd = t.nn.Embedding(num_embeddings=u_num, embedding_dim=embd_dim)
        self.i_embd = t.nn.Embedding(num_embeddings=i_num, embedding_dim=embd_dim)
        self.sigmoid = t.nn.Sigmoid()

    def forward(self, input_data):
        u_idx, i_idx = input_data
        u_ = self.u_embd(u_idx)
        i_ = self.i_embd(i_idx)
        return self.sigmoid((u_ * i_).sum(dim=1))
    