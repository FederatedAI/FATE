from federatedml.nn.backend.fate_torch.operation import  *
from federatedml.nn.backend.fate_torch import Sequential
from federatedml.nn.backend.fate_torch import Linear, Embedding, ReLU, LSTM, Sigmoid
from federatedml.nn.backend.fate_torch import xavier_normal_
from federatedml.nn.backend.fate_torch import Adam

a = torch.randint(0, 10, (2, 3, 4))
dtype = Astype('float64')
a = dtype(a).requires_grad_(True)
reshape_1 = Reshape((2, -1))
reshape_2 = Reshape((2, 4, -1))
print(reshape_1(a))
print(reshape_2(a))
flatten = Flatten(start_dim=-1, end_dim=-1)
dtype_2 = Astype('int64')
usqe = Unsqueeze(dim=1)
a = usqe(a)
sq = Squeeze()
a = sq(a)
sum_ = Sum(dim=0)
b = sum_(a)

seq = Sequential(
    Astype('int64'),
    Embedding(50, 50),
    Linear(50, 32),
    ReLU(),
    Linear(32, 16),
    Squeeze()
)

test_data = torch.randint(0, 50, (100, 1)).type(torch.float64)

test_seq_data = torch.randint(0, 50, (100, 30))
seq2 = Sequential(
    Astype('int64'),
    Embedding(50, 50, padding_idx=-1),
    LSTM(input_size=50, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True),
    Index(index=0),  # take network output, ignore hidden states
    Select(1, -1),
    ReLU(),
    Linear(32, 16),
    ReLU(),
    Linear(16, 1),
    Sigmoid()
)
xavier_normal_(seq2)  # register initializer
adam = Adam(seq2, )  # register adam optimizer

# seq3 = recover_sequential_from_dict(seq2.to_dict())