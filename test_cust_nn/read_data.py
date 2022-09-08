import numpy as np
import torch as t
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

df_guest = pd.read_csv('../examples/data/anime_rec_homo_guest.csv')
df_host = pd.read_csv('../examples/data/anime_rec_homo_host.csv')
df_anime = pd.read_csv('../examples/data/anime.csv')
