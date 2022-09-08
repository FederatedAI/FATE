import numpy as np
import torch as t
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

df_guest = pd.read_csv('../examples/data/anime_homo_guest.csv')
df_host = pd.read_csv('../examples/data/anime_homo_host.csv')
df_anime = pd.read_csv('../examples/data/anime.csv')

genre_feat = list(df_anime['genre'])
rs = []
feat_set = set()
for i in genre_feat:
    if isinstance(i, str):
        feat_set.update(i.split(','))

encoder = {v: k for k, v in enumerate(list(feat_set))}
encoder[np.nan] = -1

for i in genre_feat:
    if isinstance(i, str):
        rs.append(t.Tensor([encoder[k] for k in i.split(',')]))
    else:
        rs.append(t.Tensor([len(encoder)]))

type_feat = df_anime['type']
pad_rs = pad_sequence(rs, True, 0)
pad_array = np.array(pad_rs)
df_genre = pd.DataFrame(pad_array)
df_genre['anime_id'] = df_anime['anime_id']

df_g_new = df_guest.merge(df_genre, on=['anime_id'])
df_h_new = df_host.merge(df_genre, on=['anime_id'])

df_g_new.to_csv('../examples/data/anime_rec_homo_guest.csv', index=False)
df_h_new.to_csv('../examples/data/anime_rec_homo_host.csv', index=False)