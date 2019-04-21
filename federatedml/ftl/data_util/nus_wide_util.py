import os

import numpy as np
import pandas as pd


def balance_X_y(XA, XB, y, seed=5):
    np.random.seed(seed)
    num_pos = np.sum(y == 1)
    num_neg = np.sum(y == -1)
    pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0]
    neg_indexes = [i for (i, _y) in enumerate(y) if _y < 0]

    print("len(pos_indexes)", len(pos_indexes))
    print("len(neg_indexes)", len(neg_indexes))
    print("num of samples", len(pos_indexes) + len(neg_indexes))
    print("num_pos:", num_pos)
    print("num_neg:", num_neg)

    if num_pos < num_neg:
        np.random.shuffle(neg_indexes)
        # randomly pick negative samples of size equal to that of positive samples
        rand_indexes = neg_indexes[:num_pos]
        indexes = pos_indexes + rand_indexes
        np.random.shuffle(indexes)
        y = [y[i] for i in indexes]
        XA = [XA[i] for i in indexes]
        XB = [XB[i] for i in indexes]

    return np.array(XA), np.array(XB), np.array(y)


def get_top_k_labels(data_dir, top_k=5):
    data_path = "NUS_WIDE/Groundtruth/AllLabels"
    label_counts = {}
    for filename in os.listdir(os.path.join(data_dir, data_path)):
        file = os.path.join(data_dir, data_path, filename)
        # print(file)
        if os.path.isfile(file):
            label = file[:-4].split("_")[-1]
            df = pd.read_csv(os.path.join(data_dir, file))
            df.columns = ['label']
            label_counts[label] = (df[df['label'] == 1].shape[0])
    label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [k for (k, v) in label_counts[:top_k]]
    return selected


def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    # get labels
    data_path = "NUS_WIDE/Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        print("df shape", df.shape)
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    # print(data_labels)
    if len(selected_label) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels
    print(selected.shape)

    # get XA, which are image low level features
    features_path = "NUS_WIDE/NUS_WID_Low_Level_Features/Low_Level_Features"
    dfs = []
    for file in os.listdir(os.path.join(data_dir, features_path)):
        if file.startswith("_".join([dtype, "Normalized"])):
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ")
            df.dropna(axis=1, inplace=True)
            print("b data features", len(df.columns))
            dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    data_XA_selected = data_XA.loc[selected.index]
    print("XA shape:", data_XA_selected.shape)  # 634 columns

    # get XB, which are tags
    tag_path = "NUS_WIDE/NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t")
    tagsdf.dropna(axis=1, inplace=True)
    data_XB_selected = tagsdf.loc[selected.index]
    print("XB shape:", data_XB_selected.shape)
    return data_XA_selected.values[:n_samples], data_XB_selected.values[:n_samples], selected.values[:n_samples]


def image_and_text_data(data_dir, selected, n_samples=2000):
    return get_labeled_data(data_dir, selected, n_samples)


if __name__ == '__main__':
    file_dir = "D:/Data/"
    # sel = get_top_k_labels(data_dir=file_dir, top_k=5)
    # print("sel", sel)

    sel = ['water', 'person', 'sky']
    # print(sel)
    Xa, Xb, y = get_labeled_data(data_dir=file_dir, selected_label=sel, n_samples=500)
    print("Xa shape:", Xa.shape)
    print("Xb shape:", Xb.shape)
    print("y shape", y.shape)

    # Xa, Xb, y = balance_X_y(Xa, Xb, y)

    y_ = []
    pos_count = 0
    neg_count = 0
    for i in range(y.shape[0]):
        if y[i, 0] == 1:
            y_.append(1)
            pos_count += 1
        else:
            y_.append(-1)
            neg_count += 1

    print("pos counts:", pos_count)
    print("neg counts:", neg_count)

    y_ = np.array(y_)
    Xa, Xb, y = balance_X_y(Xa, Xb, y_)

    num_pos = np.sum(y == 1)
    num_neg = np.sum(y == -1)

    print("Xa shape:", Xa.shape)
    print("Xb shape:", Xb.shape)
    print("y shape:", y.shape)
    print("num_pos:", num_pos)
    print("num_neg:", num_neg)
