import numpy as np


def generate_random_matrix(r, c):
    a1 = np.random.rand(r, c)
    a2 = -np.random.rand(r, c)
    return a1 + a2


def generate_random_3_dim_matrix(r, c, z):
    a1 = np.random.rand(r, c, z)
    a2 = -np.random.rand(r, c, z)
    return a1 + a2


def share(X):
    num_dim = len(X.shape)
    if num_dim == 2:
        n, d = X.shape
        X0 = generate_random_matrix(n, d)
    elif num_dim == 3:
        n, d, k = X.shape
        X0 = generate_random_3_dim_matrix(n, d, k)
    else:
        raise TypeError("does not support {0} num of dim".format(num_dim))
    X1 = X - X0
    return X0, X1


def reconstruct(share0, share1):
    return share0 + share1


def local_compute_alpha_beta_share(share_map):
    As = share_map["As"]
    Bs = share_map["Bs"]
    Xs = share_map["Xs"]
    Ys = share_map["Ys"]

    alpha_s = Xs - As
    beta_s = Ys - Bs

    return alpha_s, beta_s


def compute_add_share(share_0, share_1):
    return share_0 + share_1


def compute_minus_share(share_0, share_1):
    return share_0 - share_1


def compute_matmul_share(alpha_0, alpha_1, beta_0, beta_1, share_map):
    alpha = reconstruct(alpha_0, alpha_1)
    beta = reconstruct(beta_0, beta_1)

    As = share_map["As"]
    Bs = share_map["Bs"]
    Cs = share_map["Cs"]
    i = 0 if share_map["is_party_a"] else 1

    Zs = i * np.matmul(alpha, beta) + np.matmul(As, beta) + np.matmul(alpha, Bs) + Cs
    return Zs


def compute_sum_of_matmul_share(alpha_0, alpha_1, beta_0, beta_1, share_map, axis=None):
    Zs = compute_matmul_share(alpha_0, alpha_1, beta_0, beta_1, share_map)
    return np.sum(Zs, axis=axis)


def compute_multiply_share(alpha_0, alpha_1, beta_0, beta_1, share_map):
    alpha = reconstruct(alpha_0, alpha_1)
    beta = reconstruct(beta_0, beta_1)

    As = share_map["As"]
    Bs = share_map["Bs"]
    Cs = share_map["Cs"]
    i = 0 if share_map["is_party_a"] else 1

    Zs = i * np.multiply(alpha, beta) + np.multiply(As, beta) + np.multiply(alpha, Bs) + Cs
    return Zs


def compute_sum_of_multiply_share(alpha_0, alpha_1, beta_0, beta_1, share_map, axis=None):
    Zs = compute_multiply_share(alpha_0, alpha_1, beta_0, beta_1, share_map)
    return np.sum(Zs, axis=axis)
