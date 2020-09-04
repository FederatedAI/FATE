import numpy as np
import pandas as pd
import sys

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import adjusted_rand_score

from arch.api import session
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector

import copy


class JaccardSimilarityScore(object):
    """
    Compute jaccard_similarity_score
    """

    def compute(self, labels, pred_scores):
        return jaccard_similarity_score(labels, pred_scores)


class FowlkesMallowsScore(object):
    """
    Compute fowlkes_mallows_score, as in FMI
    """

    def compute(self, labels, pred_scores):
        return fowlkes_mallows_score(labels, pred_scores)


class AdjustedRandScore(object):
    """
    Compute adjusted_rand_score，as in RI
    """

    def compute(self, labels, pred_scores):
        return adjusted_rand_score(labels, pred_scores)


class DaviesBouldinIndex(object):
    """
        Compute dbi，as in dbi
    """
    def compute(self,dist_table,cluster_dist):
        max_dij_list=[]
        for i in range(0,len(dist_table)):
            dij_list=[]
            for j in range (0,len(dist_table)):
                if j!=i:
                    dij_list.append((dist_table[i]+dist_table[j])/(cluster_dist[i+j] ** 0.5))
            max_dij=max(dij_list)
        max_dij_list.append(max_dij)
        return np.sum(max_dij_list)/len(dist_table)