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


class ContengincyMatrix(object):
    """
    Compute contengincy_matrix
    """

    def compute(self, labels, pred_scores):
        total_count = len(labels)
        label_predict = list(zip(labels, pred_scores))
        unique_predicted_label = np.unique(pred_scores)
        unique_true_label = np.unique(labels)
        result_array = np.zeros([len(unique_true_label), len(unique_predicted_label)])
        for v1, v2 in label_predict:
            result_array[v1][v2] += 1
        return result_array, unique_predicted_label, unique_true_label


class DistanceMeasure(object):
    """
    Compute distance_measure
    """

    def compute(self, dist_table, inter_cluster_dist, max_radius):
        max_radius_result = max_radius
        cluster_nearest_result = []
        for j in range(0, len(dist_table)):
            arr = inter_cluster_dist[j * len(dist_table): j * 2 * len(dist_table)]
            smallest = np.inf
            smallest_index = 0
            for k in range(0, len(arr)):
                if arr[k] < smallest:
                    smallest = arr[k]
                    smallest_index = k
                if smallest_index >= j:
                    smallest_index += 1
            cluster_nearest_result.append(smallest_index)
        distance_measure_result = dict()
        for n in range(0, len(dist_table)):
            distance_measure_result[n] = [max_radius_result[n], cluster_nearest_result[n]]
        return distance_measure_result


class DaviesBouldinIndex(object):
    """
        Compute dbi，as in dbi
    """

    def compute(self, dist_table, cluster_dist):
        max_dij_list = []
        for i in range(0, len(dist_table)):
            dij_list = []
            for j in range(0, len(dist_table)):
                if j != i:
                    dij_list.append((dist_table[i] + dist_table[j]) / (cluster_dist[i + j] ** 0.5))
            max_dij = max(dij_list)
        max_dij_list.append(max_dij)
        return np.sum(max_dij_list) / len(dist_table)
