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


