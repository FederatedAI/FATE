Evaluation
==========

This module provides some evaluation method for classification and regression. It contains:

1. AUC: Compute AUC for binary classification.
2. KS: Compute Kolmogorov-Smirnov for binary classification.
3. LIFT: Compute lift of binary classification.
4. PRECISION: Compute the precision for binary and multiple classification
5. RECALL: Compute the recall for binary and multiple classification
6. ACCURACY: Compute the accuracy for binary and multiple classification
7. EXPLAINED_VARIANCE: Compute explain variance
8. MEAN_ABSOLUTE_ERROR: Compute mean absolute error
9. MEAN_SQUARED_ERROR: Compute mean square error
10. MEAN_SQUARED_LOG_ERROR: Compute mean squared logarithmic error
11. MEDIAN_ABSOLUTE_ERROR: Compute median absolute error
12. R2_SCORE: Compute R^2 (coefficient of determination) score
13. ROOT_MEAN_SQUARED_ERROR: Compute the root of mean square error

All of the evaluation metrics above can be used for classification, while regression only support EXPLAINED_VARIANCE, MEAN_ABSOLUTE_ERROR,
MEAN_SQUARED_ERROR, MEAN_SQUARED_LOG_ERROR, MEDIAN_ABSOLUTE_ERROR, R2_SCORE, ROOT_MEAN_SQUARED_ERROR


Param
------

.. automodule:: federatedml.param.evaluation_param
   :members:
