# Evaluation

The evaluation support metrics for classification, regression, tasks. You can use our default set of metrics or use specified metrics.

## Default Metric Set

We support following default metric set for binary, multi-class classification, regression tasks:

- Binary Classification
  - AUC
  - KS
  - Confusion Matrix
  - Gain
  - Lift
  - Precision Table
  - Recall Table
  - Accuracy Table
  - FScore Table

- Multi Classification
  - Accuracy
  - Precision
  - Recall

- Regression
   - RMSE
   - MAE
   - MSE
   - R2Score


Specify them in the 'default_eval_setting' parameter: 'binary', 'regression', 'multi'

## Metrics

You can also set metrics you want to use in the 'metrics' parameter. These metrics are available:

    auc
    multi_accuracy
    multi_recall
    multi_precision
    binary_accuracy
    binary_recall
    binary_precision
    multi_f1_score
    binary_f1_score
    ks
    confusion_matrix
    lift
    gain
    biclass_precision_table
    biclass_recall_table
    biclass_accuracy_table
    fscore_table
    rmse
    mse
    mae
    r2_score