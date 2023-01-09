#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from sklearn import metrics


class BinaryEvaluator(object):
    def __init__(self):
        self._auc = None

    def fit(self, ctx, y_true, y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        self._auc = metrics.auc(fpr, tpr)

        self._report(ctx)

    def _report(self, ctx):
        ctx.metrics.log_auc("auc", self._auc)
