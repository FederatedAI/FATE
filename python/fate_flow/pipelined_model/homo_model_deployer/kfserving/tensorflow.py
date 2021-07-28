#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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
#


import kfserving
import os
import tempfile
import tensorflow

from .base import KFServingDeployer


class TFServingKFDeployer(KFServingDeployer):
    def _do_prepare_model(self):
        working_dir = tempfile.mkdtemp()
        # TFServing expects an "version string" like 0001 in the model base path
        local_folder = os.path.join(working_dir, '0001')
        tensorflow.saved_model.save(self.model_object, local_folder)
        return working_dir

    def _do_prepare_predictor(self):
        self.isvc.spec.predictor.tensorflow = kfserving.V1beta1TFServingSpec(storage_uri=self.storage_uri)
