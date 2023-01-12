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
from typing import Dict
from ..entity.component_structures import ArtifactSpec


class ComponentStageSchedule(object):
    @classmethod
    def get_stage(cls, input_artifacts: Dict[str, ArtifactSpec]):
        """
        stage in [train, predict, default]
        possible:
            train_data, validate_data: stage = train
            test_data: stage = predict
            data: stage = default
            train_data & input_model: stage = train
            test_data & input_model: stage = train
        """
        stage = set()
        data_type = 0
        for input_key, artifact in input_artifacts.items():
            if input_key == "train_data":
                data_type |= 1
            elif input_key == "validate_data":
                data_type |= 2
            elif input_key == "test_data":
                data_type |= 4
            else:
                data_type |= 8

            if not artifact.stages:
                continue

            stage_list = artifact.stages
            if len(stage_list) == 1:
                stage.add(stage_list[0])

        if len(stage):
            return list(stage)[0]

        """the following is warm_start"""
        if data_type & 1:
            return "train"
        if data_type & 4:
            return "predict"

        """the following is component which only has data key"""
        return "default"
