#!/bin/bash

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
#

fate_project_base=$(cd `dirname "$(realpath "${BASH_SOURCE[0]:-${(%):-%x}}")"`; cd ../;pwd)
export FATE_PROJECT_BASE=$fate_project_base
export EGGROLL_HOME=
export PYTHONPATH=
export SPARK_HOME=

export FATE_LOG_LEVEL=INFO
export EGGROLL_LOG_LEVEL=INFO

venv=
export JAVA_HOME=
export PATH=$PATH:$JAVA_HOME/bin
source ${venv}/bin/activate

