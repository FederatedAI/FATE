#!/usr/bin/env bash

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

BASEDIR=$(dirname "$0")/..
cd $BASEDIR


python -m grpc_tools.protoc -Iarch/proto --python_out=./arch/api/proto  fate-data-structure.proto

python -m grpc_tools.protoc -Iarch/proto --python_out=./arch/api/proto  default-empty-fill.proto

python -m grpc_tools.protoc -Iarch/proto --python_out=./arch/api/proto  fate-meta.proto

python -m grpc_tools.protoc -Iarch/proto --python_out=./arch/api/proto --grpc_python_out=./arch/api/proto inference_service.proto

python -m grpc_tools.protoc -Iarch/proto --python_out=./arch/api/proto --grpc_python_out=./arch/api/proto model_service.proto

python -m grpc_tools.protoc -Iarch/proto --python_out=./arch/api/proto --grpc_python_out=./arch/api/proto proxy.proto

python -m grpc_tools.protoc -Iarch/driver/federation/proto --python_out=./arch/api/proto --grpc_python_out=./arch/api/proto federation.proto
