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

BASEDIR=$(dirname "$0")
cd "$BASEDIR" || exit

PROTO_DIR="proto"
TARGER_DIR="generated"

generate() {
  python -m grpc_tools.protoc -I./$PROTO_DIR --python_out=./$TARGER_DIR "$1"
}

generate_all() {
  for proto in "$PROTO_DIR"/*.proto; do
    echo "protoc: $proto"
    generate "$proto"
  done
}

if [ $# -gt 0 ]; then
  generate "$1"
else
  generate_all
fi
