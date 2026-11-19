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

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      echo "Usage: $(basename "$0")"
      echo "Regenerates the checked-in Python OSX protobuf and gRPC stubs."
      exit 0
      ;;
    -r|--recursive|-c|--clean) ;; # Generation now safely overwrites only its outputs.
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

python3 -m grpc_tools.protoc \
  -Ifate/arch/federation/backends/osx=proto/osx \
  --python_out=python --grpc_python_out=python proto/osx/osx.proto
