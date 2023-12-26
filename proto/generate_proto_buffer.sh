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

# Constants
BASEDIR=$(dirname "$0")
PROTO_DIR="osx"
TARGET_DIR="../python/fate/arch/federation/backends/osx"

# Function to display help
show_help() {
  echo "Usage: $(basename "$0") [options]"
  echo "Options:"
  echo "  -h, --help         Show this help message"
  echo "  -r, --recursive    Process directories recursively"
  echo "  -c, --clean        Clean target directory before generation"
}

# Function to check if grpc_tools is installed
check_grpc_tools_installed() {
  python3 -c "import grpc_tools.protoc" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "grpc_tools is not installed. Please install it using 'python3 -m pip install grpcio-tools'"
    exit 1
  fi
}

# Function to generate stubs
generate() {
  if grep -q "^service " "$3"; then
    # Generate both Python and gRPC stubs
    python3 -m grpc_tools.protoc -I"$1" \
      --python_out="$2" \
      --grpc_python_out="$2" \
      --mypy_out="$2" \
      "$3"
  else
    # Generate only Python stubs
    python3 -m grpc_tools.protoc -I"$1" \
      --python_out="$2" \
      --mypy_out="$2" \
      "$3"
  fi
}

# Function to clean target directory
clean_target() {
  rm -rf "$TARGET_DIR"/*
}

# Function to generate stubs for all .proto files
generate_all() {
  for proto in "${PROTO_DIR}"/*.proto; do
    generate "${PROTO_DIR}" "${TARGET_DIR}" "$proto"
  done
}

# Parse command line options
recursive=false
clean=false
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) show_help; exit 0 ;;
    -r|--recursive) recursive=true ;;
    -c|--clean) clean=true ;;
    *) echo "Unknown option: $1"; show_help; exit 1 ;;
  esac
  shift
done

# Main execution
check_grpc_tools_installed
cd "$BASEDIR" || { echo "Failed to change directory to $BASEDIR"; exit 1; }
[ "$clean" == "true" ] && clean_target
generate_all "$recursive"
echo "Generation completed"