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

set -e

if [[ ${is_root} -eq 1 ]]; then
  command=""
else
  sudo -nv 2>&1
  if [[ $? -eq 0 ]]; then
    command="sudo"
  else
    echo "[INFO] no sudo permission"
    exit
  fi
fi

system=$(sed -e '/"/s/"//g' /etc/os-release | awk -F= '/^NAME/{print $2}')
echo ${system}

function ln_lib() {
  local lib_name=$1
  echo "[INFO] deal $lib_name lib"
  so_name="lib$lib_name.so"
  cd /usr/lib/x86_64-linux-gnu
  if [ ! -f $so_name ] && [ ! -L $so_name ]; then
    so_file=$(ldd /usr/bin/openssl | grep $so_name | awk '{print $1}')
    if [[ -n $so_file ]]; then
      $command ln -s $so_file $so_name
      echo "[INFO] ln $so_file"
    else
      echo "[INFO] can not found openssl $so_name"
      #$command apt-get install -y libssl1.0.0 || echo ""
    fi
  fi
}

case "${system}" in
"CentOS Linux")
  echo "CentOS System"
  $command yum -y install gcc gcc-c++ make openssl-devel supervisor gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-devel snappy snappy-devel zlib zlib-devel bzip2 bzip2-devel lz4-devel libasan lsof
  ;;
"Ubuntu")
  echo "Ubuntu System"
  $command apt-get install -y gcc g++ make openssl supervisor libgmp-dev libmpfr-dev libmpc-dev libaio1 libaio-dev numactl autoconf automake libtool libffi-dev libssl-dev liblz4-1 liblz4-dev liblz4-tool zlib1g zlib1g-dev
  cd /usr/lib/x86_64-linux-gnu
  ln_lib "ssl"
  ln_lib "crypto"
  ;;
*)
  echo "Not support this system."
  exit -1
  ;;
esac
