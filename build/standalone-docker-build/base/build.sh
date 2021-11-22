#!/bin/bash

set -e

source_dir=$(cd `dirname $0`; cd ../;cd ../;cd ../;pwd)
echo "[INFO] source dir: ${source_dir}"
cd ${source_dir}
source ./bin/common.sh
support_modules=(python)

version=`grep "FATE=" fate.env | awk -F '=' '{print $2}'`
package_dir_name="docker_fate_base_${version}"
package_dir=${source_dir}/${package_dir_name}

image_namespace="federatedai"
image_tag=${version}


echo "[INFO] build info"
echo "[INFO] version: "${version}
echo "[INFO] module: "${module}
echo "[INFO] repo file: "${repo_file_path}
echo "[INFO] pip index url: "${pip_index_url}
echo "[INFO] image namespace: "${image_namespace}
echo "[INFO] image tag: "${image_tag}
echo "[INFO] package output dir is "${package_dir}

rm -rf ${package_dir}
mkdir -p ${package_dir}

build_python() {
  echo "[INFO] start ${module} base image"
  image_name="fate_${module}_base"
  image_path=${image_namespace}/${image_name}:${image_tag}

  cd ${source_dir}

  if [[ ${kernel} == "Darwin" ]];then
    dest_md5=`md5 ./python/requirements.txt | awk '{print $NF}'`
  else
    dest_md5=`md5sum ./python/requirements.txt | awk '{print $1}'`
  fi
  image_id=`docker images -q ${image_path}`
  if_build_python_base=1
  if [[ -n ${image_id} ]];then
    echo "[INFO] already have image, image id: ${image_id}"
    old_md5=`docker inspect ${image_id} --format '{{ index .Config.Labels "python-requirements-hash"}}' | awk -F ':' '{print $2}'`
    if [[ $old_md5 == $dest_md5 ]];then
      echo "[INFO] python requirements not updated, md5: ${old_md5}"
      if_build_python_base=0
    else
      echo "[INFO] python requirements have been update"
      docker rmi ${image_path}
      echo "[INFO] delete image ${image_id}"
    fi
  else
    echo "[INFO] no image, build it"
  fi

  if [[ ${if_build_python_base} -eq 1 ]];then
    echo "[INFO] start build ${module} base image"
    cp ./python/requirements.txt \
      ./bin/install_os_dependencies.sh \
      ./build/standalone-docker-build/base/python/init.sh \
      ./build/standalone-docker-build/base/python/Dockerfile ${package_dir}
    if [[ -f ${repo_file_path} ]];then
      cp ${repo_file_path} ${package_dir}/CentOS-Base.repo
      repo_file="CentOS-Base.repo"
    else
      repo_file=""
    fi
    cd ${package_dir}
    docker build -t ${image_path} . \
                --build-arg source_dir=${source_dir} \
                --build-arg pip_index_url=${pip_index_url} \
                --build-arg repo_file=${repo_file} \
                --label python-requirements-hash="md5:"${dest_md5}
    echo "[INFO] build ${module} base image done, status code: ${?}"
  else
    echo "[INFO] pass build ${module} base image"
  fi
}

usage() {
    echo "usage: $0 -r {repo file path} -i {pip index url}"
}

while getopts "m:r:i:" opt; do
  case $opt in
    m)
      module=$OPTARG
      ;;
    r)
      repo_file_path=$OPTARG
      ;;
    i)
      pip_index_url=$OPTARG
      ;;
    :)
      echo "option -$OPTARG requires an argument."
      usage
      exit 1
      ;;
    ?)
      echo "invalid option: -$OPTARG index:$OPTIND"
      ;;
  esac
done
build_${module}