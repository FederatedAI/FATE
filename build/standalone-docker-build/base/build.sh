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

#rm -rf ${package_dir}
mkdir -p ${package_dir}

build_python() {
  image_name="fate_${module}_base"
  image_path=${image_namespace}/${image_name}:${image_tag}

  cd ${source_dir}

  if_build_python_base=0
  if [[ -f ${package_dir}/installed_requirements.txt ]];then
    old_md5=`md5 ${package_dir}/installed_requirements.txt | awk '{print $NF}'`
    new_md5=`md5 ./python/requirements.txt | awk '{print $NF}'`
    if [[ $old_md5 == $new_md5 ]];then
      echo "[INFO] python requirements not updated"
    else
      if_build_python_base=1
      echo "[INFO] python requirements have been update"
    fi
  else
    if_build_python_base=1
    echo "[INFO] no installed requirements"
  fi
  if [[ ${if_build_python_base} -eq 1 ]];then
    echo "[INFO] start build ${module} base image"
    a=`docker images | grep "${image_namespace}/${image_name}" | grep "${image_tag}" | wc -l`
    if [[ a -ne 0 ]];then
      docker rmi ${image_path}
      echo "[INFO] already have ${image_path} image, delete it"
    fi
    cp ./python/requirements.txt ${package_dir}
    cp ./bin/install_os_dependencies.sh  ${package_dir}
    cp ./build/standalone-docker-build/base/python/init.sh ${package_dir}
    cp ./build/standalone-docker-build/base/python/Dockerfile ${package_dir}
    if [[ -f ${repo_file_path} ]];then
      cp ${repo_file_path} ${package_dir}/CentOS-Base.repo
    fi
    cd ${package_dir}
    docker build -f ./Dockerfile -t ${image_path} . --build-arg source_dir=${source_dir} --build-arg pip_index_url=${pip_index_url}
    build_status_code=${?}
    echo "[INFO] build ${module} base image done, status code: ${build_status_code}"
    if [[ ${build_status_code} -eq 0 ]];then
      cp ./requirements.txt ${package_dir}/installed_requirements.txt
    fi
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