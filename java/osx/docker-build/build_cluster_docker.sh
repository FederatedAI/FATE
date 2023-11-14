########################################################
# Copyright 2019-2020 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

#!/usr/bin/env bash
set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
WORKINGDIR=`pwd`
echo  ${WORKINGDIR}

source_code_dir=$(cd `dirname ${WORKINGDIR}`; pwd)


if [ -z "$TAG" ]; then
        TAG=latest
fi
if [ -z "$PREFIX" ]; then
        PREFIX=federatedai
fi

#version=$(grep "<fate.version>" ${source_code_dir}/pom.xml  | sed "s/[<|>]/ /g" | awk '{print $2}')
version=2.0.0
source ${WORKINGDIR}/.env

echo "Docker build"
echo "Info:"
echo "  version: ${version}"
echo "  PREFIX: ${PREFIX}"
echo "  Tag: ${TAG}"
echo "  BASEDIR: ${BASEDIR}"
echo "  WORKINGDIR: ${WORKINGDIR}"
echo "  source_code_dir: ${source_code_dir}"

#package() {
#  echo  $(id -u)
#  echo  $(id -g)
#  docker run --rm -u $(id -u):$(id -g) -v ${source_code_dir}:/data/projects/fate/FATE-Serving --entrypoint="" maven:3.6-jdk-8 /bin/bash -c "cd /data/projects/fate/FATE-Serving && mvn clean package -DskipTests"
#}

buildModule() {
  for module in "osx"
  do
      echo "### START BUILDING ${module} ###"
      docker build --build-arg version=${version} -t ${PREFIX}/${module}:${TAG} -f ${source_code_dir}/docker-build/${module}/Dockerfile ${source_code_dir}
      echo "### FINISH BUILDING ${module} ###"
      echo ""
  done;
}

pushImage() {
  ## push image
  for module in "osx"
  do
      echo "### START PUSH ${module} ###"
      docker push ${PREFIX}/${module}:${TAG}
      echo "### FINISH PUSH ${module} ###"
      echo ""
  done;
}

while [ "$1" != "" ]; do
    case $1 in
         modules)
                 buildModule
                 ;;
         build)
                 buildModule
                 ;;
         all)
                 package
                 buildModule
                 ;;
         push)
                pushImage
                ;;
    esac
    shift
done
