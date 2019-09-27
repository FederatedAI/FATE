########################################################
# Copyright 2019-2020 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

#!/bin/bash
set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
WORKINGDIR=`pwd`

# fetch fate-python image
source .env

localization=false

buildBase() {
  echo "START BUILDING BASE IMAGE"
  cd ${WORKINGDIR}
  
  # docker build context 
  cat >> ../.dockerignore <<EOF
*
!cluster-deploy/scripts/fate-base
!requirements.txt
EOF

  if [ "$localization" = true ]
  then
    docker build -f docker/base/Dockerfile-cn -t ${PREFIX}/base-image:${BASE_TAG} ..
  else
    docker build -f docker/base/Dockerfile -t ${PREFIX}/base-image:${BASE_TAG} ..
  fi
  
  rm -f ../.dockerignore
  
  echo "FINISH BUILDING BASE IMAGE"
}

buildBuilder() {
  echo "START BUILDING STORAGE-SERVICE BUILDER"
  cd ${WORKINGDIR}
  if [ "$localization" = true ]
  then
    docker build --build-arg PREFIX=${PREFIX} --build-arg BASE_TAG=${BASE_TAG} -f docker/builders/storage-service-builder/Dockerfile-cn -t ${PREFIX}/storage-service-builder:${BUILDER_TAG} docker/builders/storage-service-builder/
  else
    if [ -f ../.dockerignore ]
    then
      rm ../.dockerignore
    fi
    docker build --build-arg PREFIX=${PREFIX} --build-arg BASE_TAG=${BASE_TAG} -f docker/builders/storage-service-builder/Dockerfile -t ${PREFIX}/storage-service-builder:${BUILDER_TAG} .. 
  fi
  
  echo "FINISH BUILDING STORAGE-SERVICE BUILDER"
}

buildModule() {
  ## User maven to build all jar target
  echo "### START BUILDING MODULES JAR FILES ###"
  docker run -v ${WORKINGDIR}/../arch:/data/projects/fate/arch --entrypoint="" maven:3.6-jdk-8 /bin/bash -c "cd /data/projects/fate/arch && mvn clean package -DskipTests"

  docker run -v ${WORKINGDIR}/../fateboard:/data/projects/fate/fateboard -v ${WORKINGDIR}/../arch:/data/projects/fate/arch --entrypoint="" maven:3.6-jdk-8 /bin/bash -c "cd /data/projects/fate/fateboard && mvn clean package -DskipTests"

  docker run -v ${WORKINGDIR}/../fate-serving:/data/projects/fate/fate-serving -v ${WORKINGDIR}/../arch:/data/projects/fate/arch --entrypoint="" maven:3.6-jdk-8 /bin/bash -c "cd /data/projects/fate/fate-serving && mvn clean package -DskipTests"

  if [ ! -f "/bin/gtar" ]; then
      ln -s /bin/tar /bin/gtar 
  fi

  cd ${WORKINGDIR}/../cluster-deploy/scripts && bash auto-packaging.sh

  # build modules according to the dir-tree
  cd ${WORKINGDIR}/../

  for module in "federation" "proxy" "roll" "python" "meta-service" "serving-server" "fateboard"
  do
      echo "### START BUILDING ${module^^} ###"
      cp -r ./cluster-deploy/example-dir-tree/${module}/* ${WORKINGDIR}/docker/modules/${module}/
      docker build --build-arg PREFIX=${PREFIX} --build-arg BASE_TAG=${BASE_TAG} -t ${PREFIX}/${module}:${TAG} -f ${WORKINGDIR}/docker/modules/${module}/Dockerfile ${WORKINGDIR}/docker/modules/${module}
      echo "### FINISH BUILDING ${module^^} ###"
      echo ""
  done;

  ## build egg/storage-service (also copy python module to egg)
  echo "### START BUILDING EGG ###"
  ## init submodule lmdb-safe
  git submodule update --init arch/eggroll/storage-service-cxx/third_party/lmdb-safe
  mkdir -p ${WORKINGDIR}/docker/modules/egg/egg-service

  cp -r ./cluster-deploy/example-dir-tree/egg/* ${WORKINGDIR}/docker/modules/egg/egg-service/

  mkdir -p ${WORKINGDIR}/docker/modules/egg/egg-processor
  cp -r ./cluster-deploy/example-dir-tree/python/* ${WORKINGDIR}/docker/modules/egg/egg-processor/

  mkdir -p ${WORKINGDIR}/docker/modules/egg/storage-service
  cp -r ./cluster-deploy/example-dir-tree/storage-service-cxx/* ${WORKINGDIR}/docker/modules/egg/storage-service/

  ## copy lmdb-safe submodule
  cp -r arch/eggroll/storage-service-cxx/third_party/lmdb-safe/* ${WORKINGDIR}/docker/modules/egg/storage-service/third_party/lmdb-safe/

  docker build --build-arg PREFIX=${PREFIX} --build-arg BASE_TAG=${BASE_TAG} --build-arg BUILDER_TAG=${BUILDER_TAG} -t ${PREFIX}/egg:${TAG} -f ${WORKINGDIR}/docker/modules/egg/Dockerfile ${WORKINGDIR}/docker/modules/egg
  echo "### FINISH BUILDING EGG ###"
  echo ""
}

pushImage() {
  ## push image
  for module in "federation" "proxy" "roll" "python" "meta-service" "serving-server" "fateboard" "egg"
  do
      echo "### START PUSH ${module^^} ###"
      docker push ${PREFIX}/${module}:${TAG}
      echo "### FINISH PUSH ${module^^} ###"
      echo ""
  done;
}

while [ "$1" != "" ]; do
    case $1 in
         --useChineseMirror)
                 localization=true
                 ;;
         base)
                 buildBase
                 ;;
         builder)
                 buildBuilder
                 ;;
         modules)
                 buildModule
                 ;;
         all)
                 buildBase
                 buildBuilder
                 buildModule
                 ;;
         push)
                pushImage
                ;;
    esac
    shift
done
