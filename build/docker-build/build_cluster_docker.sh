########################################################
# Copyright 2019-2020 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

#!/bin/bash
set -e

# cd worker_dir
BASEDIR=$(dirname "$0")
cd "$BASEDIR"
WORKING_DIR=$(pwd)

# get version
source_dir=$(cd `dirname $0`; cd ../;cd ../;pwd)
cd ${source_dir}
#git submodule foreach --recursive git pull
version=$(grep "FATE=" fate.env | awk -F '=' '{print $2}')
cd ${WORKING_DIR}



# set image PREFIX and TAG
# default PREFIX is federatedai
PREFIX="federatedai"
if [ -z "$TAG" ]; then
        TAG="${version}-release"
fi
BASE_TAG=${TAG}
source ${WORKING_DIR}/.env

# print build INFO
echo "[INFO] Build info"
echo "[INFO] Version: v"${version}
echo "[INFO] Image prefix is: "${PREFIX}
echo "[INFO] Image tag is: "${TAG}
echo "[INFO] Base image tag is: "${BASE_TAG}
echo "[INFO] source dir: "${source_dir}
echo "[INFO] Package dir is: "${WORKING_DIR}/catch/

package() {
        cd ${WORKING_DIR}

        [ -d ../package-build/build_docker.sh  ] && rm -rf ../package-build/build_docker.sh 
        cp ../package-build/build.sh ../package-build/build_docker.sh 
        sed -i 's#mvn clean package -DskipTests#docker run --rm -u $(id -u):$(id -g) -v ${source_dir}/fateboard:/data/projects/fate/fateboard --entrypoint="" maven:3.6-jdk-8 /bin/bash -c "cd /data/projects/fate/fateboard \&\& mvn clean package -DskipTests"#g' ../package-build/build_docker.sh 
        sed -i 's#bash ./auto-packaging.sh#docker run --rm -u $(id -u):$(id -g) -v ${source_dir}/eggroll:/data/projects/fate/eggroll --entrypoint="" maven:3.6-jdk-8 /bin/bash -c "cd /data/projects/fate/eggroll/deploy \&\& bash auto-packaging.sh"#g' ../package-build/build_docker.sh 

        # package all
        source ../package-build/build_docker.sh release all

        rm -rf ../package-build/build_docker.sh

        mkdir -p ${WORKING_DIR}/catch/
        cp -r ${package_dir}/* ${WORKING_DIR}/catch/
}

buildBase() {
        echo "START BUILDING BASE IMAGE"
        cd ${WORKING_DIR}
        docker build --build-arg version=${version} -f ${WORKING_DIR}/docker/base/Dockerfile -t ${PREFIX}/base-image:${BASE_TAG} ${WORKING_DIR}/catch/
        echo "FINISH BUILDING BASE IMAGE"
}

buildModule() {
        echo "START BUILDING IMAGE"
        for module in "python" "fateboard" "eggroll" "python-nn"; do
        cd ${WORKING_DIR}
                echo "### START BUILDING ${module} ###"
                docker build --build-arg PREFIX=${PREFIX} --build-arg BASE_TAG=${BASE_TAG} --no-cache -t ${PREFIX}/${module}:${TAG} -f ${WORKING_DIR}/docker/modules/${module}/Dockerfile ${WORKING_DIR}/catch/
                echo "### FINISH BUILDING ${module} ###"
                echo ""
        done
        echo "END BUILDING IMAGE"
}

pushImage() {
        ## push image
        for module in "python" "eggroll" "fateboard" "python-nn"; do
                echo "### START PUSH ${module} ###"
                docker push ${PREFIX}/${module}:${TAG}
                echo "### FINISH PUSH ${module} ###"
                echo ""
        done
}

while [ "$1" != "" ]; do
        case $1 in
        package)
                package
                ;;
        base)
                buildBase
                ;;
        modules)
                buildModule
                ;;
        all)
                package
                buildBase
                buildModule
                ;;
        push)
                pushImage
                ;;
        esac
        shift
done
