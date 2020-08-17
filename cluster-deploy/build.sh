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
source_dir=$(cd `dirname $0`; cd ../;pwd)
echo ${source_dir}
if [[ -n ${1} ]]; then
    version_tag=$1
else
    version_tag="rc"
fi

cd ${source_dir}
version=`grep "FATE=" fate.env | awk -F '=' '{print $2}'`
package_dir_name="FATE_install_"${version}
package_dir=${source_dir}/cluster-deploy/${package_dir_name}
echo "[INFO] Build info"
echo "[INFO] version: "${version}
echo "[INFO] version tag: "${version_tag}
echo "[INFO] Package output dir is "${package_dir}
rm -rf ${package_dir} ${package_dir}-${version_tag}".tar.gz"
mkdir -p ${package_dir}

eggroll_git_url=`grep -A 3 '"eggroll"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
eggroll_git_branch=`grep -A 3 '"eggroll"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
fateboard_git_url=`grep -A 3 '"fateboard"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
fateboard_git_branch=`grep -A 3 '"fateboard"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`

echo "[INFO] Package fate start"
mkdir -p ${package_dir}/python/
cp fate.env requirements.txt RELEASE.md sitecustomize.py ${package_dir}/python/

# pass copy conf
cp -r arch fate_arch compatible_hook examples federatedml fate_flow fate_client ${package_dir}/python/
cp -r bin  ${package_dir}/
echo "[INFO] Package fate done"

echo "[INFO] Compress start"
cd ${package_dir}
tar czf python.tar.gz ./python
rm -rf python fateboard eggroll
exit

echo "[INFO] Package fateboard start"
cd ${source_dir}
echo "[INFO] Git clone fateboard submodule source code from ${fateboard_git_url} branch ${fateboard_git_branch}"
if [[ -e "fateboard" ]];then
    while [[ true ]];do
        read -p "The fateboard directory already exists, delete and re-download? [y/n] " input
        case ${input} in
        [yY]*)
                echo "[INFO] Delete the original fateboard"
                rm -rf fateboard
                git clone ${fateboard_git_url} -b ${fateboard_git_branch} --depth=1 fateboard
                break
                ;;
        [nN]*)
                echo "[INFO] Use the original fateboard"
                break
                ;;
        *)
                echo "Just enter y or n, please."
                ;;
        esac
    done
else
    git clone ${fateboard_git_url} -b ${fateboard_git_branch} --depth=1 fateboard
fi
cd ./fateboard
fateboard_version=$(grep -E -m 1 -o "<version>(.*)</version>" ./pom.xml | tr -d '[\\-a-z<>//]' | awk -F "version" '{print $2}')
echo "[INFO] fateboard version "${fateboard_version}
mvn clean package -DskipTests
mkdir -p ${package_dir}/fateboard/conf
mkdir -p ${package_dir}/fateboard/ssh
cp ./target/fateboard-${fateboard_version}.jar ${package_dir}/fateboard/
cp ./bin/service.sh ${package_dir}/fateboard/
cp ./src/main/resources/application.properties ${package_dir}/fateboard/conf/
cd ${package_dir}/fateboard
touch ./ssh/ssh.properties
ln -s fateboard-${fateboard_version}.jar fateboard.jar
echo "[INFO] Package fateboard done"

echo "[INFO] Package fateboard start"
cd ${source_dir}
echo "[INFO] Git clone eggroll submodule source code from ${eggroll_git_url} branch ${eggroll_git_branch}"
if [[ -e "eggroll" ]];then
    while [[ true ]];do
        read -p "The eggroll directory already exists, delete and re-download? [y/n] " input
        case ${input} in
        [yY]*)
                echo "[INFO] Delete the original eggroll"
                rm -rf eggroll
                git clone ${eggroll_git_url} -b ${eggroll_git_branch} --depth=1 eggroll
                break
                ;;
        [nN]*)
                echo "[INFO] Use the original eggroll"
                break
                ;;
        *)
                echo "Just enter y or n, please."
                ;;
        esac
    done
else
    git clone ${eggroll_git_url} -b ${eggroll_git_branch} --depth=1 eggroll
fi
cd ./eggroll
cd ./deploy
sh ./auto-packaging.sh
mkdir -p ${package_dir}/eggroll
mv ${source_dir}/eggroll/eggroll.tar.gz ${package_dir}/eggroll/
cd ${package_dir}/eggroll/
tar xzf eggroll.tar.gz
rm -rf eggroll.tar.gz
echo "[INFO] Package eggroll done"

echo "[INFO] Compress start"
cd ${package_dir}
tar czf python.tar.gz ./python
tar czf fateboard.tar.gz ./fateboard
tar czf eggroll.tar.gz ./eggroll
rm -rf python fateboard eggroll
echo "[INFO] Compress done"
echo "[INFO] A total of `ls ${package_dir} | wc -l | awk '{print $1}'` packages:"
ls -lrt ${package_dir}
cd ${source_dir}/cluster-deploy/
tar czf ${package_dir_name}-${version_tag}".tar.gz" ${package_dir_name}
rm -rf ${package_dir}