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
set -e
set -x

basepath=$(cd `dirname $0`;pwd)
fatepath=$(cd $basepath/..;pwd)
cd ${fatepath}


#eggroll_git_url=`grep -A 3 '"eggroll"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
#eggroll_git_branch=`grep -A 3 '"eggroll"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
#echo "[INFO] Git clone eggroll submodule source code from ${eggroll_git_url} branch ${eggroll_git_branch}"
#if [[ -e "eggroll" ]];then
#    while [[ true ]];do
#        read -p "The eggroll directory already exists, delete and re-download? [y/n] " input
#        case ${input} in
#        [yY]*)
#                echo "[INFO] Delete the original eggroll"
#                rm -rf eggroll
#                git clone ${eggroll_git_url} -b ${eggroll_git_branch} eggroll
#                break
#                ;;
#        [nN]*)
#                echo "[INFO] Use the original eggroll"
#                break
#                ;;
#        *)
#                echo "Just enter y or n, please."
#                ;;
#        esac
#    done
#else
#    git clone ${eggroll_git_url} -b ${eggroll_git_branch} eggroll
#fi

cd ${fatepath}
fateboard_git_url=`grep -A 3 '"fateboard"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
fateboard_git_branch=`grep -A 3 '"fateboard"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
echo "[INFO] Git clone fateboard submodule source code from ${fateboard_git_url} branch ${fateboard_git_branch}"
if [[ -e "fateboard" ]];then
    while [[ true ]];do
        read -p "The fateboard directory already exists, delete and re-download? [y/n] " input
        case ${input} in
        [yY]*)
                echo "[INFO] Delete the original fateboard"
                rm -rf fateboard
                git clone ${fateboard_git_url} -b ${fateboard_git_branch} fateboard
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
    git clone ${fateboard_git_url} -b ${fateboard_git_branch} fateboard
fi

cd ${fatepath}

init() {
    cd ${fatepath}
    mkdir -p ${basepath}/arch
    cp -r arch/conf ${basepath}/arch
    cp -r arch/api ${basepath}/arch
    cp -r arch/transfer_variables ${basepath}/arch
    cp -r arch/standalone ${basepath}/arch
    cp fate.env requirements.txt RELEASE.md ${basepath}
    cp -r  federatedml examples fate_flow  federatedrec ${basepath}
    #docker run -v ${fatepath}/fateboard:/data/projects/fate/fateboard  --entrypoint="" maven:3.6-jdk-8 /bin/bash -c "cd /data/projects/fate/fateboard && mvn clean package -DskipTests"
    cd ${fatepath}/fateboard
    mvn clean package
    cd -
    if [ ! -d "${basepath}/fateboard" ];then
       mkdir -p ${basepath}/fateboard
    fi
    version=$(grep -E -m 1 -o "<version>(.*)</version>" ${fatepath}/fateboard/pom.xml| tr -d '[\\-a-z<>//]' | awk -F "version" '{print $2}')
    cp ${fatepath}/fateboard/target/fateboard-${version}.jar  ${basepath}/fateboard
    cd ${basepath}/fateboard
    if [ ! -f "fateboard.jar" ];then
       ln -s fateboard-$version.jar fateboard.jar
    fi
    if [ ! -d "conf" ];then
       mkdir conf
    fi 
    if [ ! -d "ssh" ];then
       mkdir ssh
    fi
    cp ${fatepath}/fateboard/src/main/resources/application.properties ./conf
    touch ./ssh/ssh.properties

    cd ${basepath}
    sed -i.bak "s#^MarkupSafe==.*#MarkupSafe==1.1.1#g" ./requirements.txt
    rm  ./requirements.txt.bak
    tar -cf ./docker/python/fate.tar arch federatedml  examples fate_flow  federatedrec fate.env requirements.txt RELEASE.md

    logPath="./fate/log"
    if [ ! -d "$logPath" ]; then
     mkdir -p "$logPath"
    fi

    dataPath="./fate/data" 
    if [ ! -d "$dataPath" ]; then
     mkdir -p "$dataPath"
    fi
    cp -r ./fate_flow/* ./fate/data

    sed -i.bak "s#^fateflow.url=.*#fateflow.url=http://python:9380#g" ./fateboard/conf/application.properties
    sed -i.bak "s#^fateboard.datasource.jdbc-url=.*#fateboard.datasource.jdbc-url=jdbc:sqlite:/fate/fate_flow/fate_flow_sqlite.db#g" ./fateboard/conf/application.properties
    sed -i.bak "s#^spring.datasource.driver-Class-Name=.*#spring.datasource.driver-Class-Name=org.sqlite.JDBC#g" ./fateboard/conf/application.properties
    tar -cf ./docker/fateboard/fateboard.tar fateboard

    docker-compose -f ./docker/docker-compose-build.yml up -d
    docker restart fate_python
    sleep 5
    docker restart fate_fateboard
    rm -rf arch federatedml  examples fate_flow  federatedrec fate.env requirements.txt RELEASE.md
    rm docker/python/fate.tar
#    rm docker/python/requirements.txt
    rm docker/fateboard/fateboard.tar

    cd ${basepath}
    rm -rf ./fateboard

}
start() {
    docker start `docker ps -a | grep -i "docker_python" | awk '{print $1}'`
    docker start `docker ps -a | grep -i "docker_fateboard" | awk '{print $1}'`
}
stop(){
    docker stop `docker ps -a | grep -i "docker_python" | awk '{print $1}'`
    docker stop `docker ps -a | grep -i "docker_fateboard" | awk '{print $1}'`

}
case "$1" in
    init)
        init
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    *)
        echo "usage: $0 {init|start|stop}."
        exit -1
esac
