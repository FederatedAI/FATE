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

download(){
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

}

init() {
    download
    cd ${fatepath}
    cp -r  bin conf examples fate.env python RELEASE.md   ${basepath}
    cd ${basepath}
    sed -i "s#work_mode: 1#work_mode: 0#g" ${basepath}/conf/service_conf.yaml
    #sed -i.bak "s#^MarkupSafe==.*#MarkupSafe==1.1.1#g" ${basepath}/python/requirements.txt

    cd ${fatepath}/fateboard
    mvn clean package
    cd ${basepath}
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
    cp ${fatepath}/fateboard/src/main/resources/application.properties ${basepath}/fateboard/conf
    touch ${basepath}/fateboard/ssh/ssh.properties

    sed -i "s#^fateflow.url=.*#fateflow.url=http://localhost:9380#g" ${basepath}/fateboard/conf/application.properties
    sed -i "s#^fateboard.datasource.jdbc-url=.*#fateboard.datasource.jdbc-url=jdbc:sqlite:/fate/python/fate_flow/fate_flow_sqlite.db#g" ${basepath}/fateboard/conf/application.properties
    sed -i "s#^spring.datasource.driver-Class-Name=.*#spring.datasource.driver-Class-Name=org.sqlite.JDBC#g" ${basepath}/fateboard/conf/application.properties

    cd ${basepath}
    wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/jdk-8u192.tar.gz
    tar -cf ./docker/fate/fate.tar bin conf examples fate.env python RELEASE.md fateboard  jdk-8u121-linux-x64.tar.gz
    rm -rf bin conf examples fate.env python RELEASE.md fateboard jdk-8u121-linux-x64.tar.gz

    cd ${basepath}/docker/fate
    docker build -t fate:latest .
    docker run -d -p 8888:8080 fate:latest /bin/bash
    rm docker/fate/fate.tar

    docker ps -a

}

start() {
    docker start `docker ps -a | grep -i "docker_fate" | awk '{print $1}'`
}

stop(){
    docker stop `docker ps -a | grep -i "docker_fate" | awk '{print $1}'`
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
