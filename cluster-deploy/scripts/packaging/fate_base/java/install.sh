#!/bin/bash
set -e
if [[ $2 == "" ]]
then
echo "usage: $0 {install} {configurations path}."
exit
else
  source $2
fi
base="$(cd `dirname $0`; pwd)"
cdir=$javadir
cd $base
echo $base
cd ..
basedir=`pwd`
echo $basedir
if [ ! -d "${cdir}" ];then
    mkdir -p $cdir
fi
echo $cdir
cd $cdir
#source build
source_build() {
        return 0
}
#build
build(){
if [ ! -f "$basedir/packages/jdk-$javaversion-linux-x64.tar.gz" ];then
    echo "No jdk-$javaversion-linux-x64.tar.gz package can't be installed"
    exit
fi
tar -xzvf $basedir/packages/jdk-$javaversion-linux-x64.tar.gz
echo "done"
}
#configuration
config(){
return 0
}
#init service
init(){
return 0
}

case "$1" in
    install)
          source_build
          build
          config
          init
        ;;
        *)
          echo "usage: $0 {install} {configurations path}."
        exit -1
esac