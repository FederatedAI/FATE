
pwd=`pwd`
cwd=$(cd `dirname $0`; pwd)
cd $cwd
rm -fr osx
if [[ ! -d "osx" ]]; then
    mkdir osx
fi
mkdir  osx/bin
mkdir  osx/lib
mkdir  osx/conf
mkdir  osx/conf/broker
#mkdir  osx/conf/cluster-manager

cd ..
mvn clean package -DskipTests


if [[ ! -d "lib" ]]; then
    mkdir lib
fi


cp -r broker/target/*.jar deploy/osx/lib
cp -r broker/target/lib/* deploy/osx/lib
cp  broker/src/main/resources/*  deploy/osx/conf/broker
cp  bin/service.sh deploy/osx/
cp  bin/common.sh  deploy/osx/bin
cd  deploy
tar -czf osx.tar.gz  osx
cd $pwd
