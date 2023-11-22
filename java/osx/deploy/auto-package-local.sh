
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
mkdir  osx/extension
mkdir  osx/conf/broker
mkdir  osx/conf/components

cd ..
mvn clean package -DskipTests

if [[ ! -d "lib" ]]; then
    mkdir lib
fi

cp -r osx-broker/target/*.jar deploy/osx/lib
cp -r osx-broker/target/lib/* deploy/osx/lib
cp  osx-broker/src/main/resources/broker/*  deploy/osx/conf/broker
cp -r osx-broker/src/main/resources/components/* deploy/osx/conf/components
cp  bin/service.sh deploy/osx/
cp  bin/common.sh  deploy/osx/bin
cd  deploy
tar -czf osx.tar.gz  osx
cd $pwd
cp osx/lib/*   /data/projects/fate/osx-ptp/osx/lib
cd /data/projects/fate/osx-ptp/osx
sh  service.sh  restart
