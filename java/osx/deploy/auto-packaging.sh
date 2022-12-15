
pwd=`pwd`
cwd=$(cd `dirname $0`; pwd)
cd $cwd
rm -fr  osx
mkdir  osx
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

#cp -r cluster-manager/target/lib/* deploy/firework/lib
#cp -r cluster-manager/target/*.jar deploy/firework/lib
cp -r broker/target/*.jar deploy/osx/lib
cp -r broker/target/lib/* deploy/osx/lib
#cp -r dashboard/target/*.jar deploy/firework/lib
#cp -r cli/target/*.jar deploy/firework/lib
#cp -r cli/target/lib/* deploy/firework/lib

#cp  cluster-manager/src/main/resources/* deploy/firework/conf/cluster-manager
cp  broker/src/main/resources/*  deploy/osx/conf/broker
#cp  dashboard/src/main/resources/application.properties   deploy/firework/conf/dashboard
cp  bin/service.sh deploy/osx/
cp  bin/common.sh  deploy/osx/bin
cd  deploy
tar -czf osx.tar.gz  osx
cd $pwd
