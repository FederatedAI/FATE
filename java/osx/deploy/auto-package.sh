
pwd=`pwd`
cwd=$(cd `dirname $0`; pwd)
cd $cwd
rm -fr osx
if [[ ! -d "osx" ]]; then
    mkdir osx
fi
mkdir  osx/lib
mkdir -p osx/pb_lib/v2
mkdir -p osx/pb_lib/v3
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
cp osx-pb-v2/target/*.jar deploy/osx/pb_lib/v2
cp osx-pb-v3/target/*.jar deploy/osx/pb_lib/v3
cp  osx-broker/src/main/resources/broker/*  deploy/osx/conf/broker
cp -r osx-broker/src/main/resources/components/* deploy/osx/conf/components
cp  bin/service.sh deploy/osx/
rm -f deploy/osx/lib/osx-pb-v*.jar
cd  deploy
sed -i 's/\r//g' osx/service.sh
tar -czf osx.tar.gz  osx
cd $pwd
