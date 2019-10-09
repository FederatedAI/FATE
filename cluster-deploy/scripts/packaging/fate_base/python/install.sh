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
cdir=$pythondir
if [ ! -d "${cdir}" ];then
    echo "Nothing to do!"
else
   rm -rf $cdir
fi
PFX=$dir
cd $base
echo $base
cd ..
basedir=`pwd`
echo $basedir
#if [ ! -d "${cdir}" ];then
#    mkdir -p $cdir
#fi
#echo $cdir
#cd $cdir
#source build
source_build() {
        return 0
}
#build
build(){
if [ ! -f "$basedir/packages/Miniconda$conda_version-Linux-x86_64.sh" ];then
    echo "No Miniconda$conda_version-Linux-x86_64.sh package can't be installed"
    exit
fi
sh  $basedir/packages/Miniconda$conda_version-Linux-x86_64.sh -b -p $cdir
$cdir/bin/pip install --upgrade $basedir/pips/pip-$pip_version-py2.py3-none-any.whl
$cdir/bin/pip install $basedir/pips/virtualenv-$virtualenv_version-py2.py3-none-any.whl
$cdir/bin/virtualenv -p $cdir/bin/python3.6  --no-wheel --no-setuptools --no-download $PFX/venv
source $PFX/venv/bin/activate
pip install --upgrade $basedir/pips/pip-$pip_version-py2.py3-none-any.whl
pip install $basedir/pips/setuptools-$setuptools_version-py2.py3-none-any.whl
pip install $basedir/pips/wheel-$wheel_version-py2.py3-none-any.whl
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r  $basedir/pip-dependencies/requirements.txt -f  $basedir/pip-dependencies --no-index
#pip install -r ./pip-dependencies/requirements.txt -f ./pip-dependencies --no-index
pip list | wc -l
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