#!/bin/bash
base="$(cd `dirname $0`; pwd)"
cdir="/data/projects/common/miniconda3"
PFX="/data/projects/fate"
sh ./packages/Miniconda3-*-Linux-x86_64.sh -b -p $cdir
$cdir/bin/pip install --upgrade $base/pips/pip-18.1-py2.py3-none-any.whl
$cdir/bin/pip install $base/pips/virtualenv-16.1.0-py2.py3-none-any.whl
$cdir/bin/virtualenv -p $cdir/bin/python3.6  --no-wheel --no-setuptools --no-download $PFX/venv
source $PFX/venv/bin/activate
pip install --upgrade $base/pips/pip-18.1-py2.py3-none-any.whl
pip install $base/pips/setuptools-40.6.3-py2.py3-none-any.whl
pip install $base/pips/wheel-0.32.3-py2.py3-none-any.whl
pip install -r ./pip-dependencies/requirements.txt -f ./pip-dependencies --no-index
pip list | wc -l
echo "done"
