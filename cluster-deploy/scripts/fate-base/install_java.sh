#!/bin/bash
base="$(cd `dirname $0`; pwd)"
cdir="/data/projects/common/jdk"
cd $cdir
tar -xzvf $base/packages/jdk-8u*-linux-x64.tar.gz
mv jdk1.8* jdk1.8
echo "done"
