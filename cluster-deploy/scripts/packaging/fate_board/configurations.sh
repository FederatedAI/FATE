#!/bin/bash

#fataboard config
#input_dir=/data/projects/xuyiming/0917/FATE
#output_dir=/data/projects/xuyiming/0917/FATE/cluster-deploy/scripts
version=1.0
java_dir=/data/projects/common/jdk
fbport=8080
flport=9380
flip=192.0.0.1
fldbip=192.0.0.1
fldbname=fate_flow
fldbuser=fate_dev
fldbpasswd=fate_dev
nodelist=('192.0.0.1 app app 22' '192.0.0.2 app app 22')
