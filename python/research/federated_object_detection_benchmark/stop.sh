#!/bin/bash
DATASET=$1
MODEL=$2
if [ ! -n "$DATASET" ];then
	echo "Please input dataset"
	exit
fi
if [ ! -n "$MODEL" ];then
	echo "Please input model"
	exit
fi
ps -ef | grep ${DATASET}/${MODEL} | grep -v grep | awk '{print $2}' | xargs kill -9
