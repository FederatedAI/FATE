#!/bin/bash

set -x
set -e

DATASET=$1
MODEL=$2
PORT=$3

if [ ! -n "$DATASET" ];then
	echo "Please input dataset"
	exit
fi

if [ ! -n "$MODEL" ];then
        echo "Please input model name"
        exit
fi

if [ ! -n "$PORT" ];then
        echo "please input server port"
        exit
fi

if [ ! -d "experiments/logs/`date +'%m%d'`/${MODEL}/${DATASET}" ];then
	mkdir -p "experiments/logs/`date +'%m%d'`/${MODEL}/${DATASET}"
fi

LOG="experiments/logs/`date +'%m%d'`/${MODEL}/${DATASET}/fl_server.log"
echo Loggin output to "$LOG"

nohup python3 fl_server.py --config_file data/task_configs/${MODEL}/${DATASET}/${MODEL}_task.json --port ${PORT} > ${LOG} &
