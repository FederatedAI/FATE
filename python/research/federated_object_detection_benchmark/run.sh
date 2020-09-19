#!/bin/bash
set -x
set -e

DATASET=$1
NUM_CLIENT=$2
MODEL=$3
PORT=$4

if [ ! -n "$DATASET" ];then
	echo "Please input dataset"
	exit
fi

if [ ! -n "$NUM_CLIENT" ];then
	echo "Please input num of client"
	exit
fi

if [ ! -n "$MODEL" ];then
	echo "please input model name"
	exit
fi

if [ ! -n "$PORT" ];then
	echo "please input server port"
	exit
fi

for i in $(seq 1 ${NUM_CLIENT}); do
	nohup python3 fl_client.py \
	     --gpu $((($i % 8)))\
	     --config_file data/task_configs/${MODEL}/${DATASET}/${MODEL}_task$i.json \
	     --ignore_load True \
	     --port ${PORT} &
done
