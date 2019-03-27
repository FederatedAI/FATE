#!/usr/bin/env bash

jobid=feature_selection_$(date +%Y%m%d%H%M%S)
cur_dir=$(pwd)

nohup python ${cur_dir}/run_binning.py 0 ${jobid} 9999 10000 result_${jobid} result_${jobid} &
