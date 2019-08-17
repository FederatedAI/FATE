#!/usr/bin/env bash

mode=${1}

nohup python show_result.py feature_select_guest_out_table example_data_namespace 10 predict_data ${mode} > nohup.result 2>$1 &