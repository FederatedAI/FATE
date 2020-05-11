#!/bin/bash
set -e
python ./fml_agent/fml_agent.py >  /dev/null 2>&1 &
python ./fate_flow/fate_flow_server.py
