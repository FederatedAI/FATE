#!/bin/bash
set -e
cd /fate/fate_flow
python fate_flow_server.py
exec "$@"
