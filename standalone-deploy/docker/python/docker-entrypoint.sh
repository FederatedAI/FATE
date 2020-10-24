#!/bin/bash
set -e
cd /fate/python/fate_flow
python fate_flow_server.py
exec "$@"
