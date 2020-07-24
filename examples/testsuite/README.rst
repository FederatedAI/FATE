quick start
============

1. install dependency with pip

   .. code-block:: bash

      pip install -r requirements.txt

2. copy or modify config.yaml

   .. code-block:: bash

      cp config.yaml my_config.yaml

   filling some field according to comments

3. run testsuite.py

   .. code-block:: bash

      python testsuite federatedml-1.x-examples -config my_config.yaml -name my_testsuite

4. check my_testsuite.info.log or my_testsuite.info.debug


special config
==============

1. deploy multiple flow services in single node: `ip:port`, and run testsuite in different node:
   - party 9999, port 9380
   - party 10000, port 9381
   .. code-block:: yaml

      ssh_tunnel:
          - address: ip:port
          - services:
              - address: 127.0.0.1:9380
                parties: [9999]
              - address: 127.0.0.1:9381
                parties: [10000]


2. deploy multiple flow services in single node: `ip:port`, and run testsuite in same node:
   - party 9999, port 9380
   - party 10000, port 9381
   .. code-block:: yaml

      local_services: # flow services in local
          - address: 127.0.0.1:9380
            parties: [9999]
          - address: 127.0.0.1:9381
            parties: [10000]


features
========

1. exclude:
   .. code-block:: bash

      python testsuite.py `path` -exclude `path1` `path2`

   will run testsuites in `path` but not in `path1` and `path2`

2. replace:
   .. code-block:: base

      python testsuite.py `path` -replace '{"maxIter": 5}'

   will find all key-value pair with key "maxIter" in submit conf and replace the value with 5