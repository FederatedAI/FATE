testsuite
==============

A useful script to running FATE's testsuites.

quick start
-----------

1. (optional) create virtual env

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate


2. install fate_testsuite

   .. code-block:: bash

      pip install fate_testsuite


3. new and edit the testsuite_config.yaml

   .. code-block:: bash

      # create a testsuite_config.yaml in current dir
      testsuite config new
      # edit priority config file with system default editor
      # filling some field according to comments
      testsuite config edit


4. run some testsuites

   .. code-block:: bash

      testsuite suite -i <path contains *testsuite.json>

5. useful logs or exception will be saved to logs dir with namespace showed in last step

testsuite_config.yaml examples
------------------------------

1. deploy multiple flow services in single node: `ip:port`, and run testsuite in different node:

   - party 9999, port 9380
   - party 10000, port 9381

   filling `ssh_tunnel` and comments out `local_services`

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

   filling `local_services` and commnets out `ssh_tunnel`

   .. code-block:: yaml

      local_services: # flow services in local
          - address: 127.0.0.1:9380
            parties: [9999]
          - address: 127.0.0.1:9381
            parties: [10000]


command options
---------------

1. exclude:

   .. code-block:: bash

      testsuite suite -i <path1 contains *testsuite.json> -e <path2 to exclude> -e <path3 to exclude> ...

   will run testsuites in `path1` but not in `path2` and `path3`

2. replace:

   .. code-block:: bash

      testsuite suite -i <path1 contains *testsuite.json> -r '{"maxIter": 5}'

   will find all key-value pair with key "maxIter" in `data conf` or `conf` or `dsl` and replace the value with 5