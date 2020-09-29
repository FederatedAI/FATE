FATE Test
=========

A collection of useful tools to running FATE's test.

quick start
-----------

1. (optional) create virtual env

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate


2. install fate_test

   .. code-block:: bash

      pip install fate_test
      fate_test --help


3. new and edit the fate_test_config.yaml

   .. code-block:: bash

      # create a fate_test_config.yaml in current dir
      fate_test config new
      # edit priority config file with system default editor
      # filling some field according to comments
      fate_test config edit


4. run some fate_test

   .. code-block:: bash

      fate_test suite -i <path contains *testsuite.json>

5. useful logs or exception will be saved to logs dir with namespace showed in last step


fate_test_config.yaml examples
------------------------------


1. no need ssh tunnel:

   - 9999, service: service_a
   - 10000, service: service_b

   and both service_a, service_b can be requested directly:

   .. code-block:: yaml

      work_mode: 1 # 0 for standalone, 1 for cluster
      data_base_dir: <path_to_data>
      parties:
        guest: [10000]
        host: [9999, 10000]
        arbiter: [9999]
      services:
        - flow_services:
          - {address: service_a, parties: [9999]}
          - {address: service_b, parties: [10000]}

2. need ssh tunnel:

   - 9999, service: service_a
   - 10000, service: service_b

   service_a, can be requested directly while service_b don't,
   but you can request service_b in other node, say B:

   .. code-block:: yaml

      work_mode: 0 # 0 for standalone, 1 for cluster
      data_base_dir: <path_to_data>
      parties:
        guest: [10000]
        host: [9999, 10000]
        arbiter: [9999]
      services:
        - flow_services:
          - {address: service_a, parties: [9999]}
        - flow_services:
          - {address: service_b, parties: [10000]}
          ssh_tunnel: # optional
          enable: true
          ssh_address: <ssh_ip_to_B>:<ssh_port_to_B>
          ssh_username: <ssh_username_to B>
          ssh_password: # optional
          ssh_priv_key: "~/.ssh/id_rsa"


command types
-------------

- suite: used for running testsuites, collection of FATE jobs

  .. code-block:: bash

     fate_test suite -i <path contains *testsuite.json>

- `benchmark-quality <./README_BENCHMARK.rst>`_: used for comparing modeling quality between FATE
and other machine learning systems

  .. code-block:: bash

      fate_test benchmark-quality -i <path contains *testsuite.json>


command options
---------------

1. include:

.. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json>

   will run testsuites in `path1`

2. exclude:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> -e <path2 to exclude> -e <path3 to exclude> ...

   will run testsuites in `path1` but not in `path2` and `path3`

3. replace:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> -r '{"maxIter": 5}'

   will find all key-value pair with key "maxIter" in `data conf` or `conf` or `dsl` and replace the value with 5

4. glob:

   .. code-block:: bash

      fate_test suite -i <path1 contains *testsuite.json> -g "hetero*"

   will run testsuites in sub directory start with `hetero` of `path1`

