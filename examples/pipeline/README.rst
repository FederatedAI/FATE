Pipeline Examples
=================

Introduction
-------------

We provide some example scripts of running
FATE jobs with `FATE-Pipeline <../../python/fate_client/README.rst>`_.

Please refer to the document linked above for details on FATE-Pipeline.

We provide a convenient tool `FATE-Test <../../python/fate_client/README.rst>`_ for running examples.
To run only pipeline jobs(excluding dsl jobs in path), use the following command:

.. code-block:: bash

      fate_test suite -i <path contains *testsuite.json> --skip-dsl-jobs

DSL version of provided Pipeline examples can be found `here <../dsl/v2>`_.
