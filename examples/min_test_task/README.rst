Description
===========

This is the minimum test task for installation. Using this you can test
if the installation of FATE is successful or not.

Before You Start
----------------

Please make sure you have already deploy FATE correctly and already upload data in both sides. You can upload data easily by a `provided script <../scripts/README.rst>`_

Usage
^^^^^

All you need to do is just run the following command in guest party:

    .. code-block:: bash

        python run_task.py -m {work_mode} -gid {guest_id} -hid {host_id}, -aid {arbiter_id}

This test will automatically take breast as test data set.

There are some more parameters that you may need:

1. -f: file type. "fast" means breast data set, "normal" means default credit data set.
2. --add_sbt: if set, it will test hetero-secureboost task after testing hetero-lr.