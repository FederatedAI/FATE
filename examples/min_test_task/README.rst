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
3. -s: whether to load and bind model for fate-serving. Its default value is 1 which means do load and bind. To unset it, set it as 0.
4. -b: indicate which backend you would like to use. 0 represent for eggroll and 1 represent for spark. The default value is 0.