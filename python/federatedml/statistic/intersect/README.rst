Intersection
============

This module provide some method of PSI(Private Set Intersection)

RSA Intersection
----------------

This mode implements algorithm based on [`RSA Intersection`__]. This work is built on FATE, eggroll and federation API that construct the secure, distributed and parallel infrastructure.

.. __: https://books.google.com.hk/books?id=zfvf37_YS8cC&pg=PA73&lpg=PA73&dq=rsa+commutative+encryption&source=bl&ots=LbOiyIlr3E&sig=IIWlTGeoU0C8dRiN10uH2OAwobQ&hl=zh-CN&sa=X&ved=0ahUKEwiLoozC1tbXAhVDnJQKHbP7DvAQ6AEIdTAJ#v=onepage&q&f=false.

Our Intersection module tries to solve Privacy-Preserving Entity Match problem.
This module helps two and more parties to find common entry ids without leaking non-overlapping ids.
The process is illustrated below in `figure 1`.

.. figure:: images/rsa_intersection.png
   :width: 500
   :align: center
   :name: figure 1

   Figure 1 (RSA Intersection between party A and party B)

In `figure 1` ,Party A has user id u1,u2,u3,u4, while Party B has u1,u2,u3,u5. After Intersection,
party A and party B both learn their common user ids, which are u1,u2,u3, while neither party A nor B could decrypt
each other's non-overlapping user ids.
Transmission parties' processed id to the other party, like :math:`Y-A` and :math:`Z-B`,
will not reveal raw ids. Processed :math:`Z-B` is safe due to the privacy key of party B.
Each :math:`Y-A` includes different random value which binds to each value in :math:`X-A` and will be safe as well.

Introduced in FATE version 1.6, `split_calculation` option is available for improved efficiency.
Different from unified process described above, `split_calculation` process first splits hash-processed ids
into even and odd groups; each group then runs through the RSA intersection process with either host or guest as the joining role.
Note than with `split_calculation`, host(s) always know about their common even ids with guest since they are responsible for finding common even ids.

With RSA intersection, participants can get their intersection ids securely and efficiently.

RAW Intersection
----------------

This mode implements the simple intersection method in which a participant sends all its ids to another participant, and the other
participant finds their common ids. Finally, the joining role will send the intersection ids to the sender.

DH Intersection
---------------

This mode implements secure intersection based on symmetric encryption with Pohlig–Hellman commutative cipher.
DH Intersection is also used in `Secure Information Retrieval(SIR) module <../../secure_information_retrieval>`_.

Multi-Host Intersection
-----------------------

RSA, RAW, and DH intersection support multi-host scenario. It means a guest can perform intersection with more than one host simultaneously and get the common ids among all participants.

.. figure:: ./images/multi_hosts.png
   :align: center
   :width: 500
   :name: figure 2

   Figure 2 (multi-hosts Intersection)

Refer to `figure 2` for a demonstration of one guest running intersection with two hosts; the same process applies to cases with more than two hosts.
First, guest will run intersection with each host and get respective overlapping ids.
Then, guest will find common IDs from all intersection results. Optionally,
guest will send common IDs to every host.

Match ID(Repeated ID) intersection
----------------------------------

Starting at ver 1.7, it is recommended to assign random sid to uploaded data.
Intersection module then automatically checks for and process data with instance ID.

Note that parameters for original repeated ID process such as `repeated_id_process` are deprecated in ver 1.7.
Specify `join_id_owner` to the role whose sid to be kept.
For instances, when `join_id_owner` is set to Guest(default), Guest's data is

::

   sid, id, value
   123, alice, 2
   125, alice, 3
   130, bob, 4

In Host, you data is

::

   sid, id, value
   210, alice, 5
   232, alice, 5
   212, bob, 4

After intersection, guest will get the intersection results:

::

   sid, id, value
   123, alice, 2
   125, alice, 3
   130, bob, 4

And for Host:

::

   sid, id, value
   123, alice, 5
   125, alice, 5
   130, bob, 4

Param
------

.. automodule:: federatedml.param.intersect_param
   :members:

Feature
-------

RSA, RAW, and DH intersection methods support:

1. Multi-host PSI task. The detailed configuration for multi-host task can be found `here. <../../../../doc/dsl_conf_v2_setting_guide.rst#multi-host-configuration>`_

2. Match ID intersection using ID expanding.

3. Configurable hashing methods, including sha256, md5, and sm3; hash operators of RSA intersection can be configured separately, please refer `here <../../param/intersect_param.py>`_ for more details.

4. Preprocessing step to pre-filter Host's data for faster PSI

RAW and DH intersection methods also support:

1. PSI with cache

2. intersection cardinality estimation

RAW intersection supports the following extra feature:

1. base64 encoding may be used for all hashing methods.
