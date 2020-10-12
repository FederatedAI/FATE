Computing API
---------

Most of the time, the federatedml's user does not need to know how to initialize a computing session because
fate flow has already cover this for you. Unless, the user is writing unittest, and CTable related functions are involved.
Initialize a computing session:

.. code-block:: python

   from fate_arch.session import computing_session
   # initialize
   computing_session.init(work_mode=0, backend=0, session_id="a great session")
   # create a table from iterable data
   table = computing_session.parallelize(range(100), include_key=False, partition=2)

.. autoclass:: fate_arch.session.computing_session
   :members:

After creating a table using computing session, many distributed computing api available

.. automodule:: fate_arch.abc._computing
   :autosummary:
   :members:
   :member-order: bysource
