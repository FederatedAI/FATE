Federation API
--------------

Low level api
~~~~~~~~~~~~~~

.. automodule:: fate_arch.abc._federation
   :autosummary:
   :members:
   :no-members:
   :member-order: bysource

user api
~~~~~~~~~

remoting or getting an object(table) from other parties is quite easy using apis provided in ``Variable``.
First to create an instance of BaseTransferVariable, which is simply a collection of Variables:

.. code-block:: python

   from federatedml.transfer_variable.transfer_class import secure_add_example_transfer_variable
   variable = secure_add_example_transfer_variable.SecureAddExampleTransferVariable()


Then remote or get object(table) by variable provided by this instance:

.. code-block:: python

   # remote
   variable.guest_share.remote("from guest")

   # get
   variable.guest_share.get()

.. automodule:: fate_arch.federation.transfer_variable._transfer_variable
   :autosummary:
   :members:
   :no-members:
   :member-order: bysource

