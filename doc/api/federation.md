# Federation API

## Low level api

::: fate_arch.abc._federation
    heading_level: 3
    show_source: true
    show_root_heading: true
    show_root_toc_entry: false
    show_root_full_path: false

## user api

remoting or getting an object(table) from other parties is quite easy using apis provided in ``Variable``.
First to create an instance of BaseTransferVariable, which is simply a collection of Variables:

```python
from federatedml.transfer_variable.transfer_class import secure_add_example_transfer_variable
variable = secure_add_example_transfer_variable.SecureAddExampleTransferVariable()
```

Then remote or get object(table) by variable provided by this instance:

```python

# remote
variable.guest_share.remote("from guest")

# get
variable.guest_share.get()
```

<!-- mkdocs
::: fate_arch.federation.transfer_variable
    heading_level: 3
    show_source: true
    show_root_heading: true
    show_root_toc_entry: false
    show_root_full_path: false
-->
