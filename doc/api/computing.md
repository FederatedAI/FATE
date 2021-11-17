# Computing API

Most of the time, the federatedml's user does not need to know how to initialize a computing session because
fate flow has already cover this for you. Unless, the user is writing unittest, and CTable related functions are involved.
Initialize a computing session:

```python
from fate_arch.session import computing_session
# initialize
computing_session.init(session_id="a great session")
# create a table from iterable data
table = computing_session.parallelize(range(100), include_key=False, partition=2)
```

<!-- mkdocs
## computing session

::: fate_arch.session.computing_session
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false


## computing table

After creating a table using computing session, many distributed computing api available

::: fate_arch.abc._computing.CTableABC
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->
