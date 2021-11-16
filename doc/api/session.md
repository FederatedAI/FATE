## Session API

`Session` is the context to use `computing`, `storage` and `federation` resouraces. 
At most situation, users should not concern how `Session` is create. 
`FATE-Flow` is responsed to create and init `Session` when `Task` launched.

For those who want to use `computing`, `storage` and `federation` api outside `FATE-Flow Task`,
flowing is a short guide.

1. init Session

    ```python
    sess = Session()
    sess.as_global()

    # flowing is optional, call if needed
    
    # init computing
    sess.init_computing(...)

    # init federation
    sess.init_federation(...)
    
    # init federation
    sess.init_storage(...)
    ```
2. calling specific api

    ```python
    computing = sess.computing
    federation = sess.federation
    storage = sess.storage
    ```

3. computing api has a shortcut

    ```python
    from fate_arch.session import computing_session
    computing_session.init(...)
    computing_session.parallelize(...)
    ```

<!-- mkdocs
## Detailed API

::: fate_arch.session.Session
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->