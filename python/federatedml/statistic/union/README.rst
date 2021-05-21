Union
======

Union module combines given tables into one while keeping unique entry ids. Union is a local module. Like DataIO, this module can be run on the side of Host or Guest, and running this module does not require any interaction with outside parties.


Use
------

Union currently only supports joining by entry id. For tables of data instances, their header, idx and label column name (if label exists) should match.

When an id appears more than once in the joining tables, user can specify whether to keep the duplicated instances by setting parameter `keep_duplicate` to True.
Otherwise, only the entry from its first appearance will be kept in the final combined table. Note that the order by which tables being fed into Union module depends on the job setting. As shown below:

with FATE-Pipeline:

.. code-block:: python

    {
        "union_0": {
                "module": "Union",
                "input": {
                    "data": {
                            "data": ["dataio_0.data", "dataio_1.data", "dataio_2.data"]
                    }
                },
                "output": {
                    "data": ["data"]
                }
            }
        }


with DSL v2:

.. code-block:: json

    {
        "union_0": {
                "module": "Union",
                "input": {
                    "data": {
                            "data": ["dataio_0.data", "dataio_1.data", "dataio_2.data"]
                    }
                },
                "output": {
                    "data": ["data"]
                }
            }
        }


Upstream tables will enter Union module in this order: `dataio_0.data`, `dataio_1.data`, `dataio_2.data` .

If an id `42` exists in both `dataio_0.data` and `dataio_1.data`, and:

1. 'keep_duplicate` set to false: the value from `dataio_0.data` is the one being kept in the final result, its id unchanged.
2. 'keep_duplicate` set to true: the value from `dataio_0.data` and the one from `dataio_1.data` are both kept; the id in `dataio_0.data` will be transformed to `42_dataio_0`, and the id in `dataio_1.data` to `42_dataio_1`.

Here are more `[Union job examples] <../../../../examples/pipeline/union>`_ with FATE-Pipeline.

For more example job configuration and dsl setting files, please refer `[here] <../../../../examples/dsl/v2/union>`_.

Param
------

.. automodule:: federatedml.param.union_param
   :members: