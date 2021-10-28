Union
=====

Union module combines given tables into one while keeping unique entry
ids. Union is a local module. Like DataIO, this module can be run on the
side of Host or Guest, and running this module does not require any
interaction with outside parties.

Use
---

Union currently only supports joining by entry id. For tables of data
instances, their header, idx and label column name (if label exists)
should match.

When an id appears more than once in the joining tables, user can
specify whether to keep the duplicated instances by setting parameter
[keep\_duplicate]{.title-ref} to True. Otherwise, only the entry from
its first appearance will be kept in the final combined table. Note that
the order by which tables being fed into Union module depends on the job
setting. As shown below:

with FATE-Pipeline:

``` {.sourceCode .python}
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
```

with DSL v2:

``` {.sourceCode .json}
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
```

Upstream tables will enter Union module in this order:
[dataio\_0.data]{.title-ref}, [dataio\_1.data]{.title-ref},
[dataio\_2.data]{.title-ref} .

If an id [42]{.title-ref} exists in both [dataio\_0.data]{.title-ref}
and [dataio\_1.data]{.title-ref}, and:

1.  \'keep\_duplicate\` set to false: the value from
    [dataio\_0.data]{.title-ref} is the one being kept in the final
    result, its id unchanged.
2.  \'keep\_duplicate\` set to true: the value from
    [dataio\_0.data]{.title-ref} and the one from
    [dataio\_1.data]{.title-ref} are both kept; the id in
    [dataio\_0.data]{.title-ref} will be transformed to
    [42\_dataio\_0]{.title-ref}, and the id in
    [dataio\_1.data]{.title-ref} to [42\_dataio\_1]{.title-ref}.

Here are more [\[Union job
examples\]](../../../../examples/pipeline/union) with FATE-Pipeline.

For more example job configuration and dsl setting files, please refer
[\[here\]](../../../../examples/dsl/v2/union).

Param
-----

::: {.automodule}
federatedml.param.union\_param
:::
