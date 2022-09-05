# Union

Union module combines given tables into one while keeping unique entry
ids. Union is a local module. Like DataTransform, this module can be run on the
side of Host or Guest, and running this module does not require any
interaction with outside parties.

## Use

Union currently only supports joining by entry id. For tables of data
instances, their header, idx and label column name (if label exists)
should match.

When an id appears more than once in the joining tables, user can
specify whether to keep the duplicated instances by setting parameter
<span class="title-ref">keep\_duplicate</span> to True. Otherwise, only
the entry from its first appearance will be kept in the final combined
table. Note that the order by which tables being fed into Union module
depends on the job setting. As shown below:

with FATE-Pipeline:

``` sourceCode python
{
    "union_0": {
            "module": "Union",
            "input": {
                "data": {
                        "data": ["data_transform_0.data", "data_transform_1.data", "data_transform_2.data"]
                }
            },
            "output": {
                "data": ["data"]
            }
        }
    }
```

with DSL v2:

``` sourceCode json
{
    "union_0": {
            "module": "Union",
            "input": {
                "data": {
                        "data": ["data_transform_0.data", "data_transform_1.data", "data_transform_2.data"]
                }
            },
            "output": {
                "data": ["data"]
            }
        }
    }
```

Upstream tables will enter Union module in this order:
<span class="title-ref">data\_transform\_0.data</span>,
<span class="title-ref">data\_transform\_1.data</span>,
<span class="title-ref">data\_transform\_2.data</span> .

If an id <span class="title-ref">42</span> exists in both
<span class="title-ref">data\_transform\_0.data</span> and
<span class="title-ref">data\_transform\_1.data</span>, and:

1.  'keep\_duplicate\` set to false: the value from
    <span class="title-ref">data\_transform\_0.data</span> is the one being kept
    in the final result, its id unchanged.
2.  'keep\_duplicate\` set to true: the value from
    <span class="title-ref">data\_transform\_0.data</span> and the one from
    <span class="title-ref">data\_transform\_1.data</span> are both kept; the id
    in <span class="title-ref">data\_transform\_0.data</span> will be transformed
    to <span class="title-ref">42\_data\_transform\_0</span>, and the id in
    <span class="title-ref">data\_transform\_1.data</span> to
    <span class="title-ref">42\_data\_transform\_1</span>.


<!-- mkdocs
## Param
::: federatedml.param.union_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
 -->

<!-- mkdocs
## Examples

{% include-examples "union" %}
-->
