### Union

Union module combines given tables into one while keeping unique entry ids. Union is a local module. Like DataIO, this module can be run on the side of Host or Guest, and it does not require any interaction with outside parties.

### Use

Union currently only supports joining by entry id. The tables being joined must have exactly the same table schema. In other words, their header, idx and label column name (if label exists) should match.

When an id appears more than once in the joining tables, only the entry from its first appearance will be kept in the final combined table.

For example job conf and dsl setting for Union module, please refer [here](../../../examples/federatedml-1.x-examples/union).