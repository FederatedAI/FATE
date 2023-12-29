# Union

Union module combines given tables into one while keeping unique entry
ids. Union is a local module. This module can be run on the
side of Host or Guest, and running this module does not require any
interaction with outside parties.

## Use

Union currently supports concatenation along axis 0.

For tables to be concatenated, their header, including sample id, match id, and label column (if label exists),
should match. Example of such a union task may be found [here](../../../../examples/pipeline/union/test_union.py).
