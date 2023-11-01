#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.nn as nn
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.versions_pb2 import VersionDef
from torch.utils.tensorboard import SummaryWriter as _SummaryWriter


def graph(model):
    """Converts a crypten.nn graph for consumption by TensorBoard."""

    # convert individual module to graph:
    assert isinstance(model, nn.Module), "model must be crypten.nn.Module"
    if not isinstance(model, nn.Graph):
        graph = nn.Graph("input", "output")
        graph.add_module("output", model, ["input"])
        model = graph

    # create mapping to more interpretable node naming:
    mapping = {input_name: input_name for input_name in model.input_names}
    modules = {name: module for name, module in model.named_modules()}
    for name, module in modules.items():
        op = str(type(module))[26:-2]
        mapping[name] = "%s_%s" % (op, name)

    # create input variables:
    nodes = [
        NodeDef(
            name=mapping[input_name].encode(encoding="utf_8"),
            op="Variable",
            input=[],
        )
        for input_name in model.input_names
    ]

    # loop all graph connections:
    for output_name, input_names in model._graph.items():

        # get parameters and type of module:
        module = modules[output_name]
        op = str(type(module))
        input_names = [mapping[name] for name in input_names]
        parameters = [
            "%s: %s" % (name, parameter.size())
            for name, parameter in module.named_parameters()
        ]
        parameter_string = "; ".join(parameters).encode(encoding="utf_8")

        # add to graph:
        nodes.append(
            NodeDef(
                name=mapping[output_name].encode(encoding="utf_8"),
                op=op,
                input=input_names,
                attr={"attr": AttrValue(s=parameter_string)},
            )
        )

    # return graph definition:
    return GraphDef(node=nodes, versions=VersionDef(producer=22))


class SummaryWriter(_SummaryWriter):
    """
    Adapts the PyTorch SummaryWriter to output crypten graphs.
    """

    def add_graph(self, model, input_to_model=None, verbose=False):
        self._get_file_writer().add_onnx_graph(graph(model))
