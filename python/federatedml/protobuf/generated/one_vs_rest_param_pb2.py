# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: one-vs-rest-param.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17one-vs-rest-param.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"u\n\x0eOneVsRestParam\x12\x0f\n\x07\x63lasses\x18\x01 \x03(\t\x12R\n\x11\x63lassifier_models\x18\x02 \x03(\x0b\x32\x37.com.webank.ai.fate.core.mlmodel.buffer.ClassifierModel\"2\n\x0f\x43lassifierModel\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x11\n\tnamespace\x18\x04 \x01(\tB\x15\x42\x13OneVsRestParamProtob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'one_vs_rest_param_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\023OneVsRestParamProto'
  _globals['_ONEVSRESTPARAM']._serialized_start=67
  _globals['_ONEVSRESTPARAM']._serialized_end=184
  _globals['_CLASSIFIERMODEL']._serialized_start=186
  _globals['_CLASSIFIERMODEL']._serialized_end=236
# @@protoc_insertion_point(module_scope)
