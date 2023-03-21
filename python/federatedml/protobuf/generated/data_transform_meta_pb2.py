# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: data-transform-meta.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19\x64\x61ta-transform-meta.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"W\n\x18\x44\x61taTransformImputerMeta\x12\x12\n\nis_imputer\x18\x01 \x01(\x08\x12\x10\n\x08strategy\x18\x02 \x01(\t\x12\x15\n\rmissing_value\x18\x03 \x03(\t\"W\n\x18\x44\x61taTransformOutlierMeta\x12\x12\n\nis_outlier\x18\x01 \x01(\x08\x12\x10\n\x08strategy\x18\x02 \x01(\t\x12\x15\n\routlier_value\x18\x03 \x03(\t\"\xd9\x04\n\x11\x44\x61taTransformMeta\x12\x14\n\x0cinput_format\x18\x01 \x01(\t\x12\x11\n\tdelimitor\x18\x02 \x01(\t\x12\x11\n\tdata_type\x18\x03 \x01(\t\x12\x16\n\x0etag_with_value\x18\x04 \x01(\x08\x12\x1b\n\x13tag_value_delimitor\x18\x05 \x01(\t\x12\x12\n\nwith_label\x18\x06 \x01(\x08\x12\x12\n\nlabel_name\x18\x07 \x01(\t\x12\x12\n\nlabel_type\x18\x08 \x01(\t\x12\x15\n\routput_format\x18\t \x01(\t\x12V\n\x0cimputer_meta\x18\n \x01(\x0b\x32@.com.webank.ai.fate.core.mlmodel.buffer.DataTransformImputerMeta\x12V\n\x0coutlier_meta\x18\x0b \x01(\x0b\x32@.com.webank.ai.fate.core.mlmodel.buffer.DataTransformOutlierMeta\x12\x10\n\x08need_run\x18\x0c \x01(\x08\x12m\n\x13\x65xclusive_data_type\x18\r \x03(\x0b\x32P.com.webank.ai.fate.core.mlmodel.buffer.DataTransformMeta.ExclusiveDataTypeEntry\x12\x15\n\rwith_match_id\x18\x0e \x01(\x08\x1a\x38\n\x16\x45xclusiveDataTypeEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x42\x11\x42\x0f\x44\x61taIOMetaProtob\x06proto3')



_DATATRANSFORMIMPUTERMETA = DESCRIPTOR.message_types_by_name['DataTransformImputerMeta']
_DATATRANSFORMOUTLIERMETA = DESCRIPTOR.message_types_by_name['DataTransformOutlierMeta']
_DATATRANSFORMMETA = DESCRIPTOR.message_types_by_name['DataTransformMeta']
_DATATRANSFORMMETA_EXCLUSIVEDATATYPEENTRY = _DATATRANSFORMMETA.nested_types_by_name['ExclusiveDataTypeEntry']
DataTransformImputerMeta = _reflection.GeneratedProtocolMessageType('DataTransformImputerMeta', (_message.Message,), {
  'DESCRIPTOR' : _DATATRANSFORMIMPUTERMETA,
  '__module__' : 'data_transform_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.DataTransformImputerMeta)
  })
_sym_db.RegisterMessage(DataTransformImputerMeta)

DataTransformOutlierMeta = _reflection.GeneratedProtocolMessageType('DataTransformOutlierMeta', (_message.Message,), {
  'DESCRIPTOR' : _DATATRANSFORMOUTLIERMETA,
  '__module__' : 'data_transform_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.DataTransformOutlierMeta)
  })
_sym_db.RegisterMessage(DataTransformOutlierMeta)

DataTransformMeta = _reflection.GeneratedProtocolMessageType('DataTransformMeta', (_message.Message,), {

  'ExclusiveDataTypeEntry' : _reflection.GeneratedProtocolMessageType('ExclusiveDataTypeEntry', (_message.Message,), {
    'DESCRIPTOR' : _DATATRANSFORMMETA_EXCLUSIVEDATATYPEENTRY,
    '__module__' : 'data_transform_meta_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.DataTransformMeta.ExclusiveDataTypeEntry)
    })
  ,
  'DESCRIPTOR' : _DATATRANSFORMMETA,
  '__module__' : 'data_transform_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.DataTransformMeta)
  })
_sym_db.RegisterMessage(DataTransformMeta)
_sym_db.RegisterMessage(DataTransformMeta.ExclusiveDataTypeEntry)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\017DataIOMetaProto'
  _DATATRANSFORMMETA_EXCLUSIVEDATATYPEENTRY._options = None
  _DATATRANSFORMMETA_EXCLUSIVEDATATYPEENTRY._serialized_options = b'8\001'
  _DATATRANSFORMIMPUTERMETA._serialized_start=69
  _DATATRANSFORMIMPUTERMETA._serialized_end=156
  _DATATRANSFORMOUTLIERMETA._serialized_start=158
  _DATATRANSFORMOUTLIERMETA._serialized_end=245
  _DATATRANSFORMMETA._serialized_start=248
  _DATATRANSFORMMETA._serialized_end=849
  _DATATRANSFORMMETA_EXCLUSIVEDATATYPEENTRY._serialized_start=793
  _DATATRANSFORMMETA_EXCLUSIVEDATATYPEENTRY._serialized_end=849
# @@protoc_insertion_point(module_scope)
