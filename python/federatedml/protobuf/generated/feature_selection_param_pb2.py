# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: feature-selection-param.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1d\x66\x65\x61ture-selection-param.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"\xa5\x01\n\x0c\x46\x65\x61tureValue\x12_\n\x0e\x66\x65\x61ture_values\x18\x01 \x03(\x0b\x32G.com.webank.ai.fate.core.mlmodel.buffer.FeatureValue.FeatureValuesEntry\x1a\x34\n\x12\x46\x65\x61tureValuesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\"\xa5\x01\n\x08LeftCols\x12\x15\n\roriginal_cols\x18\x01 \x03(\t\x12Q\n\tleft_cols\x18\x02 \x03(\x0b\x32>.com.webank.ai.fate.core.mlmodel.buffer.LeftCols.LeftColsEntry\x1a/\n\rLeftColsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x08:\x02\x38\x01\"\xba\x03\n\x1b\x46\x65\x61tureSelectionFilterParam\x12n\n\x0e\x66\x65\x61ture_values\x18\x01 \x03(\x0b\x32V.com.webank.ai.fate.core.mlmodel.buffer.FeatureSelectionFilterParam.FeatureValuesEntry\x12Q\n\x13host_feature_values\x18\x02 \x03(\x0b\x32\x34.com.webank.ai.fate.core.mlmodel.buffer.FeatureValue\x12\x43\n\tleft_cols\x18\x03 \x01(\x0b\x32\x30.com.webank.ai.fate.core.mlmodel.buffer.LeftCols\x12H\n\x0ehost_left_cols\x18\x04 \x03(\x0b\x32\x30.com.webank.ai.fate.core.mlmodel.buffer.LeftCols\x12\x13\n\x0b\x66ilter_name\x18\x05 \x01(\t\x1a\x34\n\x12\x46\x65\x61tureValuesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\"\xde\x03\n\x15\x46\x65\x61tureSelectionParam\x12T\n\x07results\x18\x01 \x03(\x0b\x32\x43.com.webank.ai.fate.core.mlmodel.buffer.FeatureSelectionFilterParam\x12I\n\x0f\x66inal_left_cols\x18\x02 \x01(\x0b\x32\x30.com.webank.ai.fate.core.mlmodel.buffer.LeftCols\x12\x11\n\tcol_names\x18\x03 \x03(\t\x12L\n\x0ehost_col_names\x18\x04 \x03(\x0b\x32\x34.com.webank.ai.fate.core.mlmodel.buffer.HostColNames\x12\x0e\n\x06header\x18\x05 \x03(\t\x12w\n\x17\x63ol_name_to_anonym_dict\x18\x06 \x03(\x0b\x32V.com.webank.ai.fate.core.mlmodel.buffer.FeatureSelectionParam.ColNameToAnonymDictEntry\x1a:\n\x18\x43olNameToAnonymDictEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"3\n\x0cHostColNames\x12\x11\n\tcol_names\x18\x01 \x03(\t\x12\x10\n\x08party_id\x18\x02 \x01(\tB\x1c\x42\x1a\x46\x65\x61tureSelectionParamProtob\x06proto3')



_FEATUREVALUE = DESCRIPTOR.message_types_by_name['FeatureValue']
_FEATUREVALUE_FEATUREVALUESENTRY = _FEATUREVALUE.nested_types_by_name['FeatureValuesEntry']
_LEFTCOLS = DESCRIPTOR.message_types_by_name['LeftCols']
_LEFTCOLS_LEFTCOLSENTRY = _LEFTCOLS.nested_types_by_name['LeftColsEntry']
_FEATURESELECTIONFILTERPARAM = DESCRIPTOR.message_types_by_name['FeatureSelectionFilterParam']
_FEATURESELECTIONFILTERPARAM_FEATUREVALUESENTRY = _FEATURESELECTIONFILTERPARAM.nested_types_by_name['FeatureValuesEntry']
_FEATURESELECTIONPARAM = DESCRIPTOR.message_types_by_name['FeatureSelectionParam']
_FEATURESELECTIONPARAM_COLNAMETOANONYMDICTENTRY = _FEATURESELECTIONPARAM.nested_types_by_name['ColNameToAnonymDictEntry']
_HOSTCOLNAMES = DESCRIPTOR.message_types_by_name['HostColNames']
FeatureValue = _reflection.GeneratedProtocolMessageType('FeatureValue', (_message.Message,), {

  'FeatureValuesEntry' : _reflection.GeneratedProtocolMessageType('FeatureValuesEntry', (_message.Message,), {
    'DESCRIPTOR' : _FEATUREVALUE_FEATUREVALUESENTRY,
    '__module__' : 'feature_selection_param_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureValue.FeatureValuesEntry)
    })
  ,
  'DESCRIPTOR' : _FEATUREVALUE,
  '__module__' : 'feature_selection_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureValue)
  })
_sym_db.RegisterMessage(FeatureValue)
_sym_db.RegisterMessage(FeatureValue.FeatureValuesEntry)

LeftCols = _reflection.GeneratedProtocolMessageType('LeftCols', (_message.Message,), {

  'LeftColsEntry' : _reflection.GeneratedProtocolMessageType('LeftColsEntry', (_message.Message,), {
    'DESCRIPTOR' : _LEFTCOLS_LEFTCOLSENTRY,
    '__module__' : 'feature_selection_param_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.LeftCols.LeftColsEntry)
    })
  ,
  'DESCRIPTOR' : _LEFTCOLS,
  '__module__' : 'feature_selection_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.LeftCols)
  })
_sym_db.RegisterMessage(LeftCols)
_sym_db.RegisterMessage(LeftCols.LeftColsEntry)

FeatureSelectionFilterParam = _reflection.GeneratedProtocolMessageType('FeatureSelectionFilterParam', (_message.Message,), {

  'FeatureValuesEntry' : _reflection.GeneratedProtocolMessageType('FeatureValuesEntry', (_message.Message,), {
    'DESCRIPTOR' : _FEATURESELECTIONFILTERPARAM_FEATUREVALUESENTRY,
    '__module__' : 'feature_selection_param_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureSelectionFilterParam.FeatureValuesEntry)
    })
  ,
  'DESCRIPTOR' : _FEATURESELECTIONFILTERPARAM,
  '__module__' : 'feature_selection_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureSelectionFilterParam)
  })
_sym_db.RegisterMessage(FeatureSelectionFilterParam)
_sym_db.RegisterMessage(FeatureSelectionFilterParam.FeatureValuesEntry)

FeatureSelectionParam = _reflection.GeneratedProtocolMessageType('FeatureSelectionParam', (_message.Message,), {

  'ColNameToAnonymDictEntry' : _reflection.GeneratedProtocolMessageType('ColNameToAnonymDictEntry', (_message.Message,), {
    'DESCRIPTOR' : _FEATURESELECTIONPARAM_COLNAMETOANONYMDICTENTRY,
    '__module__' : 'feature_selection_param_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureSelectionParam.ColNameToAnonymDictEntry)
    })
  ,
  'DESCRIPTOR' : _FEATURESELECTIONPARAM,
  '__module__' : 'feature_selection_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureSelectionParam)
  })
_sym_db.RegisterMessage(FeatureSelectionParam)
_sym_db.RegisterMessage(FeatureSelectionParam.ColNameToAnonymDictEntry)

HostColNames = _reflection.GeneratedProtocolMessageType('HostColNames', (_message.Message,), {
  'DESCRIPTOR' : _HOSTCOLNAMES,
  '__module__' : 'feature_selection_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.HostColNames)
  })
_sym_db.RegisterMessage(HostColNames)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\032FeatureSelectionParamProto'
  _FEATUREVALUE_FEATUREVALUESENTRY._options = None
  _FEATUREVALUE_FEATUREVALUESENTRY._serialized_options = b'8\001'
  _LEFTCOLS_LEFTCOLSENTRY._options = None
  _LEFTCOLS_LEFTCOLSENTRY._serialized_options = b'8\001'
  _FEATURESELECTIONFILTERPARAM_FEATUREVALUESENTRY._options = None
  _FEATURESELECTIONFILTERPARAM_FEATUREVALUESENTRY._serialized_options = b'8\001'
  _FEATURESELECTIONPARAM_COLNAMETOANONYMDICTENTRY._options = None
  _FEATURESELECTIONPARAM_COLNAMETOANONYMDICTENTRY._serialized_options = b'8\001'
  _FEATUREVALUE._serialized_start=74
  _FEATUREVALUE._serialized_end=239
  _FEATUREVALUE_FEATUREVALUESENTRY._serialized_start=187
  _FEATUREVALUE_FEATUREVALUESENTRY._serialized_end=239
  _LEFTCOLS._serialized_start=242
  _LEFTCOLS._serialized_end=407
  _LEFTCOLS_LEFTCOLSENTRY._serialized_start=360
  _LEFTCOLS_LEFTCOLSENTRY._serialized_end=407
  _FEATURESELECTIONFILTERPARAM._serialized_start=410
  _FEATURESELECTIONFILTERPARAM._serialized_end=852
  _FEATURESELECTIONFILTERPARAM_FEATUREVALUESENTRY._serialized_start=187
  _FEATURESELECTIONFILTERPARAM_FEATUREVALUESENTRY._serialized_end=239
  _FEATURESELECTIONPARAM._serialized_start=855
  _FEATURESELECTIONPARAM._serialized_end=1333
  _FEATURESELECTIONPARAM_COLNAMETOANONYMDICTENTRY._serialized_start=1275
  _FEATURESELECTIONPARAM_COLNAMETOANONYMDICTENTRY._serialized_end=1333
  _HOSTCOLNAMES._serialized_start=1335
  _HOSTCOLNAMES._serialized_end=1386
# @@protoc_insertion_point(module_scope)
