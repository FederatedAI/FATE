# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lr-model-param.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import sshe_cipher_param_pb2 as sshe__cipher__param__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14lr-model-param.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\x1a\x17sshe-cipher-param.proto\"\x85\x05\n\x0cLRModelParam\x12\r\n\x05iters\x18\x01 \x01(\x05\x12\x14\n\x0closs_history\x18\x02 \x03(\x01\x12\x14\n\x0cis_converged\x18\x03 \x01(\x08\x12P\n\x06weight\x18\x04 \x03(\x0b\x32@.com.webank.ai.fate.core.mlmodel.buffer.LRModelParam.WeightEntry\x12\x11\n\tintercept\x18\x05 \x01(\x01\x12\x0e\n\x06header\x18\x06 \x03(\t\x12S\n\x12one_vs_rest_result\x18\x07 \x01(\x0b\x32\x37.com.webank.ai.fate.core.mlmodel.buffer.OneVsRestResult\x12\x18\n\x10need_one_vs_rest\x18\x08 \x01(\x08\x12\x16\n\x0e\x62\x65st_iteration\x18\t \x01(\x05\x12\x63\n\x10\x65ncrypted_weight\x18\n \x03(\x0b\x32I.com.webank.ai.fate.core.mlmodel.buffer.LRModelParam.EncryptedWeightEntry\x12>\n\x06\x63ipher\x18\x0b \x01(\x0b\x32..com.webank.ai.fate.core.mlmodel.buffer.Cipher\x1a-\n\x0bWeightEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\x1aj\n\x14\x45ncryptedWeightEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x41\n\x05value\x18\x02 \x01(\x0b\x32\x32.com.webank.ai.fate.core.mlmodel.buffer.CipherText:\x02\x38\x01\"\x93\x04\n\x0bSingleModel\x12\r\n\x05iters\x18\x01 \x01(\x05\x12\x14\n\x0closs_history\x18\x02 \x03(\x01\x12\x14\n\x0cis_converged\x18\x03 \x01(\x08\x12O\n\x06weight\x18\x04 \x03(\x0b\x32?.com.webank.ai.fate.core.mlmodel.buffer.SingleModel.WeightEntry\x12\x11\n\tintercept\x18\x05 \x01(\x01\x12\x0e\n\x06header\x18\x06 \x03(\t\x12\x16\n\x0e\x62\x65st_iteration\x18\x07 \x01(\x05\x12\x62\n\x10\x65ncrypted_weight\x18\x08 \x03(\x0b\x32H.com.webank.ai.fate.core.mlmodel.buffer.SingleModel.EncryptedWeightEntry\x12>\n\x06\x63ipher\x18\t \x01(\x0b\x32..com.webank.ai.fate.core.mlmodel.buffer.Cipher\x1a-\n\x0bWeightEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\x1aj\n\x14\x45ncryptedWeightEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x41\n\x05value\x18\x02 \x01(\x0b\x32\x32.com.webank.ai.fate.core.mlmodel.buffer.CipherText:\x02\x38\x01\"}\n\x0fOneVsRestResult\x12M\n\x10\x63ompleted_models\x18\x01 \x03(\x0b\x32\x33.com.webank.ai.fate.core.mlmodel.buffer.SingleModel\x12\x1b\n\x13one_vs_rest_classes\x18\x02 \x03(\tB\x13\x42\x11LRModelParamProtob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'lr_model_param_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\021LRModelParamProto'
  _LRMODELPARAM_WEIGHTENTRY._options = None
  _LRMODELPARAM_WEIGHTENTRY._serialized_options = b'8\001'
  _LRMODELPARAM_ENCRYPTEDWEIGHTENTRY._options = None
  _LRMODELPARAM_ENCRYPTEDWEIGHTENTRY._serialized_options = b'8\001'
  _SINGLEMODEL_WEIGHTENTRY._options = None
  _SINGLEMODEL_WEIGHTENTRY._serialized_options = b'8\001'
  _SINGLEMODEL_ENCRYPTEDWEIGHTENTRY._options = None
  _SINGLEMODEL_ENCRYPTEDWEIGHTENTRY._serialized_options = b'8\001'
  _globals['_LRMODELPARAM']._serialized_start=90
  _globals['_LRMODELPARAM']._serialized_end=735
  _globals['_LRMODELPARAM_WEIGHTENTRY']._serialized_start=582
  _globals['_LRMODELPARAM_WEIGHTENTRY']._serialized_end=627
  _globals['_LRMODELPARAM_ENCRYPTEDWEIGHTENTRY']._serialized_start=629
  _globals['_LRMODELPARAM_ENCRYPTEDWEIGHTENTRY']._serialized_end=735
  _globals['_SINGLEMODEL']._serialized_start=738
  _globals['_SINGLEMODEL']._serialized_end=1269
  _globals['_SINGLEMODEL_WEIGHTENTRY']._serialized_start=582
  _globals['_SINGLEMODEL_WEIGHTENTRY']._serialized_end=627
  _globals['_SINGLEMODEL_ENCRYPTEDWEIGHTENTRY']._serialized_start=629
  _globals['_SINGLEMODEL_ENCRYPTEDWEIGHTENTRY']._serialized_end=735
  _globals['_ONEVSRESTRESULT']._serialized_start=1271
  _globals['_ONEVSRESTRESULT']._serialized_end=1396
# @@protoc_insertion_point(module_scope)
