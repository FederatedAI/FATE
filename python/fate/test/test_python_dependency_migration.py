#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from fate.arch import Context
from fate.arch.federation.backends.osx import osx_pb2, osx_pb2_grpc
from fate.components.components.nn.nn_runner import load_model_dict_from_path
from fate.ml.nn.trainer.trainer_base import FedArguments, HomoTrainerClient, TrainingArguments
from fate.ml.utils.model_serdes import serialize_models


def test_protobuf_model_serialization_preserves_default_fields():
    message = osx_pb2.Outbound(payload=b"model")
    name, wire, json = serialize_models({"test": message})["test"]

    assert name == "Outbound"
    assert osx_pb2.Outbound.FromString(wire) == message
    assert json == {"payload": "bW9kZWw=", "code": "", "message": "", "metadata": {}}


def test_osx_grpc_round_trip():
    class Echo(osx_pb2_grpc.PrivateTransferProtocolServicer):
        def invoke(self, request, context):
            return osx_pb2.Outbound(payload=request.payload, metadata=request.metadata)

    with ThreadPoolExecutor(max_workers=1) as workers:
        server = grpc.server(workers)
        osx_pb2_grpc.add_PrivateTransferProtocolServicer_to_server(Echo(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = osx_pb2_grpc.PrivateTransferProtocolStub(channel)
                reply = stub.invoke(osx_pb2.Inbound(payload=b"hello", metadata={"source": "test"}), timeout=5)
            assert reply.payload == b"hello"
            assert reply.metadata == {"source": "test"}
        finally:
            server.stop(grace=0).wait(timeout=5)


def test_fate_legacy_training_options():
    with pytest.warns(FutureWarning, match="safetensors"):
        args = TrainingArguments(evaluation_strategy="epoch", no_cuda=True, use_cpu=False, save_safetensors=False)

    assert args.eval_strategy == "epoch"
    assert args.do_eval
    assert args.device.type == "cpu"
    assert args.to_dict()["eval_strategy"] == "epoch"
    assert not {"evaluation_strategy", "no_cuda", "save_safetensors"}.intersection(args.to_dict())


def test_fate_tuple_training_and_checkpoint_compatibility(tmp_path):
    torch.manual_seed(41)
    model = nn.Linear(2, 1)
    inputs = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    dataset = TensorDataset(inputs, torch.tensor([[1.0], [0.0]]))
    trainer = HomoTrainerClient(
        ctx=Context(),
        model=model,
        training_args=TrainingArguments(output_dir=str(tmp_path), max_steps=1),
        fed_args=FedArguments(),
        train_set=dataset,
        val_set=dataset,
        loss_fn=nn.MSELoss(),
        local_mode=True,
    )
    previous = model.weight.detach().clone()
    assert trainer.train().global_step == 1
    assert not torch.equal(model.weight, previous)
    assert trainer.predict(dataset).predictions.shape == (2, 1)

    legacy_path = tmp_path / "pytorch_model.bin"
    legacy = {"weight": torch.ones_like(model.weight)}
    torch.save(legacy, legacy_path)
    torch.testing.assert_close(load_model_dict_from_path(str(tmp_path))["weight"], legacy["weight"])

    trainer.save_model()
    assert (tmp_path / "model.safetensors").is_file()
    torch.testing.assert_close(load_model_dict_from_path(str(tmp_path))["weight"], model.weight)


def test_fate_offline_transformers_training_and_tokenizer(tmp_path):
    class TinyDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                "input_ids": torch.tensor([1, 2 + index, 0]),
                "attention_mask": torch.tensor([1, 1, 0]),
                "labels": torch.tensor(index),
            }

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "fate": 3}, unk_token="[UNK]"))
    )
    model = BertForSequenceClassification(
        BertConfig(vocab_size=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    )
    dataset = TinyDataset()
    trainer = HomoTrainerClient(
        ctx=Context(),
        model=model,
        training_args=TrainingArguments(output_dir=str(tmp_path), max_steps=1),
        fed_args=FedArguments(),
        train_set=dataset,
        val_set=dataset,
        tokenizer=tokenizer,
        local_mode=True,
        use_hf_default_behavior=True,
    )
    assert trainer.processing_class is tokenizer
    assert trainer.train().global_step == 1
    assert trainer.predict(dataset).predictions.shape == (2, 2)

    trainer.save_model()
    restored = BertForSequenceClassification.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(restored.classifier.weight, model.classifier.weight)
    assert (tmp_path / "tokenizer.json").is_file()
