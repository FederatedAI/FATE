/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.webank.ai.eggroll.core.serdes.impl;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.io.KeyValue;
import com.webank.ai.eggroll.core.io.Operand;
import com.webank.ai.eggroll.core.model.Bytes;

import java.nio.ByteBuffer;

// todo: make this comply with serdes interface
public class POJOUtils {

    public static Kv.Operand buildOperand(KeyValue<Bytes, byte[]> keyValue) {
        return Kv.Operand.newBuilder().setKey(ByteString.copyFrom(keyValue.key.get()))
                .setValue(ByteString.copyFrom(keyValue.value)).build();
    }

    public static KeyValue<Bytes, byte[]> buildKeyValue(Bytes key, byte[] value) {
        return KeyValue.pair(key, value);
    }

    public static Kv.Operand buildOperand(Bytes key, byte[] value) {
        return Kv.Operand.newBuilder().setKey(ByteString.copyFrom(key.get()))
                .setValue(ByteString.copyFrom(value)).build();
    }

    public static KeyValue<Bytes, byte[]> buildKeyValue(Kv.Operand op) {
        return KeyValue.pair(Bytes.wrap(op.getKey().toByteArray()), op.getValue().toByteArray());
    }

    public static byte[] extractBytes(ByteBuffer byteBuffer) {
        if (byteBuffer == null) {
            return new byte[0];
        }
        System.out.println(byteBuffer.getClass());
        byte[] rtn = new byte[byteBuffer.remaining()];
        byteBuffer.get(rtn);
        return rtn;
    }

    public static KeyValue<Bytes, byte[]> buildKeyValue(Operand operand) {
        byte[] key = extractBytes(operand.keyAsByteBuffer());
        byte[] value = extractBytes(operand.valueAsByteBuffer());
        return new KeyValue<>(Bytes.wrap(key), value);
    }

    public static Operand createOperand(Bytes key, byte[] value) {
        FlatBufferBuilder fbb = new FlatBufferBuilder();
        int kVec = fbb.createByteVector(key.get());
        int vVec;
        if (value != null) {
            vVec = fbb.createByteVector(value);
        } else {
            vVec = fbb.createByteVector(new byte[0]);
        }
        Operand.startOperand(fbb);
        Operand.addKey(fbb, kVec);
        Operand.addValue(fbb, vVec);
        fbb.finish(Operand.endOperand(fbb));
        return Operand.getRootAsOperand(fbb.dataBuffer());
    }
}
