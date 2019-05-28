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

import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.core.DataStructure;
import com.webank.ai.eggroll.core.io.KeyValue;
import com.webank.ai.eggroll.core.model.Bytes;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
public class KeyValueToRawEntrySerDes {
    @Autowired
    private ByteStringToBytesSerDes byteStringToBytesSerDes;

    private DataStructure.RawEntry.Builder builder;

    public KeyValueToRawEntrySerDes() {
        this.builder = DataStructure.RawEntry.newBuilder();
    }

    public DataStructure.RawEntry serialize(KeyValue<Bytes, byte[]> kv) {
        this.builder.setKey(byteStringToBytesSerDes.serialize(kv.key))
                .setValue(ByteString.copyFrom(kv.value))
                .build();

        return this.builder.build();
    }

    public KeyValue<Bytes, byte[]> deserialize(DataStructure.RawEntry entry) {
        return KeyValue.pair(byteStringToBytesSerDes.deserialize(entry.getKey()), entry.getValue().toByteArray());
    }
}
