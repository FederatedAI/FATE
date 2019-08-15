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

package com.webank.ai.fate.core.storage.dtable;

import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.eggroll.storage.KVServiceGrpc;
import com.webank.ai.fate.api.eggroll.storage.Kv;
import com.webank.ai.fate.api.eggroll.storage.StorageBasic;
import com.webank.ai.fate.core.constant.MetaConstants.CompositeHeaderKey;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.stub.MetadataUtils;
import com.webank.ai.fate.core.network.grpc.client.GrpcClientPool;
import com.webank.ai.fate.core.utils.Configuration;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class DistributedDTable implements DTable {
    private static final Logger LOGGER = LogManager.getLogger();
    private ManagedChannel channel;
    private String name;
    private String nameSpace;
    private int partition;

    public DistributedDTable(String name, String nameSpace, int partition) {
        this.channel = GrpcClientPool.getChannel(Configuration.getProperty("roll"));
        this.name = name;
        this.nameSpace = nameSpace;
        this.partition = partition;
    }

    @Override
    public byte[] get(String key) {
        Kv.Operand.Builder requestOperand = Kv.Operand.newBuilder();
        requestOperand.setKey(ByteString.copyFrom(key.getBytes()));
        KVServiceGrpc.KVServiceBlockingStub kvServiceBlockingStub = KVServiceGrpc.newBlockingStub(this.channel);
        Kv.Operand resultOperand = MetadataUtils.attachHeaders(kvServiceBlockingStub, this.genHeader()).get(requestOperand.build());
        if (resultOperand.getValue() != null) {
            return resultOperand.getValue().toByteArray();
        } else {
            return null;
        }
    }

    @Override
    public void put(String key, byte[] value) {
        StorageBasic.StorageLocator.Builder storageLocator = StorageBasic.StorageLocator.newBuilder();
        storageLocator.setType(StorageBasic.StorageType.LMDB)
                .setNamespace(this.nameSpace)
                .setName(this.name);

        Kv.CreateTableInfo.Builder tableInfo = Kv.CreateTableInfo.newBuilder();
        tableInfo.setStorageLocator(storageLocator.build()).setFragmentCount(this.partition);

        KVServiceGrpc.KVServiceBlockingStub kvServiceBlockingStub = KVServiceGrpc.newBlockingStub(channel);
        Kv.CreateTableInfo newTableInfo = kvServiceBlockingStub.createIfAbsent(tableInfo.build());

        Kv.Operand.Builder requestOperand = Kv.Operand.newBuilder();
        requestOperand.setKey(ByteString.copyFrom(key.getBytes()));
        requestOperand.setValue(ByteString.copyFrom(value));
        MetadataUtils.attachHeaders(kvServiceBlockingStub, this.genHeader()).put(requestOperand.build());
    }

    @Override
    public Map<String, byte[]> collect() {
        Map<String, byte[]> result = new HashMap<>();
        Kv.Range.Builder rangeOrBuilder = Kv.Range.newBuilder();
        KVServiceGrpc.KVServiceBlockingStub kvServiceBlockingStub = KVServiceGrpc.newBlockingStub(this.channel);
        Iterator<Kv.Operand> item = MetadataUtils.attachHeaders(kvServiceBlockingStub, this.genHeader()).iterate(rangeOrBuilder.build());
        while (item.hasNext()) {
            Kv.Operand tmp = item.next();
            result.put(tmp.getKey().toStringUtf8(), tmp.getValue().toByteArray());
        }
        return result;
    }

    private Metadata genHeader() {
        Metadata header = new Metadata();
        header.put(CompositeHeaderKey.from("table_name").asMetaKey(), this.name);
        header.put(CompositeHeaderKey.from("name_space").asMetaKey(), this.nameSpace);
        header.put(CompositeHeaderKey.from("store_type").asMetaKey(), "LMDB");
        return header;
    }
}
