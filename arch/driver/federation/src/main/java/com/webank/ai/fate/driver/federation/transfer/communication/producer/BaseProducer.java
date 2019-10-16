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

package com.webank.ai.fate.driver.federation.transfer.communication.producer;

import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.core.DataStructure;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.factory.ReturnStatusFactory;
import com.webank.ai.eggroll.core.serdes.impl.KeyValueToRawEntrySerDes;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.factory.KeyValueStoreFactory;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.utils.FederationServerUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;

import javax.annotation.PostConstruct;
import java.util.concurrent.Callable;

public abstract class BaseProducer implements Callable<BasicMeta.ReturnStatus> {
    protected static final int DEFAULT_CHUNK_SIZE = 32 << 10; // 32k
    private static final Logger LOGGER = LogManager.getLogger();
    protected final Federation.TransferMeta transferMeta;
    protected final TransferBroker transferBroker;
    @Autowired
    protected StorageMetaClient storageMetaClient;
    @Autowired
    protected KeyValueStoreFactory keyValueStoreFactory;
    @Autowired
    protected KeyValueToRawEntrySerDes keyValueToRawEntrySerDes;
    @Autowired
    protected ErrorUtils errorUtils;
    @Autowired
    protected ReturnStatusFactory returnStatusFactory;
    @Autowired
    protected ToStringUtils toStringUtils;
    @Autowired
    protected FederationServerUtils federationServerUtils;
    protected int chunkSize;

    public BaseProducer(TransferBroker transferBroker) {
        this.transferBroker = transferBroker;
        this.transferMeta = transferBroker.getTransferMeta();
        this.chunkSize = DEFAULT_CHUNK_SIZE;
    }

    @PostConstruct
    public void init() {
        storageMetaClient.init(federationServerUtils.getMetaServiceEndpoint());
    }

    protected synchronized void putToBroker(DataStructure.RawMap.Builder rawMapBuilder) {
        // LOGGER.warn("--- broker: {}", transferBroker);
        DataStructure.RawMap chunk = rawMapBuilder.build();
        // serialize and ready to send
        putToBroker(chunk.toByteString());
        rawMapBuilder.clear();
    }

    protected void putToBroker(ByteString byteString) {
        transferBroker.put(byteString);
    }
}
