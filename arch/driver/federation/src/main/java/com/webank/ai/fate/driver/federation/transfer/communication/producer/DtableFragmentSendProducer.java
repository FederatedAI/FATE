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

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.core.DataStructure;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.core.error.exception.CrudException;
import com.webank.ai.eggroll.core.io.KeyValue;
import com.webank.ai.eggroll.core.io.KeyValueIterator;
import com.webank.ai.eggroll.core.io.KeyValueStore;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.factory.KeyValueStoreFactory;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.transfer.utils.PrintUtils;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.Fragment;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.Node;
import com.webank.ai.eggroll.framework.storage.service.model.RemoteKeyValueStore;
import com.webank.ai.eggroll.framework.storage.service.model.enums.Stores;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
public class DtableFragmentSendProducer extends BaseProducer {
    private static final Logger LOGGER = LogManager.getLogger();
    private final Fragment fragment;
    @Autowired
    private PrintUtils printUtils;
    @Autowired
    private ToStringUtils toStringUtils;

    public DtableFragmentSendProducer(Fragment fragment, TransferBroker transferBroker) {
        super(transferBroker);
        this.fragment = fragment;
        this.chunkSize = DEFAULT_CHUNK_SIZE;
    }

    @Override
    public BasicMeta.ReturnStatus call() throws Exception {
        String fragmentString = toStringUtils.toOneLineString(fragment);
        LOGGER.info("[FEDERATION][PRODUCER][DTABLE] dtable send producer: {}", fragmentString);
        BasicMeta.ReturnStatus result = null;

        // printUtils.printTransferBrokerTemporaryHolder("in send producer");

        Node node = storageMetaClient.getNodeByFragmentId(fragment.getFragmentId());
        if (node == null) {
            throw new CrudException(301, "no such node");
        }
        KeyValueStore<Bytes, byte[]> keyValueStore = null;
        int entryCount = 0;
        int packetCount = 0;

        Federation.TransferDataDesc dataDesc = transferMeta.getDataDesc();
        StorageBasic.StorageLocator storageLocator = dataDesc.getStorageLocator();

        String storageLocatorString = toStringUtils.toOneLineString(storageLocator);
        try {
            // todo: defaulting to kv store. need to consider redis in future release
            KeyValueStoreFactory.KeyValueStoreBuilder builder = keyValueStoreFactory.createKeyValueStoreBuilder();

            // todo: make this configurable
            String target = node.getIp();
            if (StringUtils.isBlank(target)) {
                target = node.getHost();
            }

            // getting transfer data desc
            builder.setDataDir(RuntimeConstants.getDefaultDataDir())
                    .setHost(target)
                    .setPort(node.getPort())
                    .setStorageType(Stores.valueOf(Stores.class, storageLocator.getType().name()))
                    .setNamespace(storageLocator.getNamespace())
                    .setTableName(storageLocator.getName())
                    .setFragment(fragment.getFragmentOrder())
                    .setStorageType(Stores.valueOf(storageLocator.getType().name()));

            keyValueStore = builder.build(RemoteKeyValueStore.class);

            // reading all data out and send
            try (KeyValueIterator<Bytes, byte[]> iterator = keyValueStore.all()) {
                LOGGER.info("[FEDERATION][PRODUCER][DTABLE] count of keyValueStore {}: {}", storageLocatorString, keyValueStore.count());
                KeyValue<Bytes, byte[]> cur = null;
                DataStructure.RawEntry entry = null;
                DataStructure.RawMap.Builder rawMapBuilder = DataStructure.RawMap.newBuilder();

                int serializedSize = 0;
                while (iterator.hasNext()) {
                    cur = iterator.next();
                    entry = keyValueToRawEntrySerDes.serialize(cur);
                    serializedSize += entry.getSerializedSize();

                    rawMapBuilder.addEntries(entry);

                    if (serializedSize >= chunkSize) {
                        ++packetCount;
                        putToBroker(rawMapBuilder);
                        serializedSize = 0;
                    }
                }

                if (rawMapBuilder.getEntriesCount() > 0) {
                    ++packetCount;
                    putToBroker(rawMapBuilder);
                }
                if (entryCount == 0) {
                    LOGGER.warn("[FEDERATION][PRODUCER][DTABLE] empty entryCount: {}", fragmentString);
                }
                // transferBroker.setFinished();
                result = returnStatusFactory.createSucessful("[FEDERATION][PRODUCER][DTABLE] fragment sendProducer Done. fragment: " + fragmentString);
            } // end try resource
        } catch (Exception e) {
            LOGGER.info("[FEDERATION][PRODUCER][DTABLE][ERROR] Fragment send producer exception: " + errorUtils.getStackTrace(e));
            throw e;
        } finally {
            LOGGER.info("[FEDERATION][PRODUCER][DTABLE] finish producing data for {}, total entryCount: {}, packetCount: {}", storageLocatorString, entryCount, packetCount);
            if (keyValueStore != null) {
                keyValueStore.close();
            }
        }
        return result;
    }
}
