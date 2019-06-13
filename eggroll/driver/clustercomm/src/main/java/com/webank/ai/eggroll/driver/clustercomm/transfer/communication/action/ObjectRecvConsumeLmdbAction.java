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

package com.webank.ai.eggroll.driver.clustercomm.transfer.communication.action;

import com.google.common.collect.Lists;
import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.driver.clustercomm.transfer.manager.RecvBrokerManager;
import com.webank.ai.eggroll.driver.clustercomm.transfer.utils.TransferPojoUtils;
import com.webank.ai.eggroll.driver.clustercomm.constant.ClusterCommConstants;
import com.webank.ai.eggroll.driver.clustercomm.factory.KeyValueStoreFactory;
import com.webank.ai.eggroll.driver.clustercomm.transfer.model.TransferBroker;
import com.webank.ai.eggroll.framework.roll.api.grpc.client.RollKvServiceClient;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Scope("prototype")
public class ObjectRecvConsumeLmdbAction extends BaseRecvConsumeAction {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    protected KeyValueStoreFactory keyValueStoreFactory;
    @Autowired
    protected TransferPojoUtils transferPojoUtils;
    @Autowired
    private RecvBrokerManager recvBrokerManager;
    @Autowired
    private RollKvServiceClient rollKvServiceClient;
    private ClusterComm.TransferMeta finalTransferMeta;
    private ByteString serializedObjectResult;
    private List<ByteString> serializedObjects;
    private long chunkCount = 0;
    private StorageBasic.StorageLocator clusterCommStorageLocator;

    public ObjectRecvConsumeLmdbAction(TransferBroker transferBroker) {
        super(transferBroker);
        this.serializedObjectResult = ByteString.EMPTY;
        this.serializedObjects = Lists.newLinkedList();
    }

    @Override
    public void onInit() {
        super.onInit();
        LOGGER.info("[CLUSTERCOMM][SEND][OBJECT][CONSUMER] objectRecvConsumeAction.onInit. TransferMetaId: {}", transferMetaId);

        if (!transferMeta.hasDataDesc() || !transferMeta.getDataDesc().hasStorageLocator()) {
            ClusterComm.TransferMeta passedInTransferMeta = recvBrokerManager.getPassedInTransferMetaNow(transferMetaId);

            ClusterComm.TransferMeta.Builder finalTransferMetaBuilder = ClusterComm.TransferMeta.newBuilder();

            finalTransferMetaBuilder.mergeFrom(transferMeta)
                    .setDataDesc(passedInTransferMeta.getDataDesc());
            finalTransferMeta = finalTransferMetaBuilder.build();
        } else {
            finalTransferMeta = transferMeta;
        }

        clusterCommStorageLocator = StorageBasic.StorageLocator.newBuilder()
                .setNamespace(finalTransferMeta.getJob().getJobId())
                .setName(ClusterCommConstants.OBJECT_STORAGE_NAMESPACE)
                .setType(StorageBasic.StorageType.LMDB)
                .setFragment(0)
                .build();

        Kv.CreateTableInfo createTableInfo = Kv.CreateTableInfo.newBuilder()
                .setStorageLocator(clusterCommStorageLocator)
                .setFragmentCount(10)
                .build();

        // create new table
        rollKvServiceClient.create(createTableInfo);
    }

    @Override
    public void onProcess() {
        // LOGGER.info("[CLUSTERCOMM][SEND][OBJECT][CONSUMER] objectRecvConsumeAction.onProcess. TransferMetaId: {}", transferMetaId);
        super.onProcess();
        List<ByteString> parts = Lists.newArrayList();
        transferBroker.drainTo(parts);

        int partSize = parts.size();
        if (partSize > 0) {
            serializedObjects.add(ByteString.copyFrom(parts));

            ++chunkCount;
        }
    }

    @Override
    public void onComplete() {
        onProcess();

        serializedObjectResult = ByteString.copyFrom(serializedObjects);

        LOGGER.info("[CLUSTERCOMM][SEND][OBJECT][CONSUMER] objectRecvConsumeAction.onComplete. total size: {}, total chunkCount: {}, transferMetaId: {}",
                serializedObjectResult.size(), chunkCount, transferMetaId);

        BasicMeta.ReturnStatus result = null;

        LOGGER.info("objectRecvConsumeAction: table created");
        StoreInfo storeInfo = StoreInfo.fromStorageLocator(clusterCommStorageLocator);

        Kv.Operand request = Kv.Operand.newBuilder()
                .setKey(finalTransferMeta.getDataDesc().getTaggedVariableName())
                .setValue(serializedObjectResult)
                .build();

        // put operand
        rollKvServiceClient.put(request, storeInfo);
        LOGGER.info("objectRecvConsumeAction: data put into table");

        String transferMetaId = transferPojoUtils.generateTransferId(finalTransferMeta);
        recvBrokerManager.remove(transferMetaId);
        recvBrokerManager.setFinishedTask(finalTransferMeta, ClusterComm.TransferStatus.COMPLETE);

        transferMetaHelper.update(finalTransferMeta);
        transferMeta = finalTransferMeta;
        super.onComplete();
    }
}
