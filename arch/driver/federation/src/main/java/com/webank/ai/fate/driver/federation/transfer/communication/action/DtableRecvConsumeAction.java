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

package com.webank.ai.fate.driver.federation.transfer.communication.action;

import com.google.common.collect.Lists;
import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.core.DataStructure;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.constant.StringConstants;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.serdes.impl.KeyValueToRawEntrySerDes;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.transfer.manager.RecvBrokerManager;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.transfer.utils.PrintUtils;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferPojoUtils;
import com.webank.ai.fate.driver.federation.utils.FederationServerUtils;
import com.webank.ai.fate.driver.federation.utils.ThreadPoolTaskExecutorUtil;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.Dtable;
import com.webank.ai.eggroll.framework.roll.api.grpc.client.RollKvServiceClient;
import com.webank.ai.eggroll.framework.roll.factory.RollModelFactory;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBroker;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;
import org.springframework.util.concurrent.ListenableFuture;
import org.springframework.util.concurrent.ListenableFutureCallback;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

@Component
@Scope("prototype")
public class DtableRecvConsumeAction extends BaseRecvConsumeAction {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    protected RollModelFactory rollModelFactory;
    @Autowired
    protected KeyValueToRawEntrySerDes keyValueToRawEntrySerDes;
    @Autowired
    private RollKvServiceClient rollKvServiceClient;
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private RecvBrokerManager recvBrokerManager;
    @Autowired
    private TransferPojoUtils transferPojoUtils;
    @Autowired
    private ThreadPoolTaskExecutor ioConsumerPool;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private PrintUtils printUtils;
    @Autowired
    private FederationServerUtils federationServerUtils;
    private Federation.TransferMeta finalTransferMeta;
    private OperandBroker operandBroker;
    private int entryCount = 0;

    private final CountDownLatch putAllFinishLatch;
    private ListenableFuture<?> putallListenableFuture;

    public DtableRecvConsumeAction(TransferBroker transferBroker) {
        super(transferBroker);
        putAllFinishLatch = new CountDownLatch(1);
    }

    @Override
    public void onInit() {
        LOGGER.info("[FEDERATION][SEND][DTABLE][CONSUMER] DtableRecvConsumeAction.onInit. transferMetaId: {}",
                transferMetaId);
        super.onInit();

        // todo: make this configurable
        this.operandBroker = rollModelFactory.createOperandBroker(500_000);
        storageMetaClient.init(federationServerUtils.getMetaServiceEndpoint());

        // if no data specified in recv side, namespace: jobid; name: "__" + transferMetaId; fragment: use sender's
        if (!transferMeta.hasDataDesc() || !transferMeta.getDataDesc().hasStorageLocator()) {
            Federation.TransferMeta passedInTransferMeta = recvBrokerManager.getPassedInTransferMetaNow(transferMetaId);
            Federation.TransferDataDesc passedInTransferDataDesc = passedInTransferMeta.getDataDesc();

            StorageBasic.StorageLocator.Builder storageLocatorBuilder = StorageBasic.StorageLocator.newBuilder();
            Federation.TransferMeta.Builder finalTransferMetaBuilder = Federation.TransferMeta.newBuilder();
            Federation.TransferDataDesc.Builder dataDescBuilder = Federation.TransferDataDesc.newBuilder();

            // table info
            storageLocatorBuilder.setNamespace(transferMeta.getJob().getJobId())
                    .setName(StringConstants.DOUBLE_UNDERLINES + transferMetaId)
                    .setFragment(passedInTransferDataDesc.getStorageLocator().getFragment())    // total fragment count
                    .setType(passedInTransferDataDesc.getStorageLocator().getType());

            dataDescBuilder.setStorageLocator(storageLocatorBuilder)
                    .setTransferDataType(Federation.TransferDataType.DTABLE)
                    .setTaggedVariableName(passedInTransferMeta.getDataDesc().getTaggedVariableName());

            finalTransferMetaBuilder.mergeFrom(transferMeta)
                    .setDataDesc(dataDescBuilder);
            finalTransferMeta = finalTransferMetaBuilder.build();

            // todo: return value check
            transferMetaHelper.update(finalTransferMeta);
        } else {
            finalTransferMeta = transferMeta;
        }

        Federation.TransferDataDesc finalTransferDataDesc = finalTransferMeta.getDataDesc();
        StorageBasic.StorageLocator finalStorageLocator = finalTransferDataDesc.getStorageLocator();
        Dtable dtable = storageMetaClient.getTable(typeConversionUtils.toDtable(finalStorageLocator));
        int fragmentCount = finalStorageLocator.getFragment();

        // create if not exists. using passed in fragment count
        if (dtable == null) {
            // double check
            if (fragmentCount <= 0) {
                throw new IllegalArgumentException("invalid passed in fragment count");
            }

            Kv.CreateTableInfo createTableInfo = Kv.CreateTableInfo.newBuilder()
                    .setStorageLocator(finalStorageLocator)
                    .setFragmentCount(fragmentCount)
                    .build();
            rollKvServiceClient.create(createTableInfo);
        }

        StoreInfo storeInfo = StoreInfo.fromStorageLocator(finalStorageLocator);
        // storeInfo.setFragment(fragmentCount);

        // todo: consider if this should be pulled up and be monitored
        putallListenableFuture= ThreadPoolTaskExecutorUtil.submitListenable(ioConsumerPool,() -> {
            rollKvServiceClient.putAll(operandBroker, storeInfo);
        },new int[]{500,1000,5000},new int[]{5,5,3});
//        putallListenableFuture = ioConsumerPool.submitListenable(() -> {
//            rollKvServiceClient.putAll(operandBroker, storeInfo);
//        });
        putallListenableFuture.addCallback(new ListenableFutureCallback<Object>() {
            @Override
            public void onFailure(Throwable throwable) {
                LOGGER.error("[FEDERATION][RECV][CONSUMER][DTABLE] addAll failed for data: {}, error: {}",
                        transferMetaId, errorUtils.getStackTrace(throwable));
                putAllFinishLatch.countDown();
            }

            @Override
            public void onSuccess(Object o) {
                LOGGER.info("[FEDERATION][RECV][CONSUMER][DTABLE] addAll finished for data: {}",
                        transferMetaId);
                putAllFinishLatch.countDown();
            }
        });

        LOGGER.info("DtableRecvConsumeAction: transferBroker size: {}", transferBroker.getQueueSize());
    }

    @Override
    public void onProcess() {
        // LOGGER.info("DtableRecvConsumeAction.onProcess");
        super.onProcess();
        List<ByteString> rawMaps = Lists.newArrayList();
        transferBroker.drainTo(rawMaps);

        try {
            if (!rawMaps.isEmpty()) {
                for (ByteString rawMapBS : rawMaps) {
                    DataStructure.RawMap rawMap = DataStructure.RawMap.parseFrom(rawMapBS);

                    for (DataStructure.RawEntry entry : rawMap.getEntriesList()) {
                        Kv.Operand operand = typeConversionUtils.toOperand(entry);
                        // LOGGER.info("operand: {}", toStringUtils.toOneLineString(operand));
                        operandBroker.put(operand);
                        ++entryCount;
                    }
                }
            }
        } catch (Exception e) {
            LOGGER.error(errorUtils.getStackTrace(e));
            throw new RuntimeException(e);
        }
    }

    @Override
    public void onComplete() {
        LOGGER.info("[FEDERATION][SEND][DTABLE][CONSUMER] trying to complete DtableRecvConsumeAction. entryCount: {}, transferMetaId: {}",
                entryCount, transferMetaId);
        while (!transferBroker.isClosable()) {
            onProcess();
        }

        LOGGER.info("[FEDERATION][SEND][DTABLE][CONSUMER] actual completes DtableRecvConsumeAction. entryCount: {}, transferMetaId: {}",
                entryCount, transferMetaId);
        if (transferBroker.hasError()) {
            throw new RuntimeException(transferBroker.getError());
        }

        // operand is new-ed here so setting finished here
        operandBroker.setFinished();

        try {
            long startWaitTime = System.currentTimeMillis();
            boolean latchAwaitResult = false;
            while (!latchAwaitResult) {
                latchAwaitResult = putAllFinishLatch.await(1, TimeUnit.SECONDS);
                long now = System.currentTimeMillis();

                if (((now - startWaitTime) / 1000) % 10 == 0) {
                    LOGGER.info("[FEDERATION][RECV][CONSUMER][DTABLE] addAll waiting to finished: {}", transferMetaId);
                }

                // todo: add timeout mechanism
            }
        } catch (Exception e) {
            LOGGER.error("[FEDERATION][RECV][CONSUMER][DTABLE] addAll latch await failed. exception: {}", errorUtils.getStackTrace(e));
        }

        String transferMetaId = transferPojoUtils.generateTransferId(transferBroker.getTransferMeta());
        recvBrokerManager.remove(transferMetaId);
        recvBrokerManager.setFinishedTask(finalTransferMeta, Federation.TransferStatus.COMPLETE);

        printUtils.printTransferBrokerTemporaryHolder();
        super.onComplete();
    }
}
