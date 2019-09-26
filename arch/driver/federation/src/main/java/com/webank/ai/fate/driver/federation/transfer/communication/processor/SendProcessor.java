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

package com.webank.ai.fate.driver.federation.transfer.communication.processor;

import com.google.common.collect.Lists;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.core.error.exception.StorageNotExistsException;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.fate.driver.federation.factory.FederationCallbackFactory;
import com.webank.ai.fate.driver.federation.transfer.api.grpc.client.ProxyClient;
import com.webank.ai.fate.driver.federation.transfer.communication.action.TransferQueueConsumeAction;
import com.webank.ai.fate.driver.federation.transfer.communication.consumer.TransferBrokerConsumer;
import com.webank.ai.fate.driver.federation.transfer.communication.producer.DtableFragmentSendProducer;
import com.webank.ai.fate.driver.federation.transfer.communication.producer.ObjectLmdbSendProducer;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.transfer.service.ProxySelectionService;
import com.webank.ai.fate.driver.federation.utils.ThreadPoolTaskExecutorUtil;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.Dtable;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.Fragment;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;
import org.springframework.util.concurrent.ListenableFuture;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;

@Component
@Scope("prototype")
public class SendProcessor extends BaseTransferProcessor {
    private static Logger LOGGER = LogManager.getLogger();
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private ProxySelectionService proxySelectionService;
    @Autowired
    private ProxyClient proxyClient;
    @Autowired
    private FederationCallbackFactory federationCallbackFactory;

    public SendProcessor(Federation.TransferMeta transferMeta) {
        super(transferMeta);
    }

    @Override
    public void run() {
        LOGGER.info("[FEDERATION][SEND][PROCESSOR] begin");
        BasicMeta.Endpoint targetProxy = null;
        try {
            LOGGER.info("[FEDERATION][SEND][PROCESSOR] transferMetaId: {}", transferMetaId);
            targetProxy = proxySelectionService.select();
            Federation.TransferDataDesc dataDesc = transferMeta.getDataDesc();

            // request send start
            proxyClient.requestSendStart(transferMeta, targetProxy);

            final List<Throwable> errorContainer = Collections.synchronizedList(Lists.newLinkedList());
            CountDownLatch finishLatch = null;
            switch (dataDesc.getTransferDataType()) {
                case DTABLE:
                    finishLatch = processDtable(dataDesc, errorContainer);
                    break;
                case OBJECT:
                    finishLatch = processObject(dataDesc, errorContainer);
                    break;
                default:
                    throw new IllegalArgumentException("unsupport send data type for: " + dataDesc.getTransferDataType().name());
            }

            // wait for send finish
            while (!finishLatch.await(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)) {
                LOGGER.info("[FEDERATION][SEND][PROCESSOR] waiting for send job finished: {}", transferMetaId);
            }
        } catch (Exception e) {
            LOGGER.error(errorUtils.getStackTrace(e));
            throw new RuntimeException(e);
        } finally {
            // request send end
            proxyClient.requestSendEnd(transferMeta, targetProxy);
            LOGGER.info("[FEDERATION][SEND][PROCESSOR] send job finished: {}", transferMetaId);
        }
    }

    private CountDownLatch processDtable(Federation.TransferDataDesc dataDesc,
                                         final List<Throwable> errorContainer) {
        LOGGER.info("[FEDERATION][SEND][PROCESSOR][DTABLE] transferMetaId: {}", transferMetaId);
        StorageBasic.StorageLocator storageLocator = dataDesc.getStorageLocator();
        Dtable dtable = storageMetaClient.getTable(storageLocator.getNamespace(), storageLocator.getName());

        if (dtable == null || dtable.getTableId() == null) {
            transferMetaHelper.onError(transferMeta, 202, "no table exist: " + dtable);
            // todo: add more error handling
            throw new StorageNotExistsException(storageLocator);
        }

        List<Fragment> fragments = storageMetaClient.getFragmentsByTableId(dtable.getTableId());

        BasicMeta.Endpoint targetProxy = proxySelectionService.select();

        int fragmentSize = fragments.size();
        final List<BasicMeta.ReturnStatus> results = Collections.synchronizedList(Lists.newArrayList());
        CountDownLatch finishLatch = new CountDownLatch(fragmentSize);

        for (Fragment fragment : fragments) {
            // todo: make this configurable
            final TransferBroker broker = transferServiceFactory.createTransferBroker(transferMeta, 10_000);
            TransferQueueConsumeAction sendConsumeAction = transferServiceFactory.createSendConsumeAction(broker, targetProxy);
            broker.setAction(sendConsumeAction);

            // attach broker to producer so that data can be put in attached broker
            // each producer has its own broker. no broker sharing among producers
            DtableFragmentSendProducer producer = transferServiceFactory.createDtableFragmentSendProducer(fragment, broker);

            // consumer subscribes broker
            TransferBrokerConsumer consumer = transferServiceFactory.createTransferBrokerConsumer();
            broker.addSubscriber(consumer);

            // todo: add result tracking and retry mechanism
            ListenableFuture<BasicMeta.ReturnStatus> producerResult = (ListenableFuture<BasicMeta.ReturnStatus>)ThreadPoolTaskExecutorUtil.submitListenable(ioProducerPool,producer,new int[]{500,1000,5000},new int[]{5,5,3});
            producerResult.addCallback(
                    federationCallbackFactory.createDtableSendProducerListenableCallback(results, broker, errorContainer, null, -1));
            ListenableFuture<?> consumerListenableFuture = ThreadPoolTaskExecutorUtil.submitListenable(ioConsumerPool,consumer,new int[]{500,1000,5000},new int[]{5,5,3});

            consumerListenableFuture.addCallback(
                    federationCallbackFactory.createDefaultConsumerListenableCallback(errorContainer, finishLatch, null, -1));
        }

        return finishLatch;
    }

    private CountDownLatch processObject(Federation.TransferDataDesc dataDesc,
                                         final List<Throwable> errorContainer) {
        LOGGER.info("[FEDERATION][SEND][PROCESSOR][OBJECT] transferMetaId: {}", transferMetaId);
        TransferBroker broker = transferServiceFactory.createTransferBroker(transferMeta);
        TransferQueueConsumeAction sendConsumeAction = transferServiceFactory.createSendConsumeAction(broker, proxySelectionService.select());
        broker.setAction(sendConsumeAction);

        ObjectLmdbSendProducer producer = transferServiceFactory.createLmdbSendProducer(broker);

        TransferBrokerConsumer consumer = transferServiceFactory.createTransferBrokerConsumer();
        broker.addSubscriber(consumer);

        final List<BasicMeta.ReturnStatus> results = Collections.synchronizedList(Lists.newArrayList());
        CountDownLatch finishLatch = new CountDownLatch(1);

        ListenableFuture<BasicMeta.ReturnStatus> producerListenableFuture = ThreadPoolTaskExecutorUtil.submitListenable(ioProducerPool,producer,new int[]{500,1000,5000},new int[]{5,5,3});

        producerListenableFuture.addCallback(
                federationCallbackFactory.createDtableSendProducerListenableCallback(results, broker, errorContainer, null, -1));

        ListenableFuture<?> consumerListenableFuture = ThreadPoolTaskExecutorUtil.submitListenable(ioConsumerPool,consumer,new int[]{500,1000,5000},new int[]{5,5,3});


        consumerListenableFuture.addCallback(
                federationCallbackFactory.createDefaultConsumerListenableCallback(errorContainer, finishLatch, null, -1));

        return finishLatch;
    }

}
