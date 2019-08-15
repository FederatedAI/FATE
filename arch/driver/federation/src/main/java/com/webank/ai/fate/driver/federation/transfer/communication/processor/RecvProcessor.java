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

import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.fate.driver.federation.factory.TransferServiceFactory;
import com.webank.ai.fate.driver.federation.transfer.communication.action.TransferQueueConsumeAction;
import com.webank.ai.fate.driver.federation.transfer.communication.consumer.TransferBrokerConsumer;
import com.webank.ai.fate.driver.federation.transfer.manager.RecvBrokerManager;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.utils.ThreadPoolTaskExecutorUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

@Component
@Scope("prototype")
public class RecvProcessor extends BaseTransferProcessor {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private RecvBrokerManager recvBrokerManager;
    @Autowired
    private TransferServiceFactory transferServiceFactory;

    public RecvProcessor(Federation.TransferMeta transferMeta) {
        super(transferMeta);
    }

    @Override
    public void run() {
        try {
            LOGGER.info("[FEDERATION][RECV][PROCESSOR] transferMetaId: {}", transferMetaId);
            String transferMetaId = transferPojoUtils.generateTransferId(transferMeta);

            TransferBroker transferBroker = recvBrokerManager.getBroker(transferMetaId);
            if (transferBroker == null) {
                transferBroker = recvBrokerManager.createIfNotExists(transferMeta);

/*                if (transferBroker == null) {
                    transferBroker = recvBrokerManager.getBroker(transferMetaId);
                }*/
            }

            transferBroker = recvBrokerManager.getBroker(transferMetaId);
            LOGGER.info("[FEDERATION][RECV][PROCESSOR] broker: {}, transferMetaId: {}", transferBroker, transferMetaId);

            //LOGGER.info("recv async broker status before: {}", transferBroker.getBrokerStatus().name());
            if (transferBroker.getTransferMeta() == null) {
                transferBroker.setTransferMeta(transferMeta);
            }
            //LOGGER.info("recv async broker status after: {}", transferBroker.getBrokerStatus().name());

            TransferQueueConsumeAction recvConsumeAction = null;

            Federation.TransferDataType transferDataType = transferMeta.getDataDesc().getTransferDataType();

//            LOGGER.info("transferType: {}", transferDataType == null ? "null" : transferDataType.name());

            // if transfer data type is not specified in recv request, then use passed in type from send side
            if (transferDataType == null || transferDataType == Federation.TransferDataType.NOT_SPECIFIED) {
                // todo: make this configurable
                Federation.TransferMeta passedInTransferMeta = recvBrokerManager.blockingGetPassedInTransferMeta(transferMetaId,
                        1, TimeUnit.DAYS);
                if (passedInTransferMeta == null) {
                    throw new TimeoutException("time exceeds when waiting for send request");
                }
                transferDataType = passedInTransferMeta.getDataDesc().getTransferDataType();
            }

            switch (transferDataType) {
                case DTABLE:
                    recvConsumeAction = transferServiceFactory.createDtableRecvConsumeAction(transferBroker);
                    break;
                case OBJECT:
                    recvConsumeAction = transferServiceFactory.createObjectRecvConsumeLmdbAction(transferBroker);
                    break;
                default:
                    throw new UnsupportedOperationException("illegal transfer type for transfermeta: " + toStringUtils.toOneLineString(transferMeta));
            }

            transferBroker.setAction(recvConsumeAction);

            TransferBrokerConsumer consumer = transferServiceFactory.createTransferBrokerConsumer();
            transferBroker.addSubscriber(consumer);
            ThreadPoolTaskExecutorUtil.submitListenable(ioConsumerPool,consumer,new int[]{500,1000,5000},new int[]{5,5,3});

            //ioConsumerPool.submit(consumer);

            consumer.onListenerChange(transferBroker);

            LOGGER.info("[FEDERATION][RECV][PROCESSOR] broker isFinished: {}", transferBroker.isFinished());
        } catch (Exception e) {
            LOGGER.error(errorUtils.getStackTrace(e));
            throw new RuntimeException(e);
        }
    }
}
