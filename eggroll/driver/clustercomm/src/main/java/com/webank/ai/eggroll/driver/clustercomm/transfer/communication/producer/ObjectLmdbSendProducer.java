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

package com.webank.ai.eggroll.driver.clustercomm.transfer.communication.producer;

import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.driver.clustercomm.constant.ClusterCommConstants;
import com.webank.ai.eggroll.driver.clustercomm.transfer.model.TransferBroker;
import com.webank.ai.eggroll.driver.clustercomm.transfer.utils.TransferPojoUtils;
import com.webank.ai.eggroll.framework.roll.api.grpc.client.RollKvServiceClient;
import com.webank.ai.eggroll.framework.storage.service.model.enums.Stores;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.nio.ByteBuffer;

@Component
@Scope("prototype")
public class ObjectLmdbSendProducer extends BaseProducer {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private RollKvServiceClient rollKvServiceClient;
    @Autowired
    private TransferPojoUtils transferPojoUtils;

    public ObjectLmdbSendProducer(TransferBroker transferBroker) {
        super(transferBroker);
    }

    @Override
    public BasicMeta.ReturnStatus call() throws Exception {
        LOGGER.info("[CLUSTERCOMM][SEND][OBJECT][PRODUCER] start. jobId: {}");
        BasicMeta.ReturnStatus result = null;
        try {
            StoreInfo storeInfo = StoreInfo.builder()
                    .nameSpace(transferMeta.getJob().getJobId())
                    .tableName(ClusterCommConstants.OBJECT_STORAGE_NAMESPACE)
                    .type(Stores.LMDB.name())
                    .build();

            Kv.Operand request = Kv.Operand.newBuilder()
                    .setKey(transferMeta.getDataDesc().getTaggedVariableName())
                    .build();

            Kv.Operand response = rollKvServiceClient.get(request, storeInfo);

            byte[] value = response.getValue().toByteArray();

            int valueLength = value.length;
            int readCount = 0;
            int chunkCount = 0;

            ByteBuffer byteBuffer = ByteBuffer.wrap(value);

            while (byteBuffer.hasRemaining()) {
                ++chunkCount;

                ByteString bs = ByteString.copyFrom(byteBuffer, Math.min(chunkSize, byteBuffer.remaining()));
                putToBroker(bs);
                readCount += bs.size();
            }
            LOGGER.info("[CLUSTERCOMM][SEND][OBJECT][PRODUCER] valueLength: {}, readCount: {}, chunkCount: {}, transferMetaId: {}",
                    valueLength, readCount, chunkCount, transferPojoUtils.generateTransferId(transferMeta));

            result = returnStatusFactory.createSucessful("redis sendProducer Done. total size: " + readCount + ", total chunks: " + chunkCount);

            transferBroker.setFinished();
        } catch (Exception e) {
            LOGGER.error(errorUtils.getStackTrace(e));
            throw e;
        }

        return result;
    }
}
