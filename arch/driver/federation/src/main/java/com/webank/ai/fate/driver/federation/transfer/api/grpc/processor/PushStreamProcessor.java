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

package com.webank.ai.fate.driver.federation.transfer.api.grpc.processor;

import com.google.common.collect.Lists;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.core.api.grpc.client.crud.BaseStreamProcessor;
import com.webank.ai.fate.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferPojoUtils;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferProtoMessageUtils;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.List;

@Component
@Scope("prototype")
public class PushStreamProcessor extends BaseStreamProcessor<Proxy.Packet> {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private TransferPojoUtils transferPojoUtils;
    @Autowired
    private TransferProtoMessageUtils transferProtoMessageUtils;
    @Autowired
    private ToStringUtils toStringUtils;
    private Proxy.Packet.Builder packetBuilder;
    private Proxy.Data.Builder bodyBuilder;
    private Proxy.Metadata.Builder headerBuilder;
    private long seq;
    private TransferBroker transferBroker;
    private Federation.TransferMeta transferMeta;

    public PushStreamProcessor(StreamObserver<Proxy.Packet> streamObserver, TransferBroker transferBroker) {
        super(streamObserver);
        this.transferBroker = transferBroker;
        this.transferMeta = transferBroker.getTransferMeta();
        this.seq = 0;

        this.packetBuilder = Proxy.Packet.newBuilder();
        this.headerBuilder = Proxy.Metadata.newBuilder();
        this.bodyBuilder = Proxy.Data.newBuilder();
    }

    @PostConstruct
    public void init() {
        headerBuilder
                .setTask(Proxy.Task.newBuilder().setTaskId(transferPojoUtils.generateTransferId(transferMeta)))
                .setSrc(transferProtoMessageUtils.partyToTopic(transferMeta.getSrc()))
                .setDst(transferProtoMessageUtils.partyToTopic(transferMeta.getDst()));
    }

    @Override
    public void process() {
        // LOGGER.info("processing send stream for task: {}", toStringUtils.toOneLineString(transferMeta));
        List<ByteString> dataList = Lists.newLinkedList();

        transferBroker.drainTo(dataList);

        Proxy.Packet packet = null;
        for (ByteString data : dataList) {
            headerBuilder.setSeq(++seq);
            bodyBuilder.setValue(data);

            packet = packetBuilder.setHeader(headerBuilder)
                    .setBody(bodyBuilder)
                    .build();
            streamObserver.onNext(packet);
        }
    }

    @Override
    public void complete() {
        LOGGER.info("[FEDERATION][PUSHPROCESSOR] completing send stream for task: {}, transferBroker remaining: {}",
                toStringUtils.toOneLineString(transferMeta), transferBroker.getQueueSize());
        // transferBroker.setFinished();
        super.complete();
    }
}
