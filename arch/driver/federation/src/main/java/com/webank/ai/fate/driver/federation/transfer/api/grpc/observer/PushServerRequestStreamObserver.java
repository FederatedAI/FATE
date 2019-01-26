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

package com.webank.ai.fate.driver.federation.transfer.api.grpc.observer;

import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.core.api.grpc.observer.BaseCalleeRequestStreamObserver;
import com.webank.ai.fate.core.utils.ErrorUtils;
import com.webank.ai.fate.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.factory.TransferServiceFactory;
import com.webank.ai.fate.driver.federation.transfer.manager.RecvBrokerManager;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferPojoUtils;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;


@Component
@Scope("prototype")
public class PushServerRequestStreamObserver extends BaseCalleeRequestStreamObserver<Proxy.Packet, Proxy.Metadata> {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private TransferPojoUtils transferPojoUtils;
    @Autowired
    private RecvBrokerManager recvBrokerManager;
    @Autowired
    private TransferServiceFactory transferServiceFactory;
    private TransferBroker transferBroker;
    private volatile boolean inited = false;
    private Proxy.Metadata response;
    private int packetCount = 0;

    public PushServerRequestStreamObserver(StreamObserver<Proxy.Metadata> callerNotifier) {
        super(callerNotifier);
    }


    public void init(Proxy.Metadata metadata) {
        if (inited) {
            return;
        }

        String transferMetaId = metadata.getTask().getTaskId();
        transferBroker = recvBrokerManager.getBroker(transferMetaId);
        if (transferBroker == null) {
            transferBroker = recvBrokerManager.createIfNotExists(transferMetaId);

/*            // 2nd try, in case there is a conflict create which results in null return
            if (transferBroker == null) {
                transferBroker = recvBrokerManager.createIfNotExists(transferMetaId);
            }*/
        }

        transferBroker = recvBrokerManager.getBroker(transferMetaId);
        LOGGER.info("[FEDERATION][SERVER][OBSERVER] broker: {}, transferMetaId: {}", transferBroker, transferMetaId);

        this.response = metadata;
        inited = true;
    }

    @Override
    public void onNext(Proxy.Packet packet) {
        // LOGGER.info("[SEND][SERVER][OBSERVER][ONNEXT] header: {}", toStringUtils.toOneLineString(packet.getHeader()));
        if (!inited) {
            init(packet.getHeader());
        }

        byte[] rawData = packet.getBody().getValue().toByteArray();

        transferBroker.put(rawData);

        ++packetCount;
    }

    @Override
    public void onError(Throwable throwable) {
        LOGGER.info("[SEND][SERVER][OBSERVER] onError: {}", errorUtils.getStackTrace(throwable));
        transferBroker.setError(throwable);
        super.onError(throwable);
    }

    @Override
    public void onCompleted() {
        LOGGER.info("[SEND][SERVER][OBSERVER] onComplete: header: {}, total packet count: {}",
                toStringUtils.toOneLineString(response), packetCount);
        // transferBroker.setFinished();
        callerNotifier.onNext(response);
        super.onCompleted();
    }
}
