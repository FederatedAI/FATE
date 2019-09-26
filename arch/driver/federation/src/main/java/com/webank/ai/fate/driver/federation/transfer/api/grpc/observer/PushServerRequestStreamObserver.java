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

import com.google.common.collect.Lists;
import com.google.common.collect.Queues;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.eggroll.core.api.grpc.observer.BaseCalleeRequestStreamObserver;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.factory.TransferServiceFactory;
import com.webank.ai.fate.driver.federation.transfer.manager.RecvBrokerManager;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferPojoUtils;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Queue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;


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
    private List<ByteString> receivedData;

    private volatile boolean inited = false;
    private Proxy.Metadata response;
    private AtomicLong packetCount = new AtomicLong(0L);
    private AtomicLong maxSeq = new AtomicLong(0L);
    private int resetInterval = 10000;
    private int resetCount = resetInterval;
    private String transferMetaId;

    // todo: implement this in framework
    private final AtomicBoolean wasReady;
    private final ServerCallStreamObserver<Proxy.Metadata> serverCallStreamObserver;

    private final Object receivedDataLock = new Object();

    public PushServerRequestStreamObserver(final StreamObserver<Proxy.Metadata> callerNotifier, final AtomicBoolean wasReady) {
        super(callerNotifier);
        this.serverCallStreamObserver = (ServerCallStreamObserver<Proxy.Metadata>) callerNotifier;
        this.wasReady = wasReady;
    }

    public synchronized void init(Proxy.Metadata metadata) {
        if (inited) {
            return;
        }

        transferMetaId = metadata.getTask().getTaskId();
        transferBroker = recvBrokerManager.getBroker(transferMetaId);
        if (transferBroker == null) {
            transferBroker = recvBrokerManager.createIfNotExists(transferMetaId);

/*            // 2nd try, in case there is a conflict create which results in null return
            if (transferBroker == null) {
                transferBroker = recvBrokerManager.createIfNotExists(transferMetaId);
            }*/
        }
        transferBroker = recvBrokerManager.getBroker(transferMetaId);
        LOGGER.info("[FEDERATION][SERVER][OBSERVER] broker: {}, transferMetaId: {}, broker capacity: {}", transferBroker, transferMetaId, transferBroker.getQueueCapacity());

        this.receivedData = Lists.newLinkedList();
        this.response = metadata;

        recvBrokerManager.createRecvTaskFromPassedInTransferMetaId(transferMetaId);

        this.inited = true;
    }

    @Override
    public void onNext(Proxy.Packet packet) {
        // LOGGER.info("[SEND][SERVER][OBSERVER][ONNEXT] header: {}", toStringUtils.toOneLineString(packet.getHeader()));
        if (!inited) {
            init(packet.getHeader());
        }

        // byte[] rawData = packet.getBody().getValue().toByteArray();
        ByteString value = packet.getBody().getValue();
        long seq = packet.getHeader().getSeq();
        long currentMaxSeq = maxSeq.get();
        if (seq > currentMaxSeq) {
            maxSeq.compareAndSet(currentMaxSeq, seq);
        }

        /*boolean result = transferBroker.add(value);
        if (!result) {
            synchronized (receivedDataLock) {
                receivedData.add(value);
            }
        }*/

        transferBroker.put(value);
        packetCount.incrementAndGet();

        if (serverCallStreamObserver.isReady()) {
            serverCallStreamObserver.request(1);
        } else {
            LOGGER.warn("[SEND][SERVER][FLOWCONTROL] not ready");
            wasReady.set(false);
        }
    }

    @Override
    public void onError(Throwable throwable) {
        LOGGER.info("[SEND][SERVER][OBSERVER] onError: {}", errorUtils.getStackTrace(throwable));
        transferBroker.setError(throwable);
        super.onError(throwable);
    }

    @Override
    public void onCompleted() {
        LOGGER.info("[SEND][SERVER][OBSERVER] trying to complete PushServerRequestStreamObserver: header: {}, transferBrokerRemaining: {}, total packetCount: {}, receivedData size: {}",
                toStringUtils.toOneLineString(response), transferBroker.getQueueSize(), packetCount, receivedData.size());

        int whileInterval = 300;
        int whileCount = 10;
        while (transferBroker.isReady() || packetCount.get() < maxSeq.get()) {
            if (--whileCount <= 0) {
                LOGGER.info("[SEND][SERVER][OBSERVER] still trying. transferMetaId: {}, isReady: {}, isFinished: {}, isClosable: {}, packetCount: {}, maxSeq: {}",
                        transferMetaId, transferBroker.isReady(), transferBroker.isFinished(), transferBroker.isClosable(), packetCount, maxSeq.get());
                whileCount = whileInterval;
            }
            try {
                transferBroker.awaitClose(1, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                LOGGER.info(errorUtils.getStackTrace(e));
                onError(e);
            }
        }

        int putCount = 0;
/*        synchronized (receivedDataLock) {
            for (ByteString bs : receivedData) {
                transferBroker.put(bs);
                ++putCount;
            }
        }*/
        LOGGER.info("[SEND][SERVER][OBSERVER] actual completes PushServerRequestStreamObserver: header: {}, transferMetaId: {}, transferBrokerRemaining: {}, total packetCount: {}, putCount: {}",
                toStringUtils.toOneLineString(response), transferMetaId, transferBroker.getQueueSize(), packetCount, putCount);

        // transferBroker.setFinished();
        callerNotifier.onNext(response);
        super.onCompleted();
    }
}
