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

package com.webank.ai.fate.eggroll.storage.service.model;

import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.eggroll.storage.KVServiceGrpc;
import com.webank.ai.fate.api.eggroll.storage.Kv;
import com.webank.ai.fate.core.constant.MetaConstants;
import com.webank.ai.fate.core.constant.ModelConstants;
import com.webank.ai.fate.core.error.exception.InvalidStateStoreException;
import com.webank.ai.fate.core.factory.GrpcChannelFactory;
import com.webank.ai.fate.core.io.KeyValue;
import com.webank.ai.fate.core.io.KeyValueIterator;
import com.webank.ai.fate.core.io.KeyValueStore;
import com.webank.ai.fate.core.io.StoreInfo;
import com.webank.ai.fate.core.model.Bytes;
import com.webank.ai.fate.core.serdes.impl.POJOUtils;
import com.webank.ai.fate.core.utils.AbstractIterator;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.stub.MetadataUtils;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;


public class RemoteKeyValueStore implements KeyValueStore<Bytes, byte[]> {
    private static final Logger LOGGER = LogManager.getLogger(RemoteKeyValueStore.class);
    private final StoreInfo storeInfo;
    @Autowired
    GrpcChannelFactory channelFactory;
    private volatile ManagedChannel channel;
    private volatile KVServiceGrpc.KVServiceBlockingStub blockingStub;
    private volatile KVServiceGrpc.KVServiceStub stub;
    // TODO temporary configs
    private int maxWaitMins = 0;
    private int maxMessageSize = 16 * 1024 * 1024;

    public RemoteKeyValueStore(StoreInfo storeInfo) {
        this.storeInfo = storeInfo;
    }


    @Override
    public void put(Bytes key, byte[] value) {
        Objects.requireNonNull(key, "key cannot be null");
        if (value == null) {
            blockingStub.delOne(POJOUtils.buildOperand(key, new byte[0]));
        } else {
            blockingStub.put(POJOUtils.buildOperand(key, value));
        }
    }

    @Override
    public byte[] putIfAbsent(Bytes key, byte[] value) {
        Objects.requireNonNull(key, "key cannot be null");
        byte[] rtn;
        if (value == null) {
            rtn = blockingStub.delOne(POJOUtils.buildOperand(key, new byte[0])).getValue().toByteArray();
        } else {
            rtn = blockingStub.putIfAbsent(POJOUtils.buildOperand(key, value)).getValue().toByteArray();
        }
        if (null == rtn || rtn.length == 0) {
            return null;
        }
        return rtn;
    }

    @Override
    public void putAll(List<KeyValue<Bytes, byte[]>> entries) {
        final CountDownLatch finishLatch = new CountDownLatch(1);

        List<Kv.Operand> putList = new ArrayList<>(entries.size());

        for (KeyValue<Bytes, byte[]> entry : entries) {
            Objects.requireNonNull(entry.key, "key cannot be null");
            putList.add(POJOUtils.buildOperand(entry));
        }

        StreamObserver<Kv.Operand> requestObs = stub.putAll(new StreamObserver<Kv.Empty>() {
            @Override
            public void onNext(Kv.Empty empty) {

            }

            @Override
            public void onError(Throwable throwable) {
                LOGGER.error(throwable.getMessage(), throwable);
                finishLatch.countDown();
            }

            @Override
            public void onCompleted() {
                finishLatch.countDown();
            }
        });
        for (Kv.Operand operand : putList) {
            requestObs.onNext(operand);
        }
        requestObs.onCompleted();
        try {
            if (maxWaitMins > 0 && !finishLatch.await(maxWaitMins, TimeUnit.MINUTES)) {
                LOGGER.error("putAll cannot finished");
            } else {
                finishLatch.await();
            }
        } catch (InterruptedException e) {
            LOGGER.error(e.getMessage(), e);
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public StreamObserver<KeyValue<Bytes, byte[]>> putAll() {
        throw new NotImplementedException("RemoteKeyValueStore doesn't support putAll without parameters");
    }

    @Override
    public byte[] delete(Bytes key) {
        Objects.requireNonNull(key, "key cannot be null");
        byte[] rtn = blockingStub.delOne(POJOUtils.buildOperand(key, new byte[0])).getValue().toByteArray();
        if (null == rtn || rtn.length == 0) {
            return null;
        }
        return rtn;
    }

    @Override
    public byte[] get(Bytes key) {
        Objects.requireNonNull(key, "key cannot be null");
        byte[] rtn = blockingStub.get(POJOUtils.buildOperand(key, new byte[0])).getValue().toByteArray();
        if (null == rtn || rtn.length == 0) {
            return null;
        }
        return rtn;
    }

    @Override
    public KeyValueIterator<Bytes, byte[]> range(Bytes from, Bytes to) {
        return new WrappedRangedRemoteIterator(from, to);
    }

    @Override
    public KeyValueIterator<Bytes, byte[]> all() {
        return new WrappedRemoteIterator();
    }


    @Override
    public synchronized void destroy() {
        blockingStub.destroy(Kv.Empty.newBuilder().build());
        close();
        this.channel = null;
        this.blockingStub = null;
        this.stub = null;
    }

    @Override
    public long count() {
        Kv.Count result = blockingStub.count(ModelConstants.EMPTY);

        return result.getValue();
    }

    @Override
    public String name() {
        return storeInfo.getTableName();
    }

    @Override
    public synchronized void init(Properties properties) {
        final Metadata metadata = new Metadata();
        metadata.put(MetaConstants.STORE_TYPE.asMetaKey(), storeInfo.getType());
        metadata.put(MetaConstants.TABLE_NAME.asMetaKey(), storeInfo.getTableName());
        metadata.put(MetaConstants.NAME_SPACE.asMetaKey(), storeInfo.getNameSpace());
        metadata.put(MetaConstants.FRAGMENT.asMetaKey(), storeInfo.getFragment() + "");
        String host = properties.getProperty("host");
        int port = -1;
        Object portObj = properties.get("port");

        if (portObj instanceof String) {
            port = Integer.valueOf(portObj.toString());
        } else {
            port = (int) properties.get("port");
        }
        channel = NettyChannelBuilder.forAddress(host, port).usePlaintext().maxInboundMessageSize(maxMessageSize).build();
        blockingStub = MetadataUtils.attachHeaders(KVServiceGrpc.newBlockingStub(channel), metadata);
        stub = MetadataUtils.attachHeaders(KVServiceGrpc.newStub(channel), metadata);
    }


    @Override
    public void flush() {

    }

    @Override
    public void close() {
        channel.shutdown();
    }

    @Override
    public boolean persistent() {
        return false;
    }

    @Override
    public boolean isOpen() {
        return channel != null && !channel.isShutdown();
    }

    private class WrappedRemoteIterator extends AbstractIterator<KeyValue<Bytes, byte[]>> implements KeyValueIterator<Bytes, byte[]> {

        Iterator<Kv.Operand> iter;
        KeyValue<Bytes, byte[]> next;
        private volatile boolean open = true;


        public WrappedRemoteIterator() {

        }

        @Override
        public void close() {
            this.open = false;
        }

        @Override
        public Bytes peekNextKey() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return next.key;
        }

        @Override
        public synchronized boolean hasNext() {
            if (!open) {
                throw new InvalidStateStoreException(String.format("Remote store %s has closed", storeInfo.toString()));
            }
            return super.hasNext();
        }

        @Override
        protected KeyValue<Bytes, byte[]> makeNext() {
            if (iter == null || !iter.hasNext()) {
                if (next == null) {
                    iter = blockingStub.iterate(Kv.Range.newBuilder().buildPartial());
                } else {
                    iter = blockingStub.iterate(Kv.Range.newBuilder().setStart(ByteString.copyFrom(next.key.get())).buildPartial());
                }
                if (!iter.hasNext()) {
                    return allDone();
                }
            }
            next = POJOUtils.buildKeyValue(iter.next());
            return next;
        }
    }

    private class WrappedRangedRemoteIterator extends WrappedRemoteIterator {
        final Bytes from;
        final Bytes to;

        public WrappedRangedRemoteIterator(final Bytes from, final Bytes to) {
            this.from = from;
            this.to = to;
        }

        @Override
        protected KeyValue<Bytes, byte[]> makeNext() {
            if (iter == null || !iter.hasNext()) {
                Kv.Range.Builder rangeBuilder = Kv.Range.newBuilder();
                if (next != null) {
                    rangeBuilder.setStart(ByteString.copyFrom(next.key.get()));
                } else if (from != null) {
                    rangeBuilder.setStart(ByteString.copyFrom(from.get()));
                }
                iter = blockingStub.iterate(rangeBuilder.buildPartial());
                if (!iter.hasNext()) {
                    return allDone();
                }
            }
            next = POJOUtils.buildKeyValue(iter.next());

            if (to == null || next.key.compareTo(to) < 0) {
                return next;
            }

            return allDone();
        }
    }
}
