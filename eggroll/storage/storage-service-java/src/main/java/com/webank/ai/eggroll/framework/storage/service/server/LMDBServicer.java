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

package com.webank.ai.eggroll.framework.storage.service.server;


import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.storage.KVServiceGrpc;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.api.grpc.server.GrpcServerWrapper;
import com.webank.ai.eggroll.core.constant.MetaConstants;
import com.webank.ai.eggroll.core.io.KeyValue;
import com.webank.ai.eggroll.core.io.KeyValueIterator;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.io.StoreManager;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.core.serdes.impl.POJOUtils;
import com.webank.ai.eggroll.framework.storage.service.model.LMDBStore;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


public class LMDBServicer extends KVServiceGrpc.KVServiceImplBase {
    private static final long PAYLOAD_THRESHOLD = 2L * 1024 * 1024;
    private static Logger LOGGER = LogManager.getLogger(LMDBServicer.class);
    private final StoreManager<Bytes, byte[]> storeMgr;
    private GrpcServerWrapper grpcServerWrapper;

    public LMDBServicer(StoreManager<Bytes, byte[]> storeMgr) {
        this.storeMgr = storeMgr;
        this.grpcServerWrapper = new GrpcServerWrapper();
    }

    private LMDBStore getStore() {
        StoreInfo info = StoreInfo.fromGrpcContext();
        return (LMDBStore) storeMgr.createIfMissing(info);
    }

    @Override
    public void put(Kv.Operand request, StreamObserver<Kv.Empty> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            LMDBStore store = getStore();
            LOGGER.info("{} receive put request. keyLength: {}, valueLength: {}",
                    store, request.getKey().size(), request.getValue().size());
            store.put(Bytes.wrap(request.getKey()), request.getValue().toByteArray());
            responseObserver.onNext(Kv.Empty.newBuilder().build());
            responseObserver.onCompleted();
        });
    }

    @Override
    public void putIfAbsent(Kv.Operand request, StreamObserver<Kv.Operand> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            LMDBStore store = getStore();
            byte[] oldValue = store.putIfAbsent(Bytes.wrap(request.getKey()), request.getValue().toByteArray());
            Kv.Operand.Builder builder = Kv.Operand.newBuilder().setKey(request.getKey());
            if (oldValue != null) {
                builder.setValue(ByteString.copyFrom(oldValue));
            }
            responseObserver.onNext(builder.buildPartial());
            responseObserver.onCompleted();
        });
    }

    @Override
    public StreamObserver<Kv.Operand> putAll(StreamObserver<Kv.Empty> responseObserver) {
        LMDBStore store = getStore();
        LOGGER.info("putAll request received. store: {}", store);
        StreamObserver<KeyValue<Bytes, byte[]>> putObs = store.putAll();

        return new StreamObserver<Kv.Operand>() {
            long count = 0;

            @Override
            public void onNext(Kv.Operand operand) {
                putObs.onNext(POJOUtils.buildKeyValue(operand));
                count++;
            }

            @Override
            public void onError(Throwable throwable) {
                LOGGER.error(throwable.getMessage(), throwable);
                putObs.onError(throwable);
                LOGGER.info(String.format("%s put all %d entries before error", store.toString(), count));
            }

            @Override
            public void onCompleted() {
                putObs.onCompleted();
                responseObserver.onNext(Kv.Empty.newBuilder().build());
                responseObserver.onCompleted();
                LOGGER.info(String.format("%s put all %d entries", store.toString(), count));
            }
        };
    }

    @Override
    public void delOne(Kv.Operand request, StreamObserver<Kv.Operand> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            LMDBStore store = getStore();

            byte[] valueBuffer = store.delete(Bytes.wrap(request.getKey()));
            Kv.Operand.Builder builder = Kv.Operand.newBuilder().setKey(request.getKey());
            if (valueBuffer != null) {
                builder.setValue(ByteString.copyFrom(valueBuffer));
            }
            responseObserver.onNext(builder.buildPartial());
            responseObserver.onCompleted();
        });
    }

    @Override
    public void get(Kv.Operand request, StreamObserver<Kv.Operand> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            LMDBStore store = getStore();
            byte[] valueBuffer = store.get(Bytes.wrap(request.getKey()));
            Kv.Operand.Builder builder = Kv.Operand.newBuilder().setKey(request.getKey());
            if (valueBuffer != null)
                builder.setValue(ByteString.copyFrom(valueBuffer));
            responseObserver.onNext(builder.buildPartial());
            responseObserver.onCompleted();
        });
    }

    @Override
    public void iterate(Kv.Range request, StreamObserver<Kv.Operand> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            long bytesCount = 0;
            LMDBStore store = getStore();
            LOGGER.info("{} receive iterate request. start: {}, end: {}, minChunkSize: {}",
                    store, request.getStart().toStringUtf8(), request.getEnd().toStringUtf8(), request.getMinChunkSize());

            byte[] fromBytes = request.getStart().toByteArray();
            byte[] toBytes = request.getEnd().toByteArray();
            Bytes fromBuffer = null;
            Bytes toBuffer = null;
            long count = 0;
            if (fromBytes.length > 0) {
                fromBuffer = Bytes.wrap(fromBytes);
            }
            if (toBytes.length > 0) {
                toBuffer = Bytes.wrap(toBytes);
            }
            long threshold = request.getMinChunkSize() > 0 ? request.getMinChunkSize() : PAYLOAD_THRESHOLD;
            try (KeyValueIterator<Bytes, byte[]> keyValueIterator = store.range(fromBuffer, toBuffer)) {
                while (keyValueIterator.hasNext()) {
                    KeyValue<Bytes, byte[]> keyValue = keyValueIterator.next();
                    Kv.Operand operand = Kv.Operand.newBuilder().setKey(ByteString.copyFrom(keyValue.key.get()))
                            .setValue(ByteString.copyFrom(keyValue.value)).build();
                    responseObserver.onNext(operand);
                    ++count;
                    bytesCount += operand.getKey().size();
                    bytesCount += operand.getValue().size();
                    if (bytesCount >= threshold) {
                        break;
                    }
                }
            } finally {
                responseObserver.onCompleted();
                LOGGER.info("[STORAGE][ITERATE][RESULT]: {} store count: {}, onNext count: {}, start: '{}', end: '{}'",
                        store, store.count(), count, request.getStart().toStringUtf8(), request.getEnd().toStringUtf8());
            }
        });
    }

    @Override
    public void destroy(Kv.Empty request, StreamObserver<Kv.Empty> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            StoreInfo info = StoreInfo.fromGrpcContext();
            storeMgr.destroy(info);
            responseObserver.onNext(Kv.Empty.newBuilder().build());
            responseObserver.onCompleted();
        });
    }

    @Override
    public void count(Kv.Empty request, StreamObserver<Kv.Count> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            LMDBStore lmdbStore = getStore();
            responseObserver.onNext(Kv.Count.newBuilder().setValue(lmdbStore.count()).build());
            responseObserver.onCompleted();
        });
    }


    public static class KvStoreInterceptor implements ServerInterceptor {
        @Override
        public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> serverCall, Metadata metadata, ServerCallHandler<ReqT, RespT> serverCallHandler) {
            Context ctx = MetaConstants.updateContext(metadata, Context.current());
            return Contexts.interceptCall(ctx, serverCall, metadata, serverCallHandler);
        }
    }


}
