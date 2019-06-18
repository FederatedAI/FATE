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
import com.webank.ai.eggroll.core.constant.MetaConstants;
import com.webank.ai.eggroll.core.io.*;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.core.serdes.impl.POJOUtils;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;


public class ObjectStoreServicer extends KVServiceGrpc.KVServiceImplBase {
    private static final long PAYLOAD_THRESHOLD = 2L * 1024 * 1024;
    private static final long NUM_THRESHOLD = 2L;
    private static Logger LOGGER = LogManager.getLogger(ObjectStoreServicer.class);
    private final StoreManager storeMgr;


    public ObjectStoreServicer(StoreManager storeMgr) {
        this.storeMgr = storeMgr;
    }

    private KeyValueStore<Bytes, byte[]> getStore() {
        StoreInfo info = StoreInfo.fromGrpcContext();
        return storeMgr.createIfMissing(info);
    }

    @Override
    public void put(Kv.Operand request, StreamObserver<Kv.Empty> responseObserver) {
        KeyValueStore<Bytes, byte[]> store = getStore();
        store.put(Bytes.wrap(request.getKey().toByteArray()), request.getValue().toByteArray());
        responseObserver.onNext(Kv.Empty.newBuilder().build());
        responseObserver.onCompleted();
    }

    @Override
    public StreamObserver<Kv.Operand> putAll(StreamObserver<Kv.Empty> responseObserver) {
        return new StreamObserver<Kv.Operand>() {
            List<KeyValue<Bytes, byte[]>> toPut = new ArrayList<>();
            KeyValueStore<Bytes, byte[]> store = getStore();

            @Override
            public void onNext(Kv.Operand operand) {
                toPut.add(POJOUtils.buildKeyValue(operand));
            }

            @Override
            public void onError(Throwable throwable) {
                LOGGER.error(throwable.getMessage(), throwable);
            }

            @Override
            public void onCompleted() {
                store.putAll(toPut);
                responseObserver.onNext(Kv.Empty.newBuilder().build());
                responseObserver.onCompleted();
            }
        };
    }

    /**
     *
     */
    public void putIfAbsent(Kv.Operand request,
                            StreamObserver<Kv.Operand> responseObserver) {
        KeyValueStore<Bytes, byte[]> store = getStore();
        byte[] oldValue = store.putIfAbsent(Bytes.wrap(request.getKey().toByteArray()), request.getValue().toByteArray());
        Kv.Operand.Builder builder = Kv.Operand.newBuilder().setKey(request.getKey());
        if (oldValue != null) {
            builder.setValue(ByteString.copyFrom(oldValue));
        }
        responseObserver.onNext(builder.buildPartial());
        responseObserver.onCompleted();
    }

    @Override
    public void delOne(Kv.Operand request,
                       StreamObserver<Kv.Operand> responseObserver) {
        KeyValueStore<Bytes, byte[]> store = getStore();
        byte[] oldValue = store.delete(Bytes.wrap(request.getKey().toByteArray()));
        Kv.Operand.Builder builder = Kv.Operand.newBuilder().setKey(request.getKey());
        if (oldValue != null) {
            builder.setValue(ByteString.copyFrom(oldValue));
        }
        responseObserver.onNext(builder.buildPartial());
        responseObserver.onCompleted();
    }

    @Override
    public void get(Kv.Operand request, StreamObserver<Kv.Operand> responseObserver) {
        KeyValueStore<Bytes, byte[]> store = getStore();
        byte[] value = store.get(Bytes.wrap(request.getKey().toByteArray()));

        Kv.Operand.Builder builder = Kv.Operand.newBuilder().setKey(request.getKey());
        if (value != null)
            builder.setValue(ByteString.copyFrom(value));
        responseObserver.onNext(builder.buildPartial());
        responseObserver.onCompleted();
    }

    @Override
    public void iterate(Kv.Range request, StreamObserver<Kv.Operand> responseObserver) {
        int bytesCount = 0;
        int numEntries = 0;
        byte[] fromBytes = request.getStart().toByteArray();
        byte[] toBytes = request.getEnd().toByteArray();
        Bytes from = fromBytes.length > 0 ? Bytes.wrap(fromBytes) : null;
        Bytes to = toBytes.length > 0 ? Bytes.wrap(toBytes) : null;
        KeyValueStore<Bytes, byte[]> store = getStore();
        KeyValueIterator<Bytes, byte[]> keyValueIterator = store.range(from, to);
        while (keyValueIterator.hasNext()) {
            KeyValue<Bytes, byte[]> keyValue = keyValueIterator.next();
            responseObserver.onNext(POJOUtils.buildOperand(keyValue));
            bytesCount += keyValue.key.get().length;
            bytesCount += keyValue.value.length;
            numEntries++;
            if (bytesCount >= PAYLOAD_THRESHOLD && numEntries >= NUM_THRESHOLD) {
                break;
            }
        }
        keyValueIterator.close();
        responseObserver.onCompleted();
    }

    @Override
    public void destroy(Kv.Empty request, StreamObserver<Kv.Empty> responseObserver) {
        StoreInfo info = StoreInfo.fromGrpcContext();
        storeMgr.destroy(info);
        responseObserver.onNext(Kv.Empty.newBuilder().build());
        responseObserver.onCompleted();
    }


    public static class KvStoreInterceptor implements ServerInterceptor {
        @Override
        public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> serverCall, Metadata metadata, ServerCallHandler<ReqT, RespT> serverCallHandler) {
            Context ctx = MetaConstants.updateContext(metadata, Context.current());
            return Contexts.interceptCall(ctx, serverCall, metadata, serverCallHandler);
        }
    }
}
