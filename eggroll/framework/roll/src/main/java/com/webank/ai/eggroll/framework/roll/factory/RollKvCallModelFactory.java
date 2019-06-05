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

package com.webank.ai.eggroll.framework.roll.factory;

import com.webank.ai.eggroll.api.storage.KVServiceGrpc;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcAsyncClientContext;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcStreamingClientTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

@Component
public class RollKvCallModelFactory {
    @Autowired
    private ApplicationContext applicationContext;
    @Autowired
    private GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.CreateTableInfo, Kv.CreateTableInfo> nonSpringCreateTableContext;
    @Autowired
    private GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.CreateTableInfo, Kv.CreateTableInfo> nonSpringCreateTableTemplate;
    @Autowired
    private GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Empty> nonSpringOperandToEmptyContext;
    @Autowired
    private GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Empty> nonSpringOperandToEmptyTemplate;
    @Autowired
    private GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> nonSpringOperandToOperandContext;
    @Autowired
    private GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> nonSpringOperandToOperandTemplate;
    @Autowired
    private GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Range, Kv.Operand> nonSpringRangeToOperandContext;
    @Autowired
    private GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Range, Kv.Operand> nonSpringRangeToOperandTemplate;
    @Autowired
    private GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Empty> nonSpringEmptyToEmptyContext;
    @Autowired
    private GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Empty> nonSpringEmptyToEmptyTemplate;
    @Autowired
    private GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Count> nonSpringEmptyToCountContext;
    @Autowired
    private GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Count> nonSpringEmptyToCountTemplate;

    private Class kvServiceStubClass = KVServiceGrpc.KVServiceStub.class;

    public GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.CreateTableInfo, Kv.CreateTableInfo> createCreateTableContext() {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.CreateTableInfo, Kv.CreateTableInfo> result
                = applicationContext.getBean(nonSpringCreateTableContext.getClass());
        result.setStubClass(kvServiceStubClass);

        return result;
    }

    public GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.CreateTableInfo, Kv.CreateTableInfo> createCreateTableTemplate() {
        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.CreateTableInfo, Kv.CreateTableInfo> result
                = applicationContext.getBean(nonSpringCreateTableTemplate.getClass());

        return result;
    }

    public GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Empty> createOperandToEmptyContext() {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Empty> result
                = applicationContext.getBean(nonSpringOperandToEmptyContext.getClass());
        result.setStubClass(kvServiceStubClass);

        return result;
    }

    public GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Empty> createOperandToEmptyTemplate() {
        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Empty> result
                = applicationContext.getBean(nonSpringOperandToEmptyTemplate.getClass());

        return result;
    }

    public GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> createOperandToOperandContext() {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> result
                = applicationContext.getBean(nonSpringOperandToOperandContext.getClass());
        result.setStubClass(kvServiceStubClass);

        return result;
    }

    public GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> createOperandToOperandTemplate() {
        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Operand, Kv.Operand> result
                = applicationContext.getBean(nonSpringOperandToOperandTemplate.getClass());

        return result;
    }

    public GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Range, Kv.Operand> createRangeToOperandContext() {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Range, Kv.Operand> result
                = applicationContext.getBean(nonSpringRangeToOperandContext.getClass());
        result.setStubClass(kvServiceStubClass);

        return result;
    }

    public GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Range, Kv.Operand> createRangeToOperandTemplate() {
        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Range, Kv.Operand> result
                = applicationContext.getBean(nonSpringRangeToOperandTemplate.getClass());

        return result;
    }

    public GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Empty> createEmptyToEmptyContext() {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Empty> result
                = applicationContext.getBean(nonSpringEmptyToEmptyContext.getClass());
        result.setStubClass(kvServiceStubClass);

        return result;
    }

    public GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Empty> createEmptyToEmptyTemplate() {
        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Empty> result
                = applicationContext.getBean(nonSpringEmptyToEmptyTemplate.getClass());

        return result;
    }

    public GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Count> createEmptyToCountContext() {
        GrpcAsyncClientContext<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Count> result
                = applicationContext.getBean(nonSpringEmptyToCountContext.getClass());
        result.setStubClass(kvServiceStubClass);

        return result;
    }

    public GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Count> createEmptyToCountTemplate() {
        GrpcStreamingClientTemplate<KVServiceGrpc.KVServiceStub, Kv.Empty, Kv.Count> result
                = applicationContext.getBean(nonSpringEmptyToCountTemplate.getClass());

        return result;
    }
}
