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

import com.webank.ai.eggroll.api.computing.processor.ProcessServiceGrpc;
import com.webank.ai.eggroll.api.computing.processor.Processor;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcAsyncClientContext;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcStreamingClientTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

@Component
public class RollProcessorServiceCallModelFactory {
    @Autowired
    private ApplicationContext applicationContext;

    @Autowired
    private GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, StorageBasic.StorageLocator> nonSpringUnaryProcessToDTableContext;
    @Autowired
    private GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.BinaryProcess, StorageBasic.StorageLocator> nonSpringBinaryProcessToDTableContext;
    @Autowired
    private GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, Kv.Operand> nonSpringUnaryProcessToOperandContext;
    @Autowired
    private GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, StorageBasic.StorageLocator> nonSpringUnaryProcessToDTableTemplate;
    @Autowired
    private GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.BinaryProcess, StorageBasic.StorageLocator> nonSpringBinaryProcessToDTableTemplate;
    @Autowired
    private GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, Kv.Operand> nonSpringUnaryProcessToOperandTemplate;

    public GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, StorageBasic.StorageLocator>
    createUnaryProcessToStorageLocatorContext() {
        GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, StorageBasic.StorageLocator> result
                = applicationContext.getBean(nonSpringUnaryProcessToDTableContext.getClass());
        result.setStubClass(ProcessServiceGrpc.ProcessServiceStub.class);
        return result;
    }

    public GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.BinaryProcess, StorageBasic.StorageLocator>
    createBinaryProcessToStorageLocatorContext() {
        GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.BinaryProcess, StorageBasic.StorageLocator> result
                = applicationContext.getBean(nonSpringBinaryProcessToDTableContext.getClass());
        result.setStubClass(ProcessServiceGrpc.ProcessServiceStub.class);
        return result;
    }

    public GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, Kv.Operand>
    createUnaryProcessToOperandContext() {
        GrpcAsyncClientContext<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, Kv.Operand> result
                = applicationContext.getBean(nonSpringUnaryProcessToOperandContext.getClass());
        result.setStubClass(ProcessServiceGrpc.ProcessServiceStub.class);
        return result;
    }

    public GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, StorageBasic.StorageLocator>
    createUnaryProcessToStorageLocatorTemplate() {
        GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, StorageBasic.StorageLocator> result
                = applicationContext.getBean(nonSpringUnaryProcessToDTableTemplate.getClass());
        return result;
    }

    public GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.BinaryProcess, StorageBasic.StorageLocator>
    createBinaryProcessToStorageLocatorTemplate() {
        GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.BinaryProcess, StorageBasic.StorageLocator> result
                = applicationContext.getBean(nonSpringBinaryProcessToDTableTemplate.getClass());
        return result;
    }

    public GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, Kv.Operand>
    createUnaryProcessToOperandTemplate() {
        GrpcStreamingClientTemplate<ProcessServiceGrpc.ProcessServiceStub, Processor.UnaryProcess, Kv.Operand> result
                = applicationContext.getBean(nonSpringUnaryProcessToOperandTemplate.getClass());
        return result;
    }
}
