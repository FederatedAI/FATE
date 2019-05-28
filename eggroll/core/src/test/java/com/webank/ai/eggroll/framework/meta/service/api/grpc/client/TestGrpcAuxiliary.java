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

package com.webank.ai.eggroll.framework.meta.service.api.grpc.client.client;


import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.framework.meta.service.StorageMetaServiceGrpc;
import com.webank.ai.eggroll.core.factory.GrpcChannelFactory;
import com.webank.ai.eggroll.core.factory.GrpcStubFactory;
import io.grpc.ManagedChannel;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:applicationContext-core.xml"})
public class TestGrpcAuxiliary {
    @Autowired
    private GrpcChannelFactory grpcChannelFactory;

    @Autowired
    private GrpcStubFactory grpcStubFactory;

    private ManagedChannel managedChannel = null;

    @Test
    public void testGrpcChannel() {
        BasicMeta.Endpoint endpoint = BasicMeta.Endpoint.newBuilder().setIp("127.0.0.1").setPort(8888).build();
        managedChannel = grpcChannelFactory.getChannel(endpoint, false);
        System.out.print(managedChannel);
    }

    @Test
    public void testGrpcStub() {
        testGrpcChannel();

        StorageMetaServiceGrpc.StorageMetaServiceStub stub
                = (StorageMetaServiceGrpc.StorageMetaServiceStub) grpcStubFactory
                .createGrpcStub(true, StorageMetaServiceGrpc.class, managedChannel);

        System.out.println(stub);
    }
}
