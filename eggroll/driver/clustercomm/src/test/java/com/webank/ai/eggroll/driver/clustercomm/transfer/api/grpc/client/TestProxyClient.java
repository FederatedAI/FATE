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

package com.webank.ai.eggroll.driver.clustercomm.transfer.api.grpc.client;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:applicationContext-clustercomm.xml"})
public class TestProxyClient {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private ProxyClient proxyClient;
    private ClusterComm.TransferMeta defaultTransferMeta;
    private String name = "api_create_name";
    private String namespace = "api_create_namespace";
    private String tag = "33";

    public TestProxyClient() {
        StorageBasic.StorageLocator storageLocator = StorageBasic.StorageLocator.newBuilder()
                .setNamespace(namespace)
                .setName(name)
                .setType(StorageBasic.StorageType.LMDB)
                .build();
        ClusterComm.TransferDataDesc dataDesc = ClusterComm.TransferDataDesc.newBuilder()
                .setStorageLocator(storageLocator)
                .setTransferDataType(ClusterComm.TransferDataType.DTABLE)
                .build();

        ClusterComm.Party.Builder partyBuilder = ClusterComm.Party.newBuilder();

        BasicMeta.Job.Builder jobBuilder = BasicMeta.Job.newBuilder();

        defaultTransferMeta = ClusterComm.TransferMeta.newBuilder()
                .setDataDesc(dataDesc)
                .setSrc(partyBuilder.setPartyId("10000").setName("wb1").build())
                .setDst(partyBuilder.setPartyId("9999").setName("wb2").build())
                .setJob(jobBuilder.setJobId("jobid1").setName("jobName1").build())
                .setType(ClusterComm.TransferType.SEND)
                .setTag(tag)
                .build();
    }

    @Test
    public void testSendStartRequest() {
        proxyClient.requestSendStart(defaultTransferMeta, RuntimeConstants.getLocalEndpoint(8890));

        LOGGER.info("started");
    }

    @Test
    public void testSendEndRequest() {
        proxyClient.requestSendEnd(defaultTransferMeta, RuntimeConstants.getLocalEndpoint(8890));

        LOGGER.info("ended");
    }
}
