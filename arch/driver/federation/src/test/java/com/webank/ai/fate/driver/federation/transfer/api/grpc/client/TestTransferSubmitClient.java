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

package com.webank.ai.fate.driver.federation.transfer.api.grpc.client;

import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.utils.FederationServerUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:applicationContext-federation.xml"})
public class TestTransferSubmitClient {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private TransferSubmitClient transferSubmitClient;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private FederationServerUtils federationServerUtils;
    private String name = "api_create_name";
    private String nameRecv = name + "_recv";
    private String namespace = "api_create_namespace";
    private String tag = "30";
    private Federation.Party.Builder partyBuilder = Federation.Party.newBuilder();
    private BasicMeta.Job.Builder jobBuilder = BasicMeta.Job.newBuilder();
    private Federation.TransferDataDesc.Builder dataDescBuilder = Federation.TransferDataDesc.newBuilder();
    private StorageBasic.StorageLocator.Builder storageLocatorBuilder = StorageBasic.StorageLocator.newBuilder();

    public TestTransferSubmitClient() {
    }

    @Test
    public void testSendTableRequest() {
        StorageBasic.StorageLocator storageLocator = storageLocatorBuilder
                .setNamespace(namespace)
                .setName(name)
                .setFragment(2)
                .setType(StorageBasic.StorageType.LMDB)
                .build();

        Federation.TransferDataDesc dataDesc = dataDescBuilder
                .setStorageLocator(storageLocator)
                .setTransferDataType(Federation.TransferDataType.DTABLE)
                .setTaggedVariableName(ByteString.copyFromUtf8(nameRecv))
                .build();

        Federation.TransferMeta transferMeta = Federation.TransferMeta.newBuilder()
                .setDataDesc(dataDesc)
                .setSrc(partyBuilder.setPartyId("10000").setName("wb1").build())
                .setDst(partyBuilder.setPartyId("9999").setName("wb2").build())
                .setJob(jobBuilder.setJobId("jobid1").setName("jobName1").build())
                .setType(Federation.TransferType.SEND)
                .setTag(tag)
                .build();

        transferSubmitClient.send(transferMeta, RuntimeConstants.getLocalEndpoint(8889));
    }

    @Test
    public void testRecvTableRequest() {
        StorageBasic.StorageLocator storageLocator = storageLocatorBuilder
                .setNamespace(namespace)
                .setName(name)
                .setType(StorageBasic.StorageType.LMDB)
                .build();
        Federation.TransferDataDesc dataDesc = Federation.TransferDataDesc.newBuilder()
                .setStorageLocator(storageLocator)
                //.setTransferDataType(Federation.TransferDataType.DTABLE)
                .build();

        Federation.TransferMeta transferMeta = Federation.TransferMeta.newBuilder()
                //.setDataDesc(dataDesc)
                .setSrc(partyBuilder.setPartyId("10000").setName("wb1").build())
                .setDst(partyBuilder.setPartyId("9999").setName("wb2").build())
                .setJob(jobBuilder.setJobId("jobid1").setName("jobName1").build())
                .setType(Federation.TransferType.RECV)
                .setTag(tag)
                .build();

        transferSubmitClient.recv(transferMeta, RuntimeConstants.getLocalEndpoint(8890));
    }

    @Test
    public void testSendRecvTableRequest() {
        testSendTableRequest();
        testRecvTableRequest();
    }

    @Test
    public void testSendObjectRequest() {
        StorageBasic.StorageLocator storageLocator = storageLocatorBuilder
                .setNamespace(namespace)
                .setName(name)
                .setType(StorageBasic.StorageType.LMDB)
                .build();
        Federation.TransferDataDesc dataDesc = dataDescBuilder
                .setStorageLocator(storageLocator)
                .setTransferDataType(Federation.TransferDataType.OBJECT)
                .setTaggedVariableName(ByteString.copyFromUtf8("iter"))
                .build();

        Federation.TransferMeta transferMeta = Federation.TransferMeta.newBuilder()
                .setSrc(partyBuilder.setPartyId("10000").setName("wb1").build())
                .setDst(partyBuilder.setPartyId("10000").setName("wb2").build())
                .setJob(jobBuilder.setJobId("jobid1").setName("jobName1").build())
                .setDataDesc(dataDesc)
                .setType(Federation.TransferType.SEND)
                .setTag(tag)
                .build();

        transferSubmitClient.send(transferMeta, RuntimeConstants.getLocalEndpoint(8889));
    }

    @Test
    public void testRecvObjectRequest() {
        StorageBasic.StorageLocator storageLocator = storageLocatorBuilder
                .setNamespace(namespace)
                .setName(name + "_recv_object")
                .setType(StorageBasic.StorageType.LMDB)
                .build();
        Federation.TransferDataDesc dataDesc = Federation.TransferDataDesc.newBuilder()
                .setStorageLocator(storageLocator)
                //.setTransferDataType(Federation.TransferDataType.DTABLE)
                .build();

        Federation.TransferMeta transferMeta = Federation.TransferMeta.newBuilder()
                //.setDataDesc(dataDesc)
                .setSrc(partyBuilder.setPartyId("10000").setName("wb1").build())
                .setDst(partyBuilder.setPartyId("10000").setName("wb2").build())
                .setJob(jobBuilder.setJobId("jobid1").setName("jobName1").build())
                .setType(Federation.TransferType.RECV)
                .setTag(tag)
                .build();

        transferSubmitClient.recv(transferMeta, RuntimeConstants.getLocalEndpoint(8889));
    }

    @Test
    public void testCheckStatusNow() {
        StorageBasic.StorageLocator storageLocator = storageLocatorBuilder
                .setNamespace(namespace)
                .setName(name + "_recv_object")
                .setType(StorageBasic.StorageType.LMDB)
                .build();
        Federation.TransferDataDesc dataDesc = Federation.TransferDataDesc.newBuilder()
                .setStorageLocator(storageLocator)
                //.setTransferDataType(Federation.TransferDataType.DTABLE)
                .build();

        Federation.TransferMeta transferMeta = Federation.TransferMeta.newBuilder()
                //.setDataDesc(dataDesc)
                .setSrc(partyBuilder.setPartyId("10000").setName("wb1").build())
                .setDst(partyBuilder.setPartyId("9999").setName("wb2").build())
                .setJob(jobBuilder.setJobId("jobid1").setName("jobName1").build())
                // .setType(Federation.TransferType.RECV)
                .setTag(tag)
                .build();

        Federation.TransferMeta transferResult = transferSubmitClient.checkStatusNow(transferMeta, RuntimeConstants.getLocalEndpoint(8890));

        LOGGER.info("result: {}", transferResult);
        LOGGER.info("result in one line: {}", transferResult);
        LOGGER.info("status: {}", transferResult.getTransferStatus().name());
    }

    @Test
    public void testCheckStatus() {
        StorageBasic.StorageLocator storageLocator = storageLocatorBuilder
                .setNamespace(namespace)
                .setName(name + "_recv_object")
                .setType(StorageBasic.StorageType.LMDB)
                .build();
        Federation.TransferDataDesc dataDesc = Federation.TransferDataDesc.newBuilder()
                .setStorageLocator(storageLocator)
                //.setTransferDataType(Federation.TransferDataType.DTABLE)
                .build();

        Federation.TransferMeta transferMeta = Federation.TransferMeta.newBuilder()
                //.setDataDesc(dataDesc)
                .setSrc(partyBuilder.setPartyId("10000").setName("wb1").build())
                .setDst(partyBuilder.setPartyId("9999").setName("wb2").build())
                .setJob(jobBuilder.setJobId("jobid1").setName("jobName1").build())
                .setType(Federation.TransferType.RECV)
                .setTag(tag)
                .build();

        Federation.TransferMeta transferResult = transferSubmitClient.checkStatus(transferMeta, RuntimeConstants.getLocalEndpoint(8890));

        LOGGER.info("result: {}", transferResult);
        LOGGER.info("result in one line: {}", transferResult);
        LOGGER.info("status: {}", transferResult.getTransferStatus().name());
    }

    public void testSendTableRequestWithDifferentName() {
        StorageBasic.StorageLocator storageLocator = storageLocatorBuilder
                .setNamespace(namespace)
                .setName(nameRecv)
                .setType(StorageBasic.StorageType.LMDB)
                .build();

        Federation.TransferDataDesc dataDesc = dataDescBuilder
                .setStorageLocator(storageLocator)
                .setTransferDataType(Federation.TransferDataType.DTABLE)
                .build();

        Federation.TransferMeta transferMeta = Federation.TransferMeta.newBuilder()
                .setDataDesc(dataDesc)
                .setSrc(partyBuilder.setPartyId("10000").setName("wb1").build())
                .setDst(partyBuilder.setPartyId("9999").setName("wb2").build())
                .setJob(jobBuilder.setJobId("jobid1").setName("jobName1").build())
                .setType(Federation.TransferType.RECV)
                .setTag(tag)
                .build();

        transferSubmitClient.send(transferMeta, RuntimeConstants.getLocalEndpoint(8889));
    }
}
