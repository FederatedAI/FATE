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

package com.webank.ai.fate.core.mlmodel.model;


import java.util.Map;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.core.network.grpc.client.ClientPool;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.core.utils.ObjectTransform;
import io.grpc.ManagedChannel;
import com.webank.ai.fate.api.networking.proxy.Proxy.Packet;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;

public abstract class MachineLearningModel<B, X, P> implements MLModel<B, X, P>{
    private Map<String, String> modelInfo;

    @Override
    public void setModelInfo(Map<String, String> modelInfo) {
        this.modelInfo = modelInfo;
    }

    @Override
    public Map<String, String> getModelInfo() {
        return this.modelInfo;
    }

    @Override
    public abstract int initModel(B modelBuffer);

    @Override
    public abstract Map<String, Object> predict(X inputData, P predictParams);

    protected Map<String, Object> getFederatedPredict(Map<String, Object> requestData){
        Packet.Builder packetBuilder = Packet.newBuilder();
        requestData.putAll(this.modelInfo);
        requestData.put("myPartyId", Configuration.getProperty("partyId"));
        packetBuilder.setBody(Proxy.Data.newBuilder()
                        .setValue(ByteString.copyFrom(ObjectTransform.bean2Json(requestData).getBytes()))
                        .build());

        Proxy.Metadata.Builder metaDataBuilder = Proxy.Metadata.newBuilder();
        Proxy.Topic.Builder topicBuilder = Proxy.Topic.newBuilder();

        metaDataBuilder.setSrc(
                topicBuilder.setPartyId(Configuration.getProperty("partyId")).
                        setRole(this.modelInfo.get("myRole"))
                        .setName("partyName")
                        .build());
        metaDataBuilder.setDst(
                topicBuilder.setPartyId(this.modelInfo.get("partnerPartyId"))
                        .setRole("host")
                        .setName("partnerPartyName")
                        .build());
        metaDataBuilder.setCommand(Proxy.Command.newBuilder().setName("federatedPredict").build());
        packetBuilder.setHeader(metaDataBuilder.build());

        ManagedChannel channel1 = ClientPool.getChannel(Configuration.getProperty("proxy"));
        DataTransferServiceGrpc.DataTransferServiceBlockingStub stub1 = DataTransferServiceGrpc.newBlockingStub(channel1);
        Packet packet = stub1.unaryCall(packetBuilder.build());

        ReturnResult result = (ReturnResult) ObjectTransform.json2Bean(packet.getBody().getValue().toStringUtf8(), ReturnResult.class);
        return result.getData();
    }
}
