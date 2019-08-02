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

package com.webank.ai.fate.serving.service;

import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.api.networking.proxy.Proxy.Packet;
import com.webank.ai.fate.core.bean.FederatedParty;
import com.webank.ai.fate.core.bean.ReturnResult;
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.core.bean.BaseContext;
import com.webank.ai.fate.serving.core.bean.Context;
import com.webank.ai.fate.serving.core.bean.Dict;
import com.webank.ai.fate.serving.core.bean.BaseContext;
import com.webank.ai.fate.serving.core.monitor.WatchDog;
import com.webank.ai.fate.serving.manger.InferenceManager;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

public class ProxyService extends DataTransferServiceGrpc.DataTransferServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void unaryCall(Proxy.Packet req, StreamObserver<Proxy.Packet> responseObserver) {
        ReturnResult responseResult=null;
        Context context = new BaseContext();
        context.preProcess();
        WatchDog.enter(context);
        Map<String, Object> requestData=null;

        try {
            requestData = (Map<String, Object>) ObjectTransform.json2Bean(req.getBody().getValue().toStringUtf8(), HashMap.class);
            context.setCaseId(requestData.get(Dict.CASEID)!=null?requestData.get(Dict.CASEID).toString():Dict.NONE);
            switch (req.getHeader().getCommand().getName()) {
                case "federatedInference":
                    responseResult = InferenceManager.federatedInference(context, requestData);
                    break;
                default:
                    responseResult = new ReturnResult();
                    responseResult.setRetcode(StatusCode.PARAMERROR);
                    break;
            }

            Packet.Builder packetBuilder = Packet.newBuilder();
            packetBuilder.setBody(Proxy.Data.newBuilder()
                    .setValue(ByteString.copyFrom(ObjectTransform.bean2Json(responseResult).getBytes()))
                    .build());

            Proxy.Metadata.Builder metaDataBuilder = Proxy.Metadata.newBuilder();
            Proxy.Topic.Builder topicBuilder = Proxy.Topic.newBuilder();
            FederatedParty partnerParty = (FederatedParty) ObjectTransform.json2Bean(requestData.get("partner_local").toString(), FederatedParty.class);
            FederatedParty party = (FederatedParty) ObjectTransform.json2Bean(requestData.get("local").toString(), FederatedParty.class);

            metaDataBuilder.setSrc(
                    topicBuilder.setPartyId(String.valueOf(party.getPartyId()))
                            .setRole("host")
                            .setName("myPartyName")
                            .build());
            metaDataBuilder.setDst(
                    topicBuilder.setPartyId(String.valueOf(partnerParty.getPartyId()))
                            .setRole("guest")
                            .setName("partnerPartyName")
                            .build());
            packetBuilder.setHeader(metaDataBuilder.build());
            responseObserver.onNext(packetBuilder.build());
            responseObserver.onCompleted();
        }finally {
            WatchDog.quit(context);
            context.postProcess(requestData,responseResult);

        }
    }
}