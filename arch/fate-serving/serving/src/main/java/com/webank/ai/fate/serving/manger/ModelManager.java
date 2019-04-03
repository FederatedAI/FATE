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

package com.webank.ai.fate.serving.manger;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.core.mlmodel.model.MLModel;
import com.webank.ai.fate.core.network.grpc.client.ClientPool;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.serving.utils.FederatedUtils;
import io.grpc.ManagedChannel;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class ModelManager {
    private static ModelPool sceneModel;
    private static ModelCache modelCache;
    private static final Logger LOGGER = LogManager.getLogger();
    public ModelManager(){
        sceneModel = new ModelPool();
        modelCache = new ModelCache();
    }

    public static MLModel getModel(String sceneId, String partnerPartyId, String myRole){
        return sceneModel.get(ModelUtils.getOnlineModelKey(sceneId, partnerPartyId, myRole));
    }

    public static MLModel getModel(String sceneId, String partnerPartyId, String myRole, String commitId){
        return modelCache.get(ModelUtils.genModelKey(sceneId, partnerPartyId, myRole, commitId));
    }

    private static int pushModelIntoPool(String sceneId, String partnerPartyId, String myRole, String commitId, String tag, String branch) throws Exception{
        MLModel mlModel = ModelUtils.loadModel(sceneId, partnerPartyId, myRole, commitId, tag, branch);
        if (mlModel == null){
            return StatusCode.NOMODEL;
        }
        modelCache.put(ModelUtils.genModelKey(sceneId, partnerPartyId, myRole, commitId), mlModel);
        return StatusCode.OK;
    }

    public static ReturnResult publishLoadModel(String sceneId, String partnerPartyId, String myRole, String commitId, String tag, String branch){
        ReturnResult returnResult = new ReturnResult();
        returnResult.setStatusCode(StatusCode.OK);
        returnResult.setData("commitId", commitId);
        try{
            int localLoadStatus = pushModelIntoPool(sceneId, partnerPartyId, myRole, commitId, tag, branch);
            if (localLoadStatus != StatusCode.OK){
                returnResult.setStatusCode(localLoadStatus);
                return returnResult;
            }
            Map<String, Object> federatedLoadModelResult = requestFederatedLoadModel(sceneId, partnerPartyId, myRole, commitId, tag, branch);
            if ((int)federatedLoadModelResult.get("statusCode") != StatusCode.OK){
                returnResult.setStatusCode(StatusCode.FEDERATEDERROR);
                return returnResult;
            }
        }
        catch (IOException ex){
            LOGGER.error(ex);
            returnResult.setStatusCode(StatusCode.IOERROR);
            returnResult.setError(ex.getMessage());
        }
        catch (Exception ex){
            LOGGER.error(ex);
            returnResult.setStatusCode(StatusCode.UNKNOWNERROR);
            returnResult.setError(ex.getMessage());
        }
        return returnResult;
    }

    public static ReturnResult federatedLoadModel(Map<String, Object> requestData){
        ReturnResult returnResult = new ReturnResult();
        returnResult.setStatusCode(StatusCode.OK);
        try{
            String sceneId = String.valueOf(requestData.get("sceneId"));
            String partnerPartyId = String.valueOf(requestData.get("myPartyId"));
            String myRole = FederatedUtils.getMyRole(String.valueOf(requestData.get("myRole")));
            String commitId = String.valueOf(requestData.get("commitId"));
            String branch = String.valueOf(requestData.get("branch"));
            String tag = String.valueOf(requestData.get("tag"));
            returnResult.setData("commitId", commitId);
            returnResult.setStatusCode(pushModelIntoPool(sceneId, partnerPartyId, myRole, commitId, tag, branch));
        }
        catch (Exception ex){
            returnResult.setStatusCode(StatusCode.UNKNOWNERROR);
            returnResult.setMessage(ex.getMessage());
        }
        return returnResult;
    }

    public static ReturnResult publishOnlineModel(String sceneId, String partnerPartyId, String myRole, String commitId){
        ReturnResult returnResult = new ReturnResult();
        MLModel model = modelCache.get(ModelUtils.genModelKey(sceneId, partnerPartyId, myRole, commitId));
        if (model == null){
            returnResult.setStatusCode(StatusCode.NOMODEL);
            returnResult.setMessage("Can not found model by these information.");
            return returnResult;
        }
        try{
            sceneModel.put(ModelUtils.getOnlineModelKey(sceneId, partnerPartyId, myRole), model);
            returnResult.setStatusCode(StatusCode.OK);
        }
        catch (Exception ex){
            returnResult.setStatusCode(StatusCode.UNKNOWNERROR);
            returnResult.setMessage(ex.getMessage());
        }
        return returnResult;
    }

    private static Map<String, Object> requestFederatedLoadModel(String sceneId, String partnerPartyId, String myRole, String commitId, String tag, String branch) throws IOException {
        Proxy.Packet.Builder packetBuilder = Proxy.Packet.newBuilder();
        Map<String, String> requestData = new HashMap<>();
        requestData.put("sceneId", sceneId);
        requestData.put("myPartyId", Configuration.getProperty("partyId"));
        requestData.put("partnerPartyId", partnerPartyId);
        requestData.put("myRole", "guest");
        requestData.put("commitId", commitId);
        requestData.put("tag", tag);
        requestData.put("branch", branch);
        ObjectMapper objectMapper = new ObjectMapper();
        packetBuilder.setBody(Proxy.Data.newBuilder()
                .setValue(ByteString.copyFrom(objectMapper.writeValueAsString(requestData).getBytes()))
                .build());

        Proxy.Metadata.Builder metaDataBuilder = Proxy.Metadata.newBuilder();
        Proxy.Topic.Builder topicBuilder = Proxy.Topic.newBuilder();

        metaDataBuilder.setSrc(
                topicBuilder.setPartyId(Configuration.getProperty("partyId"))
                        .setRole(myRole)
                        .setName("myPartyName")
                        .build());
        metaDataBuilder.setDst(
                topicBuilder.setPartyId(partnerPartyId)
                        .setRole("host")
                        .setName("partnerPartyName")
                        .build());
        metaDataBuilder.setCommand(Proxy.Command.newBuilder().setName("federatedLoadModel").build());
        packetBuilder.setHeader(metaDataBuilder.build());

        ManagedChannel channel1 = ClientPool.getChannel(Configuration.getProperty("proxy"));
        DataTransferServiceGrpc.DataTransferServiceBlockingStub stub1 = DataTransferServiceGrpc.newBlockingStub(channel1);
        Proxy.Packet packet = stub1.unaryCall(packetBuilder.build());
        return objectMapper.readValue(packet.getBody().getValue().toStringUtf8(), HashMap.class);
    }
}
