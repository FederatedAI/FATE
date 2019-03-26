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
import com.webank.ai.fate.api.serving.InferenceServiceGrpc;
import com.webank.ai.fate.api.serving.InferenceServiceProto.InferenceRequest;
import com.webank.ai.fate.api.serving.InferenceServiceProto.InferenceResponse;
import com.webank.ai.fate.api.serving.InferenceServiceProto.FederatedMeta;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.manger.ModelManager;
import com.webank.ai.fate.core.mlmodel.model.MLModel;
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.serving.utils.FederatedUtils;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class PredictService extends InferenceServiceGrpc.InferenceServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void predict(InferenceRequest req, StreamObserver<InferenceResponse> responseObserver){
        FederatedMeta requestMeta = req.getMeta();

        InferenceResponse.Builder response = InferenceResponse.newBuilder();
        String myRole = FederatedUtils.getMyRole(requestMeta.getMyRole());

        // get model
        MLModel model = new ModelManager().getModel(requestMeta.getSceneId(), requestMeta.getPartnerPartyId(), myRole);
        if (model == null){
            response.setStatusCode(StatusCode.NOMODEL);
            FederatedMeta.Builder federatedMetaBuilder = FederatedUtils.genResponseMetaBuilder(requestMeta, "");
            response.setMeta(federatedMetaBuilder.build());
            responseObserver.onNext(response.build());
            responseObserver.onCompleted();
            return;
        }


        FederatedMeta.Builder federatedMetaBuilder = FederatedUtils.genResponseMetaBuilder(requestMeta, (String)model.getModelInfo().get("commitId"));
        // set response meta
        response.setMeta(federatedMetaBuilder.build());
        // deal data
        Object inputObject = ObjectTransform.json2Bean(req.getData().toStringUtf8(), HashMap.class);
        if (inputObject == null){
            response.setStatusCode(StatusCode.ILLEGALDATA);
            response.setMessage("Can not parse data json.");
            responseObserver.onNext(response.build());
            responseObserver.onCompleted();
            return;
        }
        Map<String, Object> inputData = (Map<String, Object>) inputObject;
        inputData.forEach((sid, f)->{
            Map<String, Object> predictInputData = (Map<String, Object>)f;
            Map<String, String> predictParams = new HashMap<>();
            predictParams.put("sceneId", requestMeta.getSceneId());
            predictParams.put("sid", sid);
            predictParams.put("commitId", (String)model.getModelInfo().get("commitId"));
            Map<String, Object> result = model.predict(predictInputData, predictParams);
            response.setData(ByteString.copyFrom(ObjectTransform.bean2Json(result).getBytes()));
        });

        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }

    public ReturnResult federatedPredict(Map<String, Object> requestData){
        ReturnResult returnResult = new ReturnResult();
        String myPartyId = Configuration.getProperty("partyId");
        String partnerPartyId = requestData.get("myPartyId").toString();
        String myRole = FederatedUtils.getMyRole(requestData.get("myRole").toString());
        MLModel model = new ModelManager().getModel(requestData.get("sceneId").toString(), partnerPartyId, myRole, requestData.get("commitId").toString());
        if (model == null){
            returnResult.setStatusCode(StatusCode.NOMODEL);
            returnResult.setMessage("Can not found model.");
            return returnResult;
        }
        Map<String, Object> predictParams = new HashMap<>();
        predictParams.putAll(requestData);
        predictParams.put("myPartyId", myPartyId);
        predictParams.put("partnerPartyId", partnerPartyId);
        predictParams.put("myRole", myRole);
        try{
            Map<String, Object> inputData = getFeatureData(requestData.get("sid").toString());
            if (inputData == null || inputData.size() < 1){
                returnResult.setStatusCode(StatusCode.FEDERATEDERROR);
                returnResult.setMessage("Can not get feature data.");
                return returnResult;
            }
            Map<String, Object> result = model.predict(inputData, predictParams);
            returnResult.setStatusCode(StatusCode.OK);
            returnResult.putAllData(result);
        }
        catch (Exception ex){
            returnResult.setStatusCode(StatusCode.FEDERATEDERROR);
            returnResult.setMessage(ex.getMessage());
        }
        return returnResult;
    }

    private Map<String, Object> getFeatureData(String sid){
        Map<String, Object> featureData = new HashMap<>();
        try{
            List<String> lines = Files.readAllLines(Paths.get(System.getProperty("user.dir"), "host_data.csv"));
            lines.forEach(line->{
                for(String kv: StringUtils.split(line, ",")){
                    String[] a = StringUtils.split(kv, ":");
                    featureData.put(a[0], Float.parseFloat(a[1]));
                }
            });
        }
        catch (Exception ex){
            LOGGER.error(ex);
        }
        return featureData;
    }
}
