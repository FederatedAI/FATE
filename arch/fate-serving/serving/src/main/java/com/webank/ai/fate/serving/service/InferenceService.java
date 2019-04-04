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
import com.webank.ai.fate.core.storage.dtable.DTableInfo;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.adapter.dataaccess.FeatureData;
import com.webank.ai.fate.serving.adapter.resultprocessing.ResultData;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import com.webank.ai.fate.serving.manger.ModelManager;
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.serving.utils.DTableUtils;
import com.webank.ai.fate.serving.utils.FederatedUtils;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import java.util.HashMap;
import java.util.Map;


public class InferenceService extends InferenceServiceGrpc.InferenceServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void predict(InferenceRequest req, StreamObserver<InferenceResponse> responseObserver){
        FederatedMeta requestMeta = req.getMeta();

        InferenceResponse.Builder response = InferenceResponse.newBuilder();
        // get model
        String myModelName, myModelNamespace;
        if (req.getModel() != null){
            myModelName = req.getModel().getName();
            myModelNamespace = req.getModel().getNamespace();
        }else{
            DTableInfo modelDTableInfo = DTableUtils.genTableInfo(req.getMeta().getSceneId(),
                    req.getMeta().getMyRole(),
                    req.getMeta().getMyPartyId(),
                    req.getMeta().getPartnerPartyId(),
                    "model_data");
            myModelName = modelDTableInfo.getName();
            myModelNamespace = modelDTableInfo.getNamespace();
        }
        PipelineTask model = ModelManager.getModel(myModelName, myModelNamespace);
        if (model == null){
            response.setStatusCode(StatusCode.NOMODEL);
            FederatedMeta.Builder federatedMetaBuilder = FederatedUtils.genResponseMetaBuilder(requestMeta);
            response.setMeta(federatedMetaBuilder.build());
            responseObserver.onNext(response.build());
            responseObserver.onCompleted();
            return;
        }


        FederatedMeta.Builder federatedMetaBuilder = FederatedUtils.genResponseMetaBuilder(requestMeta);
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
            Map<String, Object> predictParams = new HashMap<>();
            predictParams.put("sid", sid);
            predictParams.put("modelName", myModelName);
            predictParams.put("modelNamespace", myModelNamespace);
            predictParams.put("partyId", Configuration.getPropertyInt("partyId"));
            predictParams.put("partnerPartyId", requestMeta.getPartnerPartyId());
            Map<String, Object> modelResult = model.predict(predictInputData, predictParams);
            Map<String, Object> result = this.getProcessedResult(modelResult);
            response.setData(ByteString.copyFrom(ObjectTransform.bean2Json(result).getBytes()));
        });

        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }

    public ReturnResult federatedPredict(Map<String, Object> requestData){
        LOGGER.info(requestData);
        ReturnResult returnResult = new ReturnResult();
        int partnerPartyId = (int)requestData.get("partyId");
        String partnerModelName = requestData.get("modelName").toString();
        String partnerModelNamespace = requestData.get("modelNamespace").toString();
        LOGGER.info(requestData);
        PipelineTask model = ModelManager.getModelAcPartner(partnerPartyId, partnerModelName, partnerModelNamespace);
        if (model == null){
            returnResult.setStatusCode(StatusCode.NOMODEL);
            returnResult.setMessage("Can not found model.");
            return returnResult;
        }
        Map<String, Object> predictParams = new HashMap<>();
        predictParams.putAll(requestData);
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


    public Object getClassByName(String classPath){
        try{
            Class thisClass = Class.forName(classPath);
            return thisClass.getConstructor().newInstance();
        }
        catch (ClassNotFoundException ex){
            LOGGER.error("Can not found this class: {}.", classPath);
        }
        catch (NoSuchMethodException ex){
            LOGGER.error("Can not get this class({}) constructor.", classPath);
        }
        catch (Exception ex){
            LOGGER.error("Can not create class({}) instance.", classPath);
        }
        return null;
    }

    private Map<String, Object> getProcessedResult(Map<String, Object> modelResult){
        String classPath = ResultData.class.getPackage().getName() + "." + Configuration.getProperty("InferenceResultProcessingAdapter");
        ResultData resultData = (ResultData) this.getClassByName(classPath);
        if (resultData == null){
            return null;
        }
        return resultData.getResult(modelResult);
    }

    private Map<String, Object> getFeatureData(String sid){
        String classPath = FeatureData.class.getPackage().getName() + "." + Configuration.getProperty("OnlineDataAccessAdapter");
        FeatureData featureData = (FeatureData) getClassByName(classPath);
        if (featureData == null){
            return null;
        }
        return featureData.getData(sid);
    }
}
