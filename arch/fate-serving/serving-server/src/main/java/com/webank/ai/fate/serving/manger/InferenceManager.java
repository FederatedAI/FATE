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
import com.webank.ai.fate.core.bean.FederatedParty;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.adapter.dataaccess.FeatureData;
import com.webank.ai.fate.serving.adapter.resultprocessing.ResultData;
import com.webank.ai.fate.core.bean.FederatedRoles;
import com.webank.ai.fate.serving.bean.InferenceRequest;
import com.webank.ai.fate.serving.bean.ModelNamespaceData;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import com.webank.ai.fate.core.constant.StatusCode;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

public class InferenceManager {
    private static final Logger LOGGER = LogManager.getLogger();
    public static ReturnResult inference(InferenceRequest inferenceRequest){
        ReturnResult returnResult = new ReturnResult();
        String modelTableName = inferenceRequest.getModelName();
        String modelNamespace = inferenceRequest.getModelNamespace();
        LOGGER.info(inferenceRequest);
        if (StringUtils.isEmpty(modelNamespace) && inferenceRequest.getSceneId() != 0){
            modelNamespace = ModelManager.getNamespaceBySceneId(inferenceRequest.getSceneId());
        }
        LOGGER.info(modelNamespace);
        if (StringUtils.isEmpty(modelNamespace)){
            returnResult.setRetcode(StatusCode.NOMODEL);
            return returnResult;
        }
        ModelNamespaceData modelNamespaceData = ModelManager.getModelNamespaceData(modelNamespace);
        PipelineTask model;
        if (StringUtils.isEmpty(modelTableName)){
            modelTableName = modelNamespaceData.getUsedModelName();
            model = modelNamespaceData.getUsedModel();
        }else {
            model = ModelManager.getModel(modelNamespaceData.getLocal().getRole(), modelNamespaceData.getLocal().getPartyId(), modelNamespaceData.getRole(), modelTableName, modelNamespace);
        }
        if (model == null){
            returnResult.setRetcode(StatusCode.NOMODEL);
            return returnResult;
        }
        LOGGER.info("use model to inference, name: {}, namespace: {}", modelTableName, modelNamespace);
        Map<String, Object> featureData = inferenceRequest.getFeatureData();

        if (featureData == null){
            returnResult.setRetcode(StatusCode.ILLEGALDATA);
            returnResult.setRetmsg("Can not parse data json.");
            return returnResult;
        }

        Map<String, Object> predictParams = new HashMap<>();
        Map<String, Object> federatedParams = new HashMap<>();
        federatedParams.put("local", modelNamespaceData.getLocal());
        federatedParams.put("model_info", new ModelInfo(modelTableName, modelNamespace));
        federatedParams.put("role", modelNamespaceData.getRole());
        federatedParams.put("device_id", featureData.get("device_id"));
        federatedParams.put("phone_num", featureData.get("phone_num"));
        predictParams.put("federatedParams", federatedParams);

        Map<String, Object> modelResult = model.predict(inferenceRequest.getFeatureData(), predictParams);
        Map<String, Object> result = getProcessedResult(modelResult);
        returnResult.putAllData(result);
        LOGGER.info("Inference successfully.");
        return returnResult;
    }

    public static ReturnResult federatedInference(Map<String, Object> federatedParams){
        LOGGER.info(federatedParams);
        ReturnResult returnResult = new ReturnResult();
        FederatedParty partnerParty = (FederatedParty) ObjectTransform.json2Bean(federatedParams.get("partner_local").toString(), FederatedParty.class);
        FederatedParty party = (FederatedParty) ObjectTransform.json2Bean(federatedParams.get("local").toString(), FederatedParty.class);
        FederatedRoles federatedRoles = (FederatedRoles) ObjectTransform.json2Bean(federatedParams.get("role").toString(), FederatedRoles.class);
        ModelInfo partnerModelInfo = (ModelInfo) ObjectTransform.json2Bean(federatedParams.get("partner_model_info").toString(), ModelInfo.class);

        PipelineTask model = ModelManager.getModelByPartner(party.getRole(), party.getPartyId(), partnerParty.getRole(),
                partnerParty.getPartyId(), federatedRoles, partnerModelInfo.getName(), partnerModelInfo.getNamespace());
        if (model == null){
            returnResult.setRetcode(StatusCode.NOMODEL);
            returnResult.setRetmsg("Can not found model.");
            return returnResult;
        }
        Map<String, Object> predictParams = new HashMap<>();
        predictParams.put("federatedParams", federatedParams);
        try{
            Map<String, Object> featureData = getFeatureData(federatedParams);
            if (featureData == null || featureData.size() < 1){
                returnResult.setRetcode(StatusCode.FEDERATEDERROR);
                returnResult.setRetmsg("Can not get feature data.");
                return returnResult;
            }
            Map<String, Object> result = model.predict(featureData, predictParams);
            returnResult.setRetcode(StatusCode.OK);
            returnResult.putAllData(result);
        }
        catch (Exception ex){
            LOGGER.error(ex.getStackTrace());
            returnResult.setRetcode(StatusCode.FEDERATEDERROR);
            returnResult.setRetmsg(ex.getMessage());
        }
        return returnResult;
    }


    public static Object getClassByName(String classPath){
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

    private static Map<String, Object> getProcessedResult(Map<String, Object> modelResult){
        String classPath = ResultData.class.getPackage().getName() + "." + Configuration.getProperty("InferenceResultProcessingAdapter");
        ResultData resultData = (ResultData) getClassByName(classPath);
        if (resultData == null){
            return null;
        }
        return resultData.getResult(modelResult);
    }

    private static Map<String, Object> getFeatureData(Map<String, Object> featureId){
        String classPath = FeatureData.class.getPackage().getName() + "." + Configuration.getProperty("OnlineDataAccessAdapter");
        FeatureData featureData = (FeatureData) getClassByName(classPath);
        if (featureData == null){
            return null;
        }
        return featureData.getData(featureId);
    }
}
