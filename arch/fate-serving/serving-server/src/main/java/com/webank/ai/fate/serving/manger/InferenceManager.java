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
import com.webank.ai.fate.serving.adapter.processing.PostProcessing;
import com.webank.ai.fate.core.bean.FederatedRoles;
import com.webank.ai.fate.serving.adapter.processing.PreProcessing;
import com.webank.ai.fate.serving.bean.FederatedInferenceType;
import com.webank.ai.fate.serving.bean.InferenceRequest;
import com.webank.ai.fate.serving.bean.ModelNamespaceData;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.serving.utils.InferenceUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class InferenceManager {
    private static final Logger LOGGER = LogManager.getLogger();
    public static ReturnResult inference(InferenceRequest inferenceRequest){
        ReturnResult returnResult = new ReturnResult();
        String modelTableName = inferenceRequest.getModelName();
        String modelNamespace = inferenceRequest.getModelNamespace();
        if (StringUtils.isEmpty(modelNamespace) && inferenceRequest.getSceneid() != 0){
            modelNamespace = ModelManager.getNamespaceBySceneId(inferenceRequest.getSceneid());
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
            logAudited(inferenceRequest, modelNamespaceData, returnResult, false);
            return returnResult;
        }

        featureData = getPreProcessingFeatureData(featureData);
        if (featureData == null){
            returnResult.setRetcode(StatusCode.ILLEGALDATA);
            returnResult.setRetmsg("Can not preprocessing data");
            logAudited(inferenceRequest, modelNamespaceData, returnResult, false);
            return returnResult;
        }


        Map<String, Object> predictParams = new HashMap<>();
        Map<String, Object> federatedParams = new HashMap<>();
        federatedParams.put("sceneid", inferenceRequest.getSceneid());
        federatedParams.put("caseid", inferenceRequest.getCaseid());
        federatedParams.put("seqno", inferenceRequest.getSeqno());
        federatedParams.put("local", modelNamespaceData.getLocal());
        federatedParams.put("model_info", new ModelInfo(modelTableName, modelNamespace));
        federatedParams.put("role", modelNamespaceData.getRole());
        federatedParams.put("device_id", featureData.get("device_id"));
        federatedParams.put("phone_num", featureData.get("phone_num"));
        predictParams.put("federatedParams", federatedParams);

        Map<String, Object> modelResult = model.predict(inferenceRequest.getFeatureData(), predictParams);
        Map<String, Object> result = getPostProcessedResult(featureData, modelResult);
        for(String field: Arrays.asList("data", "log", "warn")){
            if (result.get(field) != null){
                returnResult.putAllData((Map<String, Object>) result.get(field));
            }
        }
        LOGGER.info("Inference successfully.");
        logAudited(inferenceRequest, modelNamespaceData, returnResult, true);
        return returnResult;
    }

    public static ReturnResult federatedInference(Map<String, Object> federatedParams){
        ReturnResult returnResult = new ReturnResult();
        //TODO: Very ugly, need to be optimized
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
            logAudited(federatedParams, party, federatedRoles, returnResult, true);
        }
        catch (Exception ex){
            LOGGER.info("federatedInference", ex);
            returnResult.setRetcode(StatusCode.FEDERATEDERROR);
            returnResult.setRetmsg(ex.getMessage());
        }
        return returnResult;
    }

    private static Map<String, Object> getPreProcessingFeatureData(Map<String, Object> originFeatureData){
        try{
            String classPath = PreProcessing.class.getPackage().getName() + "." + Configuration.getProperty("InferencePreProcessingAdapter");
            PreProcessing preProcessing = (PreProcessing) getClassByName(classPath);
            return preProcessing.getResult(ObjectTransform.bean2Json(originFeatureData));
        }catch (Exception ex){
            LOGGER.error("", ex);
            return null;
        }
    }

    private static Map<String, Object> getPostProcessedResult(Map<String, Object> featureData, Map<String, Object> modelResult){
        try{
            String classPath = PostProcessing.class.getPackage().getName() + "." + Configuration.getProperty("InferencePostProcessingAdapter");
            PostProcessing postProcessing = (PostProcessing) getClassByName(classPath);
            return postProcessing.getResult(featureData, modelResult);
        }catch (Exception ex){
            LOGGER.error("", ex);
            return null;
        }
    }

    private static Map<String, Object> getFeatureData(Map<String, Object> featureId){
        String classPath = FeatureData.class.getPackage().getName() + "." + Configuration.getProperty("OnlineDataAccessAdapter");
        FeatureData featureData = (FeatureData) getClassByName(classPath);
        if (featureData == null){
            return null;
        }
        try{
            return featureData.getData(featureId);
        }
        catch (Exception ex){
            LOGGER.error(ex);
        }
        return null;
    }

    private static void logAudited(InferenceRequest inferenceRequest, ModelNamespaceData modelNamespaceData, ReturnResult returnResult, boolean charge){
        InferenceUtils.logInferenceAudited(FederatedInferenceType.INITIATED, inferenceRequest.getSceneid(), modelNamespaceData.getLocal(), modelNamespaceData.getRole(), inferenceRequest.getCaseid(), returnResult.getRetcode(), charge);
    }
    private static void logAudited(Map<String, Object> federatedParams, FederatedParty federatedParty, FederatedRoles federatedRoles, ReturnResult returnResult, boolean charge){
        LOGGER.info(federatedParams);
        LOGGER.info(federatedParty);
        LOGGER.info(federatedRoles);
        InferenceUtils.logInferenceAudited(FederatedInferenceType.FEDERATED, (int)federatedParams.get("sceneid"), federatedParty, federatedRoles, federatedParams.get("caseid").toString(), returnResult.getRetcode(), charge);
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
            LOGGER.error(ex);
            LOGGER.error("Can not create class({}) instance.", classPath);
        }
        return null;
    }
}
