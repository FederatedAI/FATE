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
import com.webank.ai.fate.core.bean.FederatedRoles;
import com.webank.ai.fate.core.bean.ReturnResult;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.adapter.dataaccess.FeatureData;
import com.webank.ai.fate.serving.adapter.processing.PostProcessing;
import com.webank.ai.fate.serving.adapter.processing.PreProcessing;
import com.webank.ai.fate.serving.bean.InferenceRequest;
import com.webank.ai.fate.serving.bean.ModelNamespaceData;
import com.webank.ai.fate.serving.bean.PostProcessingResult;
import com.webank.ai.fate.serving.bean.PreProcessingResult;
import com.webank.ai.fate.serving.core.bean.FederatedInferenceType;
import com.webank.ai.fate.serving.core.bean.InferenceActionType;
import com.webank.ai.fate.serving.core.constant.InferenceRetCode;
import com.webank.ai.fate.serving.core.manager.CacheManager;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import com.webank.ai.fate.serving.utils.InferenceUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

public class InferenceManager {
    private static final Logger LOGGER = LogManager.getLogger();

    public static ReturnResult inference(InferenceRequest inferenceRequest, InferenceActionType inferenceActionType) {
        ReturnResult inferenceResultFromCache = CacheManager.getInferenceResultCache(inferenceRequest.getAppid(), inferenceRequest.getCaseid());
        if (inferenceResultFromCache != null) {
            LOGGER.info("Get inference result from cache.");
            return inferenceResultFromCache;
        }
        switch (inferenceActionType) {
            case SYNC_RUN:
                ReturnResult inferenceResult = runInference(inferenceRequest);
                return inferenceResult;
            case GET_RESULT:
                ReturnResult noCacheInferenceResult = new ReturnResult();
                noCacheInferenceResult.setRetcode(InferenceRetCode.NO_RESULT);
                return noCacheInferenceResult;
            case ASYNC_RUN:
                InferenceWorkerManager.exetute(new Runnable() {
                    @Override
                    public void run() {
                        runInference(inferenceRequest);
                        LOGGER.info("Inference task exit.");
                    }
                });
                ReturnResult startInferenceJobResult = new ReturnResult();
                startInferenceJobResult.setRetcode(InferenceRetCode.OK);
                startInferenceJobResult.setCaseid(inferenceRequest.getCaseid());
                return startInferenceJobResult;
            default:
                ReturnResult systemErrorReturnResult = new ReturnResult();
                systemErrorReturnResult.setRetcode(InferenceRetCode.SYSTEM_ERROR);
                return systemErrorReturnResult;
        }
    }

    public static ReturnResult runInference(InferenceRequest inferenceRequest) {
        ReturnResult inferenceResult = new ReturnResult();
        inferenceResult.setCaseid(inferenceRequest.getCaseid());
        String modelName = inferenceRequest.getModelVersion();
        String modelNamespace = inferenceRequest.getModelId();
        if (StringUtils.isEmpty(modelNamespace) && inferenceRequest.haveAppId()) {
            modelNamespace = ModelManager.getModelNamespaceByPartyId(inferenceRequest.getAppid());
        }
        if (StringUtils.isEmpty(modelNamespace)) {
            inferenceResult.setRetcode(InferenceRetCode.LOAD_MODEL_FAILED + 1000);
            return inferenceResult;
        }
        ModelNamespaceData modelNamespaceData = ModelManager.getModelNamespaceData(modelNamespace);
        PipelineTask model;
        if (StringUtils.isEmpty(modelName)) {
            modelName = modelNamespaceData.getUsedModelName();
            model = modelNamespaceData.getUsedModel();
        } else {
            model = ModelManager.getModel(modelName, modelNamespace);
        }
        if (model == null) {
            inferenceResult.setRetcode(InferenceRetCode.LOAD_MODEL_FAILED + 1000);
            return inferenceResult;
        }
        LOGGER.info("use model to inference for {}, id: {}, version: {}", inferenceRequest.getAppid(), modelNamespace, modelName);
        Map<String, Object> rawFeatureData = inferenceRequest.getFeatureData();

        if (rawFeatureData == null) {
            inferenceResult.setRetcode(InferenceRetCode.EMPTY_DATA + 1000);
            inferenceResult.setRetmsg("Can not parse data json.");
            logInferenceAudited(inferenceRequest, modelNamespaceData, inferenceResult, false, false, rawFeatureData);
            return inferenceResult;
        }

        PreProcessingResult preProcessingResult;
        try {
            preProcessingResult = getPreProcessingFeatureData(rawFeatureData);
        } catch (Exception ex) {
            LOGGER.error("feature data preprocessing failed", ex);
            inferenceResult.setRetcode(InferenceRetCode.INVALID_FEATURE + 1000);
            inferenceResult.setRetmsg(ex.getMessage());
            return inferenceResult;
        }
        Map<String, Object> featureData = preProcessingResult.getProcessingResult();
        Map<String, Object> featureIds = preProcessingResult.getFeatureIds();
        if (featureData == null) {
            inferenceResult.setRetcode(InferenceRetCode.NUMERICAL_ERROR + 1000);
            inferenceResult.setRetmsg("Can not preprocessing data");
            logInferenceAudited(inferenceRequest, modelNamespaceData, inferenceResult, false, false, rawFeatureData);
            return inferenceResult;
        }


        Map<String, Object> predictParams = new HashMap<>();
        Map<String, Object> federatedParams = new HashMap<>();
        federatedParams.put("caseid", inferenceRequest.getCaseid());
        federatedParams.put("seqno", inferenceRequest.getSeqno());
        federatedParams.put("local", modelNamespaceData.getLocal());
        federatedParams.put("model_info", new ModelInfo(modelName, modelNamespace));
        federatedParams.put("role", modelNamespaceData.getRole());
        federatedParams.put("feature_id", featureIds);
        predictParams.put("federatedParams", federatedParams);

        Map<String, Object> modelResult = model.predict(featureData, predictParams);
        LOGGER.info(modelResult);
        PostProcessingResult postProcessingResult;
        try {
            postProcessingResult = getPostProcessedResult(featureData, modelResult);
        } catch (Exception ex) {
            LOGGER.error("model result postprocessing failed", ex);
            inferenceResult.setRetcode(InferenceRetCode.COMPUTE_ERROR);
            inferenceResult.setRetmsg(ex.getMessage());
            return inferenceResult;
        }
        inferenceResult = postProcessingResult.getProcessingResult();
        inferenceResult.setCaseid(inferenceRequest.getCaseid());
        boolean fromCache = (boolean) federatedParams.getOrDefault("is_cache", false);
        ReturnResult federatedResult = (ReturnResult) predictParams.get("federatedResult");
        boolean billing = true;
        if (fromCache) {
            billing = false;
        } else if (federatedResult.getRetcode() == InferenceRetCode.GET_FEATURE_FAILED || federatedResult.getRetcode() == InferenceRetCode.INVALID_FEATURE || federatedResult.getRetcode() == InferenceRetCode.NO_FEATURE) {
            billing = false;
        }
        int partyInferenceRetcode = 0;
        if (inferenceResult.getRetcode() != 0) {
            partyInferenceRetcode += 1;
        }
        if (federatedResult.getRetcode() != 0) {
            partyInferenceRetcode += 2;
            inferenceResult.setRetcode(federatedResult.getRetcode());
        }
        inferenceResult.setRetcode(inferenceResult.getRetcode() + partyInferenceRetcode * 1000);
        logInferenceAudited(inferenceRequest, modelNamespaceData, inferenceResult, fromCache, billing, rawFeatureData);
        if (inferenceResult.getRetcode() == 0) {
            CacheManager.putInferenceResultCache(inferenceRequest.getAppid(), inferenceRequest.getCaseid(), inferenceResult);
            LOGGER.info("inference successfully.");
        } else {
            LOGGER.info("failed inference, retcode is {}.", inferenceResult.getRetcode());
        }
        return inferenceResult;
    }

    public static ReturnResult federatedInference(Map<String, Object> federatedParams) {
        ReturnResult returnResult = new ReturnResult();
        //TODO: Very ugly, need to be optimized
        FederatedParty party = (FederatedParty) ObjectTransform.json2Bean(federatedParams.get("local").toString(), FederatedParty.class);
        FederatedRoles federatedRoles = (FederatedRoles) ObjectTransform.json2Bean(federatedParams.get("role").toString(), FederatedRoles.class);
        ModelInfo partnerModelInfo = (ModelInfo) ObjectTransform.json2Bean(federatedParams.get("partner_model_info").toString(), ModelInfo.class);
        Map<String, Object> featureIds = (Map<String, Object>) ObjectTransform.json2Bean(federatedParams.get("feature_id").toString(), HashMap.class);

        ModelInfo modelInfo = ModelManager.getModelInfoByPartner(partnerModelInfo.getName(), partnerModelInfo.getNamespace());
        if (modelInfo == null) {
            returnResult.setRetcode(InferenceRetCode.LOAD_MODEL_FAILED);
            returnResult.setRetmsg("Can not found model.");
            return returnResult;
        }
        PipelineTask model = ModelManager.getModel(modelInfo.getName(), modelInfo.getNamespace());
        if (model == null) {
            returnResult.setRetcode(InferenceRetCode.LOAD_MODEL_FAILED);
            returnResult.setRetmsg("Can not found model.");
            return returnResult;
        }
        LOGGER.info("use model to inference on {} {}, id: {}, version: {}", party.getRole(), party.getPartyId(), modelInfo.getNamespace(), modelInfo.getName());
        Map<String, Object> predictParams = new HashMap<>();
        predictParams.put("federatedParams", federatedParams);
        try {
            ReturnResult getFeatureDataResult = getFeatureData(featureIds);
            if (getFeatureDataResult.getRetcode() == InferenceRetCode.OK) {
                if (getFeatureDataResult.getData() == null || getFeatureDataResult.getData().size() < 1) {
                    returnResult.setRetcode(InferenceRetCode.GET_FEATURE_FAILED);
                    returnResult.setRetmsg("Can not get feature data.");
                    return returnResult;
                }
                Map<String, Object> result = model.predict(getFeatureDataResult.getData(), predictParams);
                returnResult.setRetcode(InferenceRetCode.OK);
                returnResult.setData(result);
                logInferenceAudited(federatedParams, party, federatedRoles, returnResult, false, true);
            } else {
                returnResult.setRetcode(getFeatureDataResult.getRetcode());
            }
        } catch (Exception ex) {
            LOGGER.info("federatedInference error:", ex);
            returnResult.setRetcode(InferenceRetCode.SYSTEM_ERROR);
            returnResult.setRetmsg(ex.getMessage());
        }
        LOGGER.info(returnResult.getData());
        LOGGER.info("federated inference successfully");
        return returnResult;
    }

    private static PreProcessingResult getPreProcessingFeatureData(Map<String, Object> originFeatureData) {
        String classPath = PreProcessing.class.getPackage().getName() + "." + Configuration.getProperty("InferencePreProcessingAdapter");
        PreProcessing preProcessing = (PreProcessing) InferenceUtils.getClassByName(classPath);
        return preProcessing.getResult(ObjectTransform.bean2Json(originFeatureData));
    }

    private static PostProcessingResult getPostProcessedResult(Map<String, Object> featureData, Map<String, Object> modelResult) {
        String classPath = PostProcessing.class.getPackage().getName() + "." + Configuration.getProperty("InferencePostProcessingAdapter");
        PostProcessing postProcessing = (PostProcessing) InferenceUtils.getClassByName(classPath);
        return postProcessing.getResult(featureData, modelResult);
    }

    private static ReturnResult getFeatureData(Map<String, Object> featureIds) {
        ReturnResult defaultReturnResult = new ReturnResult();
        String classPath = FeatureData.class.getPackage().getName() + "." + Configuration.getProperty("OnlineDataAccessAdapter");
        FeatureData featureData = (FeatureData) InferenceUtils.getClassByName(classPath);
        if (featureData == null) {
            defaultReturnResult.setRetcode(InferenceRetCode.ADAPTER_ERROR);
            return defaultReturnResult;
        }
        try {
            return featureData.getData(featureIds);
        } catch (Exception ex) {
            defaultReturnResult.setRetcode(InferenceRetCode.GET_FEATURE_FAILED);
            LOGGER.error("get feature data error:", ex);
            return defaultReturnResult;
        }
    }

    private static void logInferenceAudited(InferenceRequest inferenceRequest, ModelNamespaceData modelNamespaceData, ReturnResult returnResult, boolean useCache, boolean billing, Map<String, Object> featureData) {
        InferenceUtils.logInference(FederatedInferenceType.INITIATED, modelNamespaceData.getLocal(), modelNamespaceData.getRole(), inferenceRequest.getCaseid(), inferenceRequest.getSeqno(), returnResult.getRetcode(), featureData);
        InferenceUtils.logInferenceAudited(FederatedInferenceType.INITIATED, modelNamespaceData.getLocal(), modelNamespaceData.getRole(), inferenceRequest.getCaseid(), inferenceRequest.getSeqno(), returnResult.getRetcode(), useCache, billing);
    }

    private static void logInferenceAudited(Map<String, Object> federatedParams, FederatedParty federatedParty, FederatedRoles federatedRoles, ReturnResult returnResult, boolean useCache, boolean billing) {
        InferenceUtils.logInferenceAudited(FederatedInferenceType.FEDERATED, federatedParty, federatedRoles, federatedParams.get("caseid").toString(), federatedParams.get("seqno").toString(), returnResult.getRetcode(), useCache, billing);
    }
}
