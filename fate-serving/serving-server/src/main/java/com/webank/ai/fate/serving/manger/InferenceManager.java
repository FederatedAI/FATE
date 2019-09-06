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
import com.google.common.collect.Maps;
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
import com.webank.ai.fate.serving.core.bean.*;
import com.webank.ai.fate.serving.core.constant.InferenceRetCode;
import com.webank.ai.fate.serving.core.manager.CacheManager;
import com.webank.ai.fate.serving.core.monitor.WatchDog;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import com.webank.ai.fate.serving.utils.InferenceUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

public class InferenceManager {
    private static final Logger LOGGER = LogManager.getLogger();

    static  PostProcessing   postProcessing ;

    static  PreProcessing    preProcessing;

    static {
        try {
            String classPathPre = PostProcessing.class.getPackage().getName();
            String postClassPath = classPathPre + "." + Configuration.getProperty(Dict.POST_PROCESSING_CONFIG);
            postProcessing = (PostProcessing) InferenceUtils.getClassByName(postClassPath);
            String preClassPath = classPathPre + "." + Configuration.getProperty(Dict.PRE_PROCESSING_CONFIG);
            preProcessing = (PreProcessing) InferenceUtils.getClassByName(preClassPath);
        }catch(Throwable e){
            LOGGER.error("load post/pre processing error",e);
        }

    }



    public static ReturnResult inference(Context  context,InferenceRequest inferenceRequest, InferenceActionType inferenceActionType) {
        long inferenceBeginTime = System.currentTimeMillis();
        ReturnResult inferenceResultFromCache = CacheManager.getInferenceResultCache(inferenceRequest.getAppid(), inferenceRequest.getCaseid());
        LOGGER.info("caseid {} query cache cost {}",inferenceRequest.getCaseid(),System.currentTimeMillis()-inferenceBeginTime);
        if (inferenceResultFromCache != null) {
            LOGGER.info("request caseId {} cost time {}  hit cache true",inferenceRequest.getCaseid(),System.currentTimeMillis()-inferenceBeginTime);
            return inferenceResultFromCache;
        }
        switch (inferenceActionType) {
            case SYNC_RUN:
                ReturnResult inferenceResult = runInference(  context,inferenceRequest);
                if (inferenceResult!=null&&inferenceResult.getRetcode() == 0) {
                    CacheManager.putInferenceResultCache(context ,inferenceRequest.getAppid(), inferenceRequest.getCaseid(), inferenceResult);
                }

                return inferenceResult;
            case GET_RESULT:
                ReturnResult noCacheInferenceResult = new ReturnResult();
                noCacheInferenceResult.setRetcode(InferenceRetCode.NO_RESULT);
                return noCacheInferenceResult;
            case ASYNC_RUN:
                long  beginTime= System.currentTimeMillis();
                InferenceWorkerManager.exetute(new Runnable() {

                    @Override
                    public void run() {
                        ReturnResult inferenceResult=null;
                        Context subContext = context.subContext();
                        subContext.preProcess();

                        try {

                             subContext.setActionType("ASYNC_EXECUTE");
                             inferenceResult=   runInference(subContext,inferenceRequest);
                             if (inferenceResult!=null&&inferenceResult.getRetcode() == 0) {
                                CacheManager.putInferenceResultCache(subContext ,inferenceRequest.getAppid(), inferenceRequest.getCaseid(), inferenceResult);
                             }
                        }finally {

                            subContext.postProcess(inferenceRequest,inferenceResult);
                        }
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

    public static ReturnResult runInference(Context  context ,InferenceRequest inferenceRequest) {
        long startTime = System.currentTimeMillis();

        context.setCaseId(inferenceRequest.getCaseid());
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
            logInference(context ,inferenceRequest, modelNamespaceData, inferenceResult, 0, false, false);
            return inferenceResult;
        }

        PreProcessingResult preProcessingResult;
        try {

            preProcessingResult = getPreProcessingFeatureData(context ,rawFeatureData);
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
            logInference(context,inferenceRequest, modelNamespaceData, inferenceResult, 0, false, false);
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

        Map<String,Object>  modelFeatureData  =  Maps.newHashMap(featureData);
        Map<String, Object> modelResult = model.predict(context,modelFeatureData, predictParams);


       // boolean getRemotePartyResult = (boolean) federatedParams.getOrDefault("getRemotePartyResult", false);
        //ReturnResult federatedResult = (ReturnResult) predictParams.get("federatedResult");

        ReturnResult federatedResult = context.getFederatedResult();
        LOGGER.info(modelResult);
        PostProcessingResult postProcessingResult;
        try {

            postProcessingResult = getPostProcessedResult(context,featureData, modelResult);


        } catch (Exception ex) {
            LOGGER.error("model result postprocessing failed", ex);
            inferenceResult.setRetcode(InferenceRetCode.COMPUTE_ERROR);
            inferenceResult.setRetmsg(ex.getMessage());
            return inferenceResult;
        }
        inferenceResult = postProcessingResult.getProcessingResult();
        inferenceResult.setCaseid(inferenceRequest.getCaseid());
        boolean getRemotePartyResult = (boolean)context.getDataOrDefault(Dict.GET_REMOTE_PARTY_RESULT,false);
        boolean billing = true;
        if (! getRemotePartyResult) {
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
        long endTime = System.currentTimeMillis();
        long inferenceElapsed = endTime - startTime;
        logInference(context,inferenceRequest, modelNamespaceData, inferenceResult, inferenceElapsed, getRemotePartyResult, billing);

        inferenceResult=postProcessing.handleResult(context,inferenceResult);


        return inferenceResult;
    }

    public static ReturnResult federatedInference(Context  context,Map<String, Object> federatedParams) {
        long startTime = System.currentTimeMillis();
        ReturnResult returnResult = new ReturnResult();
        //TODO: Very ugly, need to be optimized
        FederatedParty party = (FederatedParty) ObjectTransform.json2Bean(federatedParams.get("local").toString(), FederatedParty.class);
        FederatedRoles federatedRoles = (FederatedRoles) ObjectTransform.json2Bean(federatedParams.get("role").toString(), FederatedRoles.class);
        ModelInfo partnerModelInfo = (ModelInfo) ObjectTransform.json2Bean(federatedParams.get("partner_model_info").toString(), ModelInfo.class);
        Map<String, Object> featureIds = (Map<String, Object>) ObjectTransform.json2Bean(federatedParams.get("feature_id").toString(), HashMap.class);
        boolean billing = false;

        ModelInfo modelInfo = ModelManager.getModelInfoByPartner(partnerModelInfo.getName(), partnerModelInfo.getNamespace());
        if (modelInfo == null) {
            returnResult.setRetcode(InferenceRetCode.LOAD_MODEL_FAILED);
            returnResult.setRetmsg("Can not found model.");
            logInference(context,federatedParams, party, federatedRoles, returnResult, 0, false, false);
            return returnResult;
        }
        PipelineTask model = ModelManager.getModel(modelInfo.getName(), modelInfo.getNamespace());
        if (model == null) {
            returnResult.setRetcode(InferenceRetCode.LOAD_MODEL_FAILED);
            returnResult.setRetmsg("Can not found model.");
            logInference(context ,federatedParams, party, federatedRoles, returnResult, 0, false, false);
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
                    logInference(context,federatedParams, party, federatedRoles, returnResult, 0, false, false);
                    return returnResult;
                }
                Map<String, Object> result = model.predict(context,getFeatureDataResult.getData(), predictParams);
                returnResult.setRetcode(InferenceRetCode.OK);
                returnResult.setData(result);
                billing = true;
            } else {
                returnResult.setRetcode(getFeatureDataResult.getRetcode());
            }
        } catch (Exception ex) {
            LOGGER.info("federatedInference error:", ex);
            returnResult.setRetcode(InferenceRetCode.SYSTEM_ERROR);
            returnResult.setRetmsg(ex.getMessage());
        }
        long endTime = System.currentTimeMillis();
        long federatedInferenceElapsed = endTime - startTime;
        logInference(context ,federatedParams, party, federatedRoles, returnResult, federatedInferenceElapsed, false, billing);
        LOGGER.info(returnResult.getData());
        return returnResult;
    }

    private static PreProcessingResult getPreProcessingFeatureData(Context  context ,Map<String, Object> originFeatureData) {
        long beginTime = System.currentTimeMillis();
        try {
            return preProcessing.getResult(context ,ObjectTransform.bean2Json(originFeatureData));
        }finally {
            long  endTime =  System.currentTimeMillis();
            LOGGER.info("preprocess caseid {} cost time {}",context.getCaseId(),endTime-beginTime);
        }

    }

    private static PostProcessingResult getPostProcessedResult(Context  context ,Map<String, Object> featureData, Map<String, Object> modelResult) {
        long beginTime = System.currentTimeMillis();
        try {
            return postProcessing.getResult(context,featureData, modelResult);
        }finally {
            long  endTime =  System.currentTimeMillis();
            LOGGER.info("postprocess caseid {} cost time {}",context.getCaseId(),endTime-beginTime);
        }
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

    private static void logInference(Context  context ,InferenceRequest inferenceRequest, ModelNamespaceData modelNamespaceData, ReturnResult inferenceResult, long elapsed, boolean getRemotePartyResult, boolean billing) {
        InferenceUtils.logInference(  context,FederatedInferenceType.INITIATED, modelNamespaceData.getLocal(), modelNamespaceData.getRole(), inferenceRequest.getCaseid(), inferenceRequest.getSeqno(), inferenceResult.getRetcode(), elapsed, getRemotePartyResult, billing, new ObjectMapper().convertValue(inferenceRequest, HashMap.class), inferenceResult);
    }

    private static void logInference(Context  context,Map<String, Object> federatedParams, FederatedParty federatedParty, FederatedRoles federatedRoles, ReturnResult inferenceResult, long elapsed, boolean getRemotePartyResult, boolean billing) {
        InferenceUtils.logInference(context ,FederatedInferenceType.FEDERATED, federatedParty, federatedRoles, federatedParams.get("caseid").toString(), federatedParams.get("seqno").toString(), inferenceResult.getRetcode(), elapsed, getRemotePartyResult, billing, federatedParams, inferenceResult);
    }
}
