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
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.storage.dtable.DTableInfo;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.adapter.dataaccess.FeatureData;
import com.webank.ai.fate.serving.adapter.resultprocessing.ResultData;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.serving.utils.DTableUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class InferenceManager {
    private static final Logger LOGGER = LogManager.getLogger();
    public static ReturnResult inference(String role, int partyId, Map<String, List<Integer>> allParty, ModelInfo modelInfo, String data, int sceneId){
        ReturnResult returnResult = new ReturnResult();
        DTableInfo modelDTableInfo = DTableUtils.genTableInfo(modelInfo.getName(),
                modelInfo.getNamespace(),
                sceneId,
                role,
                partyId,
                allParty,
                "model");
        String modelTableName = modelDTableInfo.getName();
        String modelNamespace = modelDTableInfo.getNamespace();
        LOGGER.info("use model(name: {}, namespace: {})", modelTableName, modelNamespace);
        PipelineTask model = ModelManager.getModel(role, partyId, allParty, modelTableName, modelNamespace);
        if (model == null){
            returnResult.setStatusCode(StatusCode.NOMODEL);
            return returnResult;
        }

        Object inputObject = ObjectTransform.json2Bean(data, HashMap.class);
        if (inputObject == null){
            returnResult.setStatusCode(StatusCode.ILLEGALDATA);
            returnResult.setMessage("Can not parse data json.");
            return returnResult;
        }
        Map<String, Object> inputData = (Map<String, Object>) inputObject;
        inputData.forEach((sid, f)->{
            Map<String, Object> predictInputData = (Map<String, Object>)f;
            Map<String, Object> predictParams = new HashMap<>();
            predictParams.put("sid", sid);
            predictParams.put("partyId", partyId);
            predictParams.put("role", role);
            predictParams.put("allParty", allParty);
            predictParams.put("modelName", modelTableName);
            predictParams.put("modelNamespace", modelNamespace);
            Map<String, Object> modelResult = model.predict(predictInputData, predictParams);
            Map<String, Object> result = getProcessedResult(modelResult);
            returnResult.putAllData(result);
        });
        LOGGER.info("Inference successfully.");
        return returnResult;
    }

    public static ReturnResult federatedPredict(Map<String, Object> requestData){
        ReturnResult returnResult = new ReturnResult();
        int partyId = (int)requestData.get("partyId");
        String role = requestData.get("role").toString();
        int partnerPartyId = (int)requestData.get("partnerPartyId");
        String partnerRole = requestData.get("partnerRole").toString();
        Map<String, List<Integer>> allParty = (Map<String, List<Integer>>)requestData.get("allParty");
        String partnerModelName = requestData.get("partnerModelName").toString();
        String partnerModelNamespace = requestData.get("partnerModelNamespace").toString();
        PipelineTask model = ModelManager.getModelByPartner(role, partyId, partnerRole, partnerPartyId, allParty, partnerModelName, partnerModelNamespace);
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
            LOGGER.error(ex.getStackTrace());
            returnResult.setStatusCode(StatusCode.FEDERATEDERROR);
            returnResult.setMessage(ex.getMessage());
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

    private static Map<String, Object> getFeatureData(String sid){
        String classPath = FeatureData.class.getPackage().getName() + "." + Configuration.getProperty("OnlineDataAccessAdapter");
        FeatureData featureData = (FeatureData) getClassByName(classPath);
        if (featureData == null){
            return null;
        }
        return featureData.getData(sid);
    }
}
