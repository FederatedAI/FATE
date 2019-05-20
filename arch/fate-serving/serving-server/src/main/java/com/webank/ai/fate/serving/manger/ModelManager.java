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
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

public class ModelManager {
    private static ModelPool namespaceModel;
    private static ModelCache modelCache;
    private static ConcurrentHashMap<String, ModelInfo> partnerModelIndex;
    private static final Logger LOGGER = LogManager.getLogger();
    public ModelManager(){
        namespaceModel = new ModelPool();
        modelCache = new ModelCache();
        partnerModelIndex = new ConcurrentHashMap<>();
    }

    public static ReturnResult publishLoadModel(String role, int partyId, Map<String, List<Integer>> allParty, Map<String, Map<Integer, ModelInfo>> allPartyModel){
        ReturnResult returnResult = new ReturnResult();
        returnResult.setStatusCode(StatusCode.OK);
        try{
            ModelInfo modelInfo;
            if (allPartyModel.containsKey(role) && allPartyModel.get(role).containsKey(partyId)){
                modelInfo = allPartyModel.get(role).get(partyId);
            }else{
                modelInfo = null;
            }
            if (modelInfo == null){
                returnResult.setStatusCode(StatusCode.NOMODEL);
                return returnResult;
            }
            PipelineTask model = pushModelIntoPool(role, partyId, allParty, modelInfo.getName(), modelInfo.getNamespace());
            if (model == null){
                returnResult.setStatusCode(StatusCode.RUNTIMEERROR);
                return returnResult;
            }
            allPartyModel.forEach((roleName, roleModelInfo)->{
                roleModelInfo.forEach((p, m)->{
                    if (p != partyId){
                        partnerModelIndex.put(ModelUtils.genModelKey(roleName, p, allParty, m.getName(), m.getNamespace()), modelInfo);
                        LOGGER.info("Create model index for partner({}, {})", roleName, p);
                    }
                });
            });
            return returnResult;
        }
        catch (IOException ex){
            LOGGER.error(ex);
            returnResult.setStatusCode(StatusCode.IOERROR);
            returnResult.setError(Optional.ofNullable(ex.getMessage()).orElse(""));
        }
        catch (Exception ex){
            LOGGER.error(ex);
            ex.printStackTrace();
            returnResult.setStatusCode(StatusCode.UNKNOWNERROR);
            returnResult.setError(Optional.ofNullable(ex.getMessage()).orElse(""));
        }
        return returnResult;
    }


    public static ReturnResult publishOnlineModel(String role, int partyId, Map<String, List<Integer>> allParty, Map<String, Map<Integer, ModelInfo>> allPartyModel){
        ReturnResult returnResult = new ReturnResult();
        ModelInfo modelInfo = allPartyModel.get(role).get(partyId);
        if (modelInfo == null){
            returnResult.setStatusCode(StatusCode.NOMODEL);
            returnResult.setMessage("No model for me.");
            return returnResult;
        }
        PipelineTask model = modelCache.get(ModelUtils.genModelKey(role, partyId, allParty, modelInfo.getName(), modelInfo.getNamespace()));
        if (model == null){
            returnResult.setStatusCode(StatusCode.NOMODEL);
            returnResult.setMessage("Can not found model by these information.");
            return returnResult;
        }
        try{
            namespaceModel.put(modelInfo.getNamespace(), model);
            LOGGER.info("Enable model {} for namespace {} success", modelInfo.getName(), modelInfo.getNamespace());
            returnResult.setStatusCode(StatusCode.OK);
        }
        catch (Exception ex){
            returnResult.setStatusCode(StatusCode.UNKNOWNERROR);
            returnResult.setMessage(ex.getMessage());
        }
        return returnResult;
    }

    public static PipelineTask getModel(String role, int partyId, Map<String, List<Integer>> allParty, String name, String namespace){
        return modelCache.get(ModelUtils.genModelKey(role, partyId, allParty, name, namespace));
    }

    public static PipelineTask getModelByPartner(String role, int partyId, String partnerRole, int partnerPartyId, Map<String, List<Integer>> allParty, String partnerModelName, String partnerModelNamespace){
        ModelInfo modelInfo = partnerModelIndex.get(ModelUtils.genModelKey(partnerRole, partnerPartyId, allParty, partnerModelName, partnerModelNamespace));
        if (modelInfo == null){
            return null;
        }
        return modelCache.get(ModelUtils.genModelKey(role, partyId, allParty, modelInfo.getName(), modelInfo.getNamespace()));
    }

    private static PipelineTask pushModelIntoPool(String role, int partyId, Map<String, List<Integer>> allParty, String name, String namespace) throws Exception{
        PipelineTask model = ModelUtils.loadModel(name, namespace);
        if (model == null){
            return null;
        }
        modelCache.put(ModelUtils.genModelKey(role, partyId, allParty, name, namespace), model);
        LOGGER.info("Load model(name: {}, namespace: {}) success, model cache size is {}", name, namespace, modelCache.getSize());
        return model;
    }
}
