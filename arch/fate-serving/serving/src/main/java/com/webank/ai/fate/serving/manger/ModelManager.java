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

import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import java.io.IOException;
import java.util.Map;
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

    public static PipelineTask getModel(String name, String namespace){
        return modelCache.get(ModelUtils.genModelKey(name, namespace));
    }

    public static PipelineTask getModelAcPartner(int partnerPartyId, String partnerModelName, String partnerModelNamespace){
        ModelInfo modelInfo = partnerModelIndex.get(ModelUtils.genPartnerModelIndexKey(partnerPartyId, partnerModelName, partnerModelNamespace));
        LOGGER.info(ModelUtils.genPartnerModelIndexKey(partnerPartyId, partnerModelName, partnerModelNamespace));
        if (modelInfo == null){
            return null;
        }
        return modelCache.get(ModelUtils.genModelKey(modelInfo.getName(), modelInfo.getNamespace()));
    }

    private static PipelineTask pushModelIntoPool(String name, String namespace) throws Exception{
        PipelineTask model = ModelUtils.loadModel(name, namespace);
        if (model == null){
            return null;
        }
        modelCache.put(ModelUtils.genModelKey(name, namespace), model);
        LOGGER.info(modelCache.getSize());
        return model;
    }

    public static ReturnResult publishLoadModel(Map<Integer, ModelServiceProto.ModelInfo> models){
        ReturnResult returnResult = new ReturnResult();
        returnResult.setStatusCode(StatusCode.OK);
        try{
            int partyId = Configuration.getPropertyInt("partyId");
            ModelServiceProto.ModelInfo myModelInfo = models.get(partyId);
            if (myModelInfo == null){
                returnResult.setStatusCode(StatusCode.NOMODEL);
                return returnResult;
            }
            String myModelName = myModelInfo.getName();
            String myModelNamespace = myModelInfo.getNamespace();
            PipelineTask model = pushModelIntoPool(myModelName, myModelNamespace);
            if (model == null){
                returnResult.setStatusCode(StatusCode.RUNTIMEERROR);
                return returnResult;
            }
            models.forEach((p, m)->{
                if (p != partyId){
                    LOGGER.info(ModelUtils.genPartnerModelIndexKey(p, m.getName(), m.getNamespace()));
                    partnerModelIndex.put(ModelUtils.genPartnerModelIndexKey(p, m.getName(), m.getNamespace()), new ModelInfo(myModelName, myModelNamespace));
                }
            });
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
            String name = String.valueOf(requestData.get("modelName"));
            String namespace = String.valueOf(requestData.get("modelNamespace"));
            returnResult.setData("name", name);
            returnResult.setData("namespace", namespace);
            //returnResult.setStatusCode(pushModelIntoPool(name, namespace));
        }
        catch (Exception ex){
            returnResult.setStatusCode(StatusCode.UNKNOWNERROR);
            returnResult.setMessage(ex.getMessage());
        }
        return returnResult;
    }

    public static ReturnResult publishOnlineModel(Map<Integer, ModelServiceProto.ModelInfo> models){
        ReturnResult returnResult = new ReturnResult();
        ModelServiceProto.ModelInfo myModelInfo = models.get(Configuration.getPropertyInt("partyId"));
        if (myModelInfo == null){
            returnResult.setStatusCode(StatusCode.NOMODEL);
            returnResult.setMessage("No model for me.");
            return returnResult;
        }
        PipelineTask model = modelCache.get(ModelUtils.genModelKey(myModelInfo.getName(), myModelInfo.getNamespace()));
        if (model == null){
            returnResult.setStatusCode(StatusCode.NOMODEL);
            returnResult.setMessage("Can not found model by these information.");
            return returnResult;
        }
        try{
            namespaceModel.put(myModelInfo.getNamespace(), model);
            returnResult.setStatusCode(StatusCode.OK);
        }
        catch (Exception ex){
            returnResult.setStatusCode(StatusCode.UNKNOWNERROR);
            returnResult.setMessage(ex.getMessage());
        }
        return returnResult;
    }
}
