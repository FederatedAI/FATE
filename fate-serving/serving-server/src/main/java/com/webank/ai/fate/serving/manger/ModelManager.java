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
import com.webank.ai.fate.serving.bean.ModelNamespaceData;
import com.webank.ai.fate.serving.core.constant.InferenceRetCode;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ModelManager {
    private static Map<String, String> appNamespaceMap;
    private static Map<String, ModelNamespaceData> modelNamespaceDataMap;
    private static ReentrantReadWriteMapPool<String, String> appNamespaceMapPool;
    private static ReentrantReadWriteMapPool<String, ModelNamespaceData> modelNamespaceDataMapPool;
    private static ModelCache modelCache;
    private static ConcurrentHashMap<String, ModelInfo> partnerModelIndex;
    private static final Logger LOGGER = LogManager.getLogger();

    static {
        appNamespaceMap = new HashMap<>();
        modelNamespaceDataMap = new HashMap<>();
        appNamespaceMapPool = new ReentrantReadWriteMapPool<>(appNamespaceMap);
        modelNamespaceDataMapPool = new ReentrantReadWriteMapPool<>(modelNamespaceDataMap);
        modelCache = new ModelCache();
        partnerModelIndex = new ConcurrentHashMap<>();
    }

    public static ReturnResult publishLoadModel(FederatedParty federatedParty, FederatedRoles federatedRoles, Map<String, Map<String, ModelInfo>> federatedRolesModel) {
        String role = federatedParty.getRole();
        String partyId = federatedParty.getPartyId();
        ReturnResult returnResult = new ReturnResult();
        returnResult.setRetcode(InferenceRetCode.OK);
        try {
            ModelInfo modelInfo;
            if (federatedRolesModel.containsKey(role) && federatedRolesModel.get(role).containsKey(partyId)) {
                modelInfo = federatedRolesModel.get(role).get(partyId);
            } else {
                modelInfo = null;
            }
            if (modelInfo == null) {
                returnResult.setRetcode(InferenceRetCode.LOAD_MODEL_FAILED);
                return returnResult;
            }
            PipelineTask model = pushModelIntoPool(modelInfo.getName(), modelInfo.getNamespace());
            if (model == null) {
                returnResult.setRetcode(InferenceRetCode.LOAD_MODEL_FAILED);
                return returnResult;
            }
            federatedRolesModel.forEach((roleName, roleModelInfo) -> {
                roleModelInfo.forEach((p, m) -> {
                    if (!p.equals(partyId)||(p.equals(partyId)&&!role.equals(roleName))) {
                        String partnerModelKey = ModelUtils.genModelKey(m.getName(), m.getNamespace());
                        partnerModelIndex.put(partnerModelKey, modelInfo);
                        LOGGER.info("Create model index({}) for partner({}, {})", partnerModelKey, roleName, p);
                    }
                });
            });
            LOGGER.info("load the model successfully");
            return returnResult;
        } catch (Exception ex) {
            LOGGER.error(ex);
            ex.printStackTrace();
            returnResult.setRetcode(InferenceRetCode.SYSTEM_ERROR);
        }
        return returnResult;
    }


    public static ReturnResult publishOnlineModel(FederatedParty federatedParty, FederatedRoles federatedRoles, Map<String, Map<String, ModelInfo>> federatedRolesModel) {
        String role = federatedParty.getRole();
        String partyId = federatedParty.getPartyId();
        ReturnResult returnResult = new ReturnResult();
        ModelInfo modelInfo = federatedRolesModel.get(role).get(partyId);
        if (modelInfo == null) {
            returnResult.setRetcode(InferenceRetCode.LOAD_MODEL_FAILED);
            returnResult.setRetmsg("No model for me.");
            return returnResult;
        }
        PipelineTask model = modelCache.get(ModelUtils.genModelKey(modelInfo.getName(), modelInfo.getNamespace()));
        if (model == null) {
            returnResult.setRetcode(InferenceRetCode.LOAD_MODEL_FAILED);
            returnResult.setRetmsg("Can not found model by these information.");
            return returnResult;
        }
        try {
            String modelNamespace = modelInfo.getNamespace();
            String modelName = modelInfo.getName();
            modelNamespaceDataMapPool.put(modelNamespace, new ModelNamespaceData(modelNamespace, federatedParty, federatedRoles, modelName, model));
            appNamespaceMapPool.put(partyId, modelNamespace);
            LOGGER.info("Enable model {} for namespace {} success", modelName, modelNamespace);
            LOGGER.info("Get model namespace {} for app {}", modelNamespace, partyId);
            returnResult.setRetcode(InferenceRetCode.OK);
        } catch (Exception ex) {
            returnResult.setRetcode(InferenceRetCode.SYSTEM_ERROR);
            returnResult.setRetmsg(ex.getMessage());
        }
        return returnResult;
    }

    public static PipelineTask getModel(String name, String namespace) {
        return modelCache.get(ModelUtils.genModelKey(name, namespace));
    }

    public static ModelNamespaceData getModelNamespaceData(String namespace) {
        return modelNamespaceDataMapPool.get(namespace);
    }

    public static String getModelNamespaceByPartyId(String partyId) {
        return appNamespaceMapPool.get(partyId);
    }


    public static ModelInfo getModelInfoByPartner(String partnerModelName, String partnerModelNamespace) {
        return partnerModelIndex.get(ModelUtils.genModelKey(partnerModelName, partnerModelNamespace));
    }

    private static PipelineTask pushModelIntoPool(String name, String namespace) {
        PipelineTask model = ModelUtils.loadModel(name, namespace);
        if (model == null) {
            return null;
        }
        modelCache.put(ModelUtils.genModelKey(name, namespace), model);
        LOGGER.info("Load model success, name: {}, namespace: {}, model cache size is {}", name, namespace, modelCache.getSize());
        return model;
    }
}
