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
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.bean.ReturnResult;
import com.webank.ai.fate.serving.bean.ModelNamespaceData;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ModelManager {
    private static Map<Integer, String> inferenceSceneNamespaceMap;
    private static Map<String, ModelNamespaceData> modelNamespaceDataMap;
    private static ReentrantReadWriteMapPool<Integer, String> inferenceSceneNamespaceMapPool;
    private static ReentrantReadWriteMapPool<String, ModelNamespaceData> modelNamespaceDataMapPool;
    private static ModelCache modelCache;
    private static ConcurrentHashMap<String, ModelInfo> partnerModelIndex;
    private static final Logger LOGGER = LogManager.getLogger();

    static {
        inferenceSceneNamespaceMap = new HashMap<>();
        modelNamespaceDataMap = new HashMap<>();
        inferenceSceneNamespaceMapPool = new ReentrantReadWriteMapPool<>(inferenceSceneNamespaceMap);
        modelNamespaceDataMapPool = new ReentrantReadWriteMapPool<>(modelNamespaceDataMap);
        modelCache = new ModelCache();
        partnerModelIndex = new ConcurrentHashMap<>();
    }

    public static ReturnResult publishLoadModel(FederatedParty federatedParty, FederatedRoles federatedRoles, Map<String, Map<Integer, ModelInfo>> federatedRolesModel) {
        String role = federatedParty.getRole();
        int partyId = federatedParty.getPartyId();
        ReturnResult returnResult = new ReturnResult();
        returnResult.setRetcode(StatusCode.OK);
        try {
            ModelInfo modelInfo;
            if (federatedRolesModel.containsKey(role) && federatedRolesModel.get(role).containsKey(partyId)) {
                modelInfo = federatedRolesModel.get(role).get(partyId);
            } else {
                modelInfo = null;
            }
            if (modelInfo == null) {
                returnResult.setRetcode(StatusCode.NOMODEL);
                return returnResult;
            }
            PipelineTask model = pushModelIntoPool(role, partyId, federatedRoles, modelInfo.getName(), modelInfo.getNamespace());
            if (model == null) {
                returnResult.setRetcode(StatusCode.RUNTIMEERROR);
                return returnResult;
            }
            federatedRolesModel.forEach((roleName, roleModelInfo) -> {
                roleModelInfo.forEach((p, m) -> {
                    if (p != partyId) {
                        String partnerModelKey = ModelUtils.genModelKey(roleName, p, federatedRoles, m.getName(), m.getNamespace());
                        partnerModelIndex.put(partnerModelKey, modelInfo);
                        LOGGER.info("Create model index({}) for partner({}, {})", partnerModelKey, roleName, p);
                    }
                });
            });
            return returnResult;
        } catch (IOException ex) {
            LOGGER.error(ex);
            returnResult.setRetcode(StatusCode.IOERROR);
        } catch (Exception ex) {
            LOGGER.error(ex);
            ex.printStackTrace();
            returnResult.setRetcode(StatusCode.UNKNOWNERROR);
        }
        return returnResult;
    }


    public static ReturnResult publishOnlineModel(FederatedParty federatedParty, FederatedRoles federatedRoles, Map<String, Map<Integer, ModelInfo>> federatedRolesModel, int sceneId) {
        String role = federatedParty.getRole();
        int partyId = federatedParty.getPartyId();
        ReturnResult returnResult = new ReturnResult();
        ModelInfo modelInfo = federatedRolesModel.get(role).get(partyId);
        if (modelInfo == null) {
            returnResult.setRetcode(StatusCode.NOMODEL);
            returnResult.setRetmsg("No model for me.");
            return returnResult;
        }
        PipelineTask model = modelCache.get(ModelUtils.genModelKey(role, partyId, federatedRoles, modelInfo.getName(), modelInfo.getNamespace()));
        if (model == null) {
            returnResult.setRetcode(StatusCode.NOMODEL);
            returnResult.setRetmsg("Can not found model by these information.");
            return returnResult;
        }
        try {
            String modelNamespace = modelInfo.getNamespace();
            String modelName = modelInfo.getName();
            modelNamespaceDataMapPool.put(modelNamespace, new ModelNamespaceData(modelNamespace, federatedParty, federatedRoles, modelName, model));
            inferenceSceneNamespaceMapPool.put(sceneId, modelNamespace);
            LOGGER.info("Enable model {} for namespace {} success", modelName, modelNamespace);
            LOGGER.info("Get namespace {} for scene {}", modelNamespace, sceneId);
            returnResult.setRetcode(StatusCode.OK);
        } catch (Exception ex) {
            returnResult.setRetcode(StatusCode.UNKNOWNERROR);
            returnResult.setRetmsg(ex.getMessage());
        }
        return returnResult;
    }

    public static PipelineTask getModel(String role, int partyId, FederatedRoles federatedRoles, String name, String namespace) {
        return modelCache.get(ModelUtils.genModelKey(role, partyId, federatedRoles, name, namespace));
    }

    public static ModelNamespaceData getModelNamespaceData(String namespace) {
        return modelNamespaceDataMapPool.get(namespace);
    }

    public static String getNamespaceBySceneId(int sceneId) {
        return inferenceSceneNamespaceMapPool.get(sceneId);
    }


    public static PipelineTask getModelByPartner(String role, int partyId, String partnerRole, int partnerPartyId, FederatedRoles federatedRoles, String partnerModelName, String partnerModelNamespace) {
        LOGGER.info(ModelUtils.genModelKey(partnerRole, partnerPartyId, federatedRoles, partnerModelName, partnerModelNamespace));
        ModelInfo modelInfo = partnerModelIndex.get(ModelUtils.genModelKey(partnerRole, partnerPartyId, federatedRoles, partnerModelName, partnerModelNamespace));
        if (modelInfo == null) {
            return null;
        }
        return modelCache.get(ModelUtils.genModelKey(role, partyId, federatedRoles, modelInfo.getName(), modelInfo.getNamespace()));
    }

    private static PipelineTask pushModelIntoPool(String role, int partyId, FederatedRoles federatedRoles, String name, String namespace) throws Exception {
        PipelineTask model = ModelUtils.loadModel(name, namespace);
        if (model == null) {
            return null;
        }
        modelCache.put(ModelUtils.genModelKey(role, partyId, federatedRoles, name, namespace), model);
        LOGGER.info("Load model success, name: {}, namespace: {}, model cache size is {}", name, namespace, modelCache.getSize());
        return model;
    }
}
