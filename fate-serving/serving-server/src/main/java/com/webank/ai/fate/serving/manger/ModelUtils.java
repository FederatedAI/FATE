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
import com.webank.ai.fate.core.bean.FederatedRoles;
import com.webank.ai.fate.core.storage.dtable.DTable;
import com.webank.ai.fate.core.storage.dtable.DTableFactory;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class ModelUtils {
    private static final Logger LOGGER = LogManager.getLogger();
    private static final String modelKeySeparator = "&";

    public static Map<String, byte[]> readModel(String name, String namespace) {
        LOGGER.info("read model, name: {} namespace: {}", name, namespace);
        DTable dataTable = DTableFactory.getDTable(name, namespace, 1);
        return dataTable.collect();
    }

    public static PipelineTask loadModel(String name, String namespace) {
        Map<String, byte[]> modelBytes = readModel(name, namespace);
        if (modelBytes == null || modelBytes.size() == 0) {
            return null;
        }
        PipelineTask pipelineTask = new PipelineTask();
        pipelineTask.initModel(modelBytes);
        return pipelineTask;
    }

    public static String genModelKey(String name, String namespace) {
        return StringUtils.join(Arrays.asList(name, namespace), modelKeySeparator);
    }

    public static String[] splitModelKey(String key) {
        return StringUtils.split(key, modelKeySeparator);
    }


    public static FederatedRoles getFederatedRoles(Map<String, ModelServiceProto.Party> federatedRolesProto) {
        FederatedRoles federatedRoles = new FederatedRoles();
        federatedRolesProto.forEach((roleName, party) -> {
            federatedRoles.setRole(roleName, party.getPartyIdList());
        });
        return federatedRoles;
    }

    public static Map<String, Map<String, ModelInfo>> getFederatedRolesModel(Map<String, ModelServiceProto.RoleModelInfo> federatedRolesModelProto) {
        Map<String, Map<String, ModelInfo>> federatedRolesModel = new HashMap<>();
        federatedRolesModelProto.forEach((roleName, roleModelInfo) -> {
            federatedRolesModel.put(roleName, new HashMap<>());
            roleModelInfo.getRoleModelInfoMap().forEach((partyId, modelInfo) -> {
                federatedRolesModel.get(roleName).put(partyId, new ModelInfo(modelInfo.getTableName(), modelInfo.getNamespace()));
            });
        });
        return federatedRolesModel;
    }
}
